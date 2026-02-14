import torch
from torch import nn, Tensor
from model.deeplabv3 import deeplabv3_mobilenet_v3_large
import torch.nn.functional as F
from defaults import _C as cfg
from torchinfo import summary
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
import math
from torchvision.ops.misc import Conv2dNormActivation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class DAS_SKConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super(DAS_SKConv, self).__init__()

        out = math.ceil(out_channels/2)
        # Atrous Separable Convolution
        self.f_da = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=in_channels, bias=False)
        self.f_p = nn.Conv2d(in_channels, out, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.relu1 = nn.ReLU()

        # Atrous Convolution
        self.f_a = nn.Conv2d(in_channels, out, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.relu2 = nn.ReLU()
        
        # Selective Kernel Networks
        self.f_attn = SKConv(features=out*2,   # Matches input channels
                            WH=32,        # Since input is 128x128
                            M=2,           # Two branches (e.g., dilation=1 and 2)
                            G=8,           # Number of groups for grouped convolution → Should divide features evenly. 192 is divisible by G = 8, 12, 16, 24, 32, 48, 64, 96 etc.
                            r=16,          # Reduction ratio
                            stride=1,      # Keep 1 unless you're downsampling
                            L=32           # Minimum dim in FC layer (default)
                            )  

    def forward(self, x) -> Tensor:
        # Atrous Separable Convolution
        O_da = self.f_da(x)
        O_p = self.f_p (O_da)
        O_p = self.bn1(O_p)
        O_p = self.relu1(O_p)

        # Atrous Convolution
        O_a = self.f_a(x)
        O_a = self.bn2(O_a)
        O_a = self.relu2(O_a)     

        # DAS_conv output
        O_das = torch.cat([O_p, O_a], dim = 1)
        O_attn = self.f_attn (O_das)
        
        return O_attn


# “Selective Kernel Networks” (CVPR 2019)     
# Paper: https://arxiv.org/abs/1903.06586
# SKNet dynamically adjusts the receptive field by selecting between multiple convolutional kernels 
# (typically with different dilation rates or kernel sizes), using an attention mechanism.
# 🔧 Core Features:
#     ~ Multiple parallel convolution branches (e.g., 3×3, 5×5, etc.)
#     ~ Fuse their outputs, then use a soft attention mechanism to weight each branch.
#     ~ Learns to adaptively select the best kernel size per instance.
class SKConv(nn.Module):
    def __init__(self, features, WH, M=2, G=32, r=16, stride=1, L=32):
        """
        SKConv Module:
        :param features: Number of input and output channels
        :param WH: Input spatial size (for calculating the FC layer size)
        :param M: Number of convolutional branches (e.g., 2: 3x3 and 5x5)
        :param G: Number of groups in group conv
        :param r: Reduction ratio
        :param stride: Stride in convolution
        :param L: Minimum dimensionality in FC (default from paper)
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList()
        
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=stride,
                              padding=1 + i, dilation=1 + i, groups=G, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True)
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(features, d, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(d, features * M, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x).unsqueeze(dim=1) for conv in self.convs]  # Each shape: [B, 1, C, H, W]
        feats = torch.cat(feats, dim=1)                            # Shape: [B, M, C, H, W]
        U = torch.sum(feats, dim=1)                                # Fuse: [B, C, H, W]

        # Channel-wise attention
        s = self.global_pool(U)                                    # [B, C, 1, 1]
        z = self.fc1(s)                                            # [B, d, 1, 1]
        z = self.bn(z)
        z = self.relu(z)
        a_b = self.fc2(z)                                          # [B, M*C, 1, 1]
        a_b = a_b.view(batch_size, self.M, self.features, 1, 1)    # [B, M, C, 1, 1]
        a_b = self.softmax(a_b)                                    # Softmax over M

        # Attention-weighted sum
        out = torch.sum(feats * a_b, dim=1)                        # [B, C, H, W]
        return out


class HorizontalStripPooling(nn.Module):
    def __init__(self):
        super(HorizontalStripPooling, self).__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        # Pool along height (H), keep width (W)
        pooled = F.adaptive_avg_pool2d(x, (1, x.size(3)))  # (B, C, 1, W)
        out = pooled.expand_as(x)  # Broadcast to (B, C, H, W)
        return out
       
class VerticalStripPooling(nn.Module):
    def __init__(self):
        super(VerticalStripPooling, self).__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        # Pool along width (W), keep height (H)
        pooled = F.adaptive_avg_pool2d(x, (x.size(2), 1))  # (B, C, H, 1)
        out = pooled.expand_as(x)  # Broadcast to (B, C, H, W)
        return out      
    
class StripPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(StripPooling, self).__init__()
        self.strip_pooling = nn.ModuleList([HorizontalStripPooling(), VerticalStripPooling()])
        self.strip_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)
        self.BN = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        hsp_out = self.strip_pooling[0](x)  # Horizontal
        vsp_out = self.strip_pooling[1](x)  # Vertical
        
        # Combine (sum or concatenate)
        # strip_out = hsp_out + vsp_out  
        strip_out = torch.cat([hsp_out, vsp_out], dim=1)
        # Reduce channels from 960 to 256
        strip_out = self.strip_conv(strip_out)
        strip_out = self.BN(strip_out)
        strip_out = self.ReLU(strip_out)

        return strip_out

class ParallelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.sep_conv = SeparableConv2d(in_channels, out_channels//2, kernel_size, stride, padding)
        self.norm_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        out1 = self.sep_conv(x)
        out2 = self.norm_conv(x)
        # Concatenate along channel dimension
        out = torch.cat([out1, out2], dim=1)
        return out
    
     

class CustomDeepLabV3(nn.Module):
    def __init__(self, cfg):
        super(CustomDeepLabV3, self).__init__()
        
        self.aux_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)  
        self.aux_backbone = nn.Sequential(*list(self.aux_backbone.features.children())[:6])
        self.aux_backbone.add_module('conv_act', Conv2dNormActivation(in_channels=136, out_channels=480, kernel_size=1,))     
        # self.aux_backbone[0][0]=Conv2d(3, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)     
             
             
        self.model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')     
        self.backbone = self.model.backbone
        self.backbone['16'] = Conv2dNormActivation(in_channels=160, out_channels=480, kernel_size=1, activation_layer=nn.Hardswish)
        
        
        self.classifier = self.model.classifier
             
        if len(cfg.CHANNELS) > 3:              
            weight = self.backbone['0'][0].weight.clone()
            self.backbone['0'][0] = nn.Conv2d(len(cfg.CHANNELS), 16, kernel_size=3, stride=2, padding=1, bias=False)
            with torch.no_grad():
                if cfg.CHANNELS == 'nrgb' or 'rngb':
                    self.backbone['0'][0].weight[:, 0] = weight[:, 0] # r
                    self.backbone['0'][0].weight[:, 1] = weight[:, 0] # r
                    self.backbone['0'][0].weight[:, 2] = weight[:, 1] # g 
                    self.backbone['0'][0].weight[:, 3] = weight[:, 2] # b
                elif cfg.CHANNELS == 'rgbn':
                    self.backbone['0'][0].weight[:, 0] = weight[:, 0]
                    self.backbone['0'][0].weight[:, 1] = weight[:, 1]
                    self.backbone['0'][0].weight[:, 2] = weight[:, 2]
                    self.backbone['0'][0].weight[:, 3] = weight[:, 0]
                    
            aux_weight = self.aux_backbone[0][0].weight.clone()
            self.aux_backbone[0][0] = nn.Conv2d(len(cfg.CHANNELS), 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            with torch.no_grad():
                if cfg.CHANNELS == 'nrgb' or 'rngb':
                    self.aux_backbone[0][0].weight[:, 0] = aux_weight[:, 0] # r
                    self.aux_backbone[0][0].weight[:, 1] = aux_weight[:, 0] # r
                    self.aux_backbone[0][0].weight[:, 2] = aux_weight[:, 1] # g
                    self.aux_backbone[0][0].weight[:, 3] = aux_weight[:, 2] # b
                elif cfg.CHANNELS == 'rgbn':
                    self.aux_backbone[0][0].weight[:, 0] = aux_weight[:, 0] 
                    self.aux_backbone[0][0].weight[:, 1] = aux_weight[:, 1] 
                    self.aux_backbone[0][0].weight[:, 2] = aux_weight[:, 2] 
                    self.aux_backbone[0][0].weight[:, 3] = aux_weight[:, 0]            
        
        '''ASPP block'''
        num = int(768/6)
        self.classifier[0].convs[1] = DAS_SKConv(960, num, dilation = 4)
        self.classifier[0].convs[2] = DAS_SKConv(960, num, dilation = 8)
        self.classifier[0].convs[3] = DAS_SKConv(960, num, dilation = 12)
        self.classifier[0].convs.add_module('DAS_SKConv_D22', DAS_SKConv(in_channels=960, out_channels=num, dilation = 22))
        
        # additional DAS_Conv layer with dilation 18
        self.classifier[0].convs.add_module('DAS_SKConv_D18', DAS_SKConv(in_channels=960, out_channels=num, dilation = 18))
        self.classifier[0].convs.add_module('DAS_SKConv_D26', DAS_SKConv(in_channels=960, out_channels=num, dilation = 26))
        
        # replace ASPPPooling with StripPooling
        self.classifier[0].convs[4] = StripPooling(in_channels=960, out_channels=256) 
        
        # self.classifier[0].project[0] = nn.Conv2d(1286, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        # self.classifier[1] = SeparableConv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.classifier[1] = ParallelConvBlock(256, 256, kernel_size=3, stride=1, padding=1)        
        
        self.classifier[4] = SeparableConv2d(344, 148, kernel_size = 3, stride = 1, padding = 1)
        # self.classifier.add_module('4', SeparableConv2d(344, 148, kernel_size = 3, stride = 1, padding = 1))
        self.classifier.add_module('5', nn.BatchNorm2d(148, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.classifier.add_module('6', nn.ReLU())
        
        
        self.classifier.add_module('7', SeparableConv2d(204, 148, kernel_size = 3, stride = 1, padding = 1))
        self.classifier.add_module('8', nn.BatchNorm2d(148, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.classifier.add_module('9', nn.ReLU())
        
        
        # # Since we have 9 classes, change the final layer of the classifier
        self.classifier.add_module('10', nn.Conv2d(164, cfg.CLASSES, kernel_size=(1, 1), stride=(1, 1)))
        # self.classifier[4] = nn.Conv2d(296, len(cfg.CHaANNELS), kernel_size=(1, 1), stride=(1, 1))
         
        
    def forward(self, x) -> Tensor:
        input_shape = x.shape[-2:]
        
        a1= self.aux_backbone[0](x)    # 40, 256, 256
        a2 = self.aux_backbone[1](a1)  # 24, 256, 256
        a3 = self.aux_backbone[2](a2)  # 32, 128, 128
        a4 = self.aux_backbone[3](a3)  # 48, 64, 64
        a5 = self.aux_backbone[4](a4)  # 96, 32, 32
        a6 = self.aux_backbone[5](a5)  # 136, 32, 32
        a7 = self.aux_backbone[6](a6)  # 480, 32, 32
        
        
        b1 = self.backbone['0'] (x)
        b2 = self.backbone['1'] (b1) # 1, 16, 256, 256
        b3 = self.backbone['2'] (b2)
        b4 = self.backbone['3'] (b3) # 1, 24, 128, 128
        b5 = self.backbone['4'] (b4)
        
        b6 = self.backbone['5'] (b5)
        b7 = self.backbone['6'] (b6)  # 1, 40, 64, 64
        b8 = self.backbone['7'] (b7)
        b9 = self.backbone['8'] (b8)
        b10 = self.backbone['9'] (b9)
        
        b11 = self.backbone['10'] (b10)
        b12 = self.backbone['11'] (b11)
        b13 = self.backbone['12'] (b12)
        b14 = self.backbone['13'] (b13)
        b15 = self.backbone['14'] (b14) 
        
        b16 = self.backbone['15'] (b15)       
        
        b17 = self.backbone['16'] (b16) 
        
        b17_aux = torch.cat([b17, a7], dim=1)  # Concatenate auxiliary features
          
        c1 = self.classifier[0](b17_aux) # ASPP block       
        
        c2 = self.classifier[1](c1) # change to SeparableConv2d, original is nn.Conv2d(256, 256, 3, padding=1, bias=False)        
        c3 = self.classifier[2](c2) # nn.BatchNorm2d(256)
        c4 = self.classifier[3](c3) # nn.ReLU()
        
        cc4 = F.interpolate(c4, size=128, mode="bilinear", align_corners=False)
        cc5 = torch.cat([cc4, b7, a4], dim = 1) # New
        
        c5 = self.classifier[4](cc5) # SeparableConv2d
        c6 = self.classifier[5](c5)  # nn.BatchNorm2d(256)
        c7 = self.classifier[6](c6)  # nn.ReLU()
        
        cc7 = F.interpolate(c7, size=256, mode="bilinear", align_corners=False)
        cc8 = torch.cat([cc7, b4, a3.clone()], dim = 1) # [204, 128, 128] where cc7(148), b4(24), a3(32)
        
        c8 = self.classifier[7](cc8) # new SeparableConv2d
        c9 = self.classifier[8](c8)  # nn.BatchNorm2d(256)
        c10 = self.classifier[9](c9)  # nn.ReLU()
        
        cc10 = F.interpolate(c10, size=512, mode="bilinear", align_corners=False)
        cc11 = torch.cat([cc10, b2], dim = 1) # [164, 256, 256] where cc10(148), b2(16)
     
        c11 = self.classifier[10](cc11) # nn.Conv2d(256, num_classes, 1) 

        x = F.interpolate(c11, size=input_shape, mode="bilinear", align_corners=False)
        
        return x  # LC55.


    
    
if __name__ == "__main__":
    model = CustomDeepLabV3(cfg).to(device)

    # print(model)
    
    # https://medium.com/the-owl/how-to-get-model-summary-in-pytorch-57db7824d1e3
    summary(model, 
            input_size=(len(cfg.CHANNELS), 1024, 1024), batch_dim = 0,
            col_names=["input_size", "output_size", "num_params", 'trainable', 'kernel_size','mult_adds'],
            verbose=1,
            col_width=16,
            row_settings=["var_names"],
            )
    cfg.BATCH_SIZE = 2
    
    # input = torch.randn(cfg.BATCH_SIZE, len(cfg.CHANNELS), 1024, 1024).to(device='cuda')
    # output = model(input)
    
    # print(f'output shape: {output.shape}')