import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as transforms  
import albumentations as AA
from defaults import _C as cfg
import random
import torchvision.utils as vutils



class BaseDataset(Dataset):
    def __init__(self, img_directory, channels, cfg, mode=''):
        
        self.img_directory = f'{img_directory}{mode}'
        self.channels = channels
        self.mode = mode
        self.cfg = cfg
        
        self.list_samples = [os.path.basename(x) for x in glob.glob(f'{self.img_directory}/images/*.png')]

        # self.patch_size = 512
        # self.samples = []

        # for idx, ID in enumerate(self.list_samples):
        #     img = Image.open(f'{self.img_directory}/images/{ID}').convert('RGB')
        #     W, H = img.size
        #     for top in range(0, H, self.patch_size):
        #         for left in range(0, W, self.patch_size):
        #             self.samples.append((idx, top, left))
                           
        self.classes = ['background',
                        'crop',
                        'weed',
                        'partial-crop',
                        'partial-weed',
                        ]
        cfg.classes = self.classes
        
        self.classes_colour = [[0, 0, 0], # blackground    
                               [144, 238, 144],  # crop           
                               [255, 71, 76],  # weed                    
                               [255, 255, 0],  # partial-crop         
                               [255, 105, 180],  # partial-weed                  
                               ]   
        cfg.class_color = self.classes_colour
        
       
        self.aug_train = AA.Compose([
            # Geometric transforms
            AA.RandomRotate90(p=0.5),
            AA.HorizontalFlip(p=0.5),
            AA.VerticalFlip(p=0.5),

            # Photometric transforms           
            AA.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            AA.RandomBrightnessContrast(p=0.5),
            AA.GaussianBlur(p=0.3),

            # Normalization (always last step)
            AA.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255),                  
        
        ], additional_targets={'mask': 'mask'})
        
        
        self.aug_test = AA.Compose([
            AA.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255),    
        ])
        
    def __getitem__(self, index):
        
        rgb = Image.open(f"{self.img_directory}/images/{self.list_samples[index]}").convert('RGB')
        rgb = np.array(rgb)
        
        if self.mode != 'test':
            label = Image.open(f"{self.img_directory}/semantics/{self.list_samples[index]}")       
            label = np.array(label)
 
        if self.mode == 'train':
            augmented = self.aug_train(image=rgb, mask=label)
            img = augmented['image']
            label = augmented['mask']
        else: 
            augmented = self.aug_test(image=rgb)
            img = augmented['image'] 
                   
        img = torch.from_numpy(img).permute((2,0,1))   
        rgb = torch.from_numpy(rgb).permute((2,0,1)) 
        
        ID = f'{self.list_samples[index]}'

        if self.mode != 'test':
            label = torch.from_numpy(label)
        
            return img, rgb, label.long(), ID
        else:
            return img, rgb, ID
    
    
    def __len__(self):
        return len(self.list_samples)
    
    
    
if __name__ == "__main__":    
    
    for i in ['train', 'val', 'test']:
        data = BaseDataset(img_directory=cfg.IMG_PATH, 
                                channels = cfg.CHANNELS,
                                mode =i,
                                cfg = cfg,
                                )   
        print(f"Dataset {i} length: {len(data)}")
        
        cfg.BATCH_SIZE = 4
        dataloader = DataLoader(dataset=data, 
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True, )
        
        if i != 'test':
            for batch, (img, rgb, label, ID) in enumerate(dataloader):
                if batch == 0:  # Print only the first batch for brevity
                    print(f"Batch {batch}:")
                    print(f"  img: {img.shape}")
                    print(f"  rgb: {rgb.shape}")
                    print(f"  label: {label.shape}")
                    print(f"  ID: {ID}\n")
                if batch == 1:  # Stop after the first two batches
                    break
        else:
            for batch, (img, rgb, ID) in enumerate(dataloader):
                if batch == 0:  # Print only the first batch for brevity
                    print(f"Batch {batch}:")
                    print(f"  img: {img.shape}")
                    print(f"  rgb: {rgb.shape}")
                    print(f"  ID: {ID}\n")
                if batch == 1:  # Stop after the first two batches
                    break
            
