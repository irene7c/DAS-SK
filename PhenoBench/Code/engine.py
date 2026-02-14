import torch
import numpy as np
import datetime
from tqdm.auto import tqdm
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from utils import *
from collections import OrderedDict
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix,  MulticlassRecall

# from torch.utils.tensorboard import SummaryWriter   # in terminal, 'pytorch tensorboard --logdir=runs'
from segmentation_models_pytorch.losses import LovaszLoss, DiceLoss, FocalLoss


def composite_loss(pred, target, cfg):
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(mode ='multiclass', from_logits=True)
    focal_loss_fn = FocalLoss(mode='multiclass', gamma=2.0)
    lovasz_loss_fn = LovaszLoss(mode='multiclass', per_image=True, from_logits=True)

    loss_ce = ce_loss_fn(pred, target)
    loss_dice = dice_loss_fn(pred, target)
    loss_focal = focal_loss_fn(pred, target)
    loss_lovasz = lovasz_loss_fn(pred, target)
    
    loss = cfg.w_ce* loss_ce + \
            cfg.w_dice * loss_dice + \
            cfg.w_focal * loss_focal + \
            cfg.w_lovasz * loss_lovasz
    return loss


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,  
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               cfg,
               jaccard,
               loss_fn: Optional[torch.nn.Module] = None,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
               ):
    
    train_start_time = datetime.datetime.now()
    
    total_loss = 0
    total_batch = len(dataloader)
    total_correct = 0
    total_pixels = 0
    jaccard.reset()

    # Put model in train mode
    model.train()
    for batch, (img, rgb, label, ID) in enumerate(dataloader):
        
        img, label = img.to(device), label.to(device)  
        model.to(device)       
            
        # 1. Forward pass
        pred = model(img)
        if cfg.LOSS_FUNCTION == 'composite_loss':
            sum_loss = composite_loss(pred, label, cfg)
        else:
            sum_loss = loss_fn(pred, label)

        pred = torch.nn.Softmax(dim=1)(pred)    
        pred = pred.argmax(dim=1) 
        jaccard.update(pred, label)
        
        total_loss += sum_loss.item()
        
        # For accuracy calculation
        correct = (pred == label).sum().item()
        pixels = label.numel()
        total_correct += correct
        total_pixels += pixels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        sum_loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
    if cfg.TRAIN_SCHEDULER == 'yes':  # This scheduler must only update after an epoch
        scheduler.step()       
            
    miou = torch.nanmean(jaccard.compute()).item()
    epoch_acc = total_correct / total_pixels
    
    # Saving model at last epoch
    tmp_path = f"{cfg.SAVE_DIR}/temp.pth"
    model_path_dir = f"{cfg.SAVE_DIR}/model by epoch.pth"
    checkpoint = {
        'epoch' : epoch,
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict() if scheduler is not None else None,
        }
    torch.save(checkpoint, tmp_path)  # Save to temp file
    os.replace(tmp_path, model_path_dir)

    train_end_time = datetime.datetime.now()
    train_duration = train_end_time - train_start_time
       
    return model, optimizer,  scheduler, total_loss/total_batch, miou, epoch_acc, train_duration


def train(model: torch.nn.Module, 
          train_dataloader : torch.utils.data.DataLoader, 
          val_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          device : torch.device,
          cfg,
          loss_fn : Optional[torch.nn.Module] = None,
          test_dataloader : Optional[torch.utils.data.DataLoader]=None,
          ) -> None:
    
    best_epoch = 0
    
    if cfg.TRAIN_SCHEDULER == 'yes':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=0.001) 
    else: 
        scheduler = None
        
    if cfg.RESUME_TRAINING:
        checkpoint_ = torch.load(f"{cfg.SAVE_DIR}/model by epoch.pth", weights_only=False)
        model.load_state_dict(checkpoint_['model'])
        optimizer.load_state_dict(checkpoint_['optimizer'])
        if cfg.TRAIN_SCHEDULER == 'yes':
            scheduler.load_state_dict(checkpoint_['scheduler'])
        cfg.START_EPOCH = checkpoint_["epoch"] + 1
        
        loss_results_path = f'{cfg.SAVE_DIR}/loss.csv'
        history = pd.read_csv(loss_results_path)
        best_epoch = history.loc[history['test miou'].idxmax(), 'epoch']
        
        print(f'\n## Successfully loaded model of epoch {checkpoint_["epoch"]} ##', flush = True) 
        print(f'Current best epoch is {best_epoch}', flush = True)
        
    jaccard = JaccardIndex(task="multiclass", num_classes=cfg.CLASSES, ignore_index=None, average=None).to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=cfg.CLASSES, ignore_index=None).to(device)
    f1_metric = MulticlassF1Score(num_classes=cfg.CLASSES, average='macro', ignore_index=None).to(device)
    recall_metric = MulticlassRecall(num_classes=cfg.CLASSES, average='macro', ignore_index=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=cfg.CLASSES, average=None, ignore_index=None).to(device)
    recall_per_class = MulticlassRecall(num_classes=cfg.CLASSES, average=None, ignore_index=None).to(device)
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(cfg.START_EPOCH, cfg.START_EPOCH + cfg.NUM_EPOCHS)):
        
        model, optimizer, scheduler, train_loss, train_miou, train_acc, train_duration = train_step(model=model,
                                                        dataloader=train_dataloader,
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer,                                    
                                                        device=device,
                                                        epoch = epoch,
                                                        cfg = cfg,
                                                        jaccard = jaccard,
                                                        scheduler = scheduler,
                                                        )     

        val_loss, val_iou, val_miou, val_acc, val_duration, \
            val_confmat, val_f1, val_recall, \
                val_per_class_f1, val_per_class_recall = val_step(model = model,
                                        loss_fn = loss_fn,
                                        dataloader = val_dataloader,
                                        device = device, 
                                        cfg = cfg,
                                        epoch = epoch,
                                        jaccard = jaccard,
                                        confmat_metric = confmat_metric,
                                        f1_metric = f1_metric,
                                        f1_per_class = f1_per_class,
                                        recall_metric = recall_metric,
                                        recall_per_class = recall_per_class,
                                        )      
        
        loss_results_path = f'{cfg.SAVE_DIR}/loss.csv'
        record_results(epoch, 
                        train_loss, val_loss,
                        train_duration, val_duration,
                        train_miou, 
                        val_iou, val_miou,
                        loss_results_path,
                        train_acc, val_acc,
                        val_f1, val_recall, 
                        val_per_class_f1, 
                        val_per_class_recall, 
                        optimizer.param_groups[0]['lr'],
                        cfg,
                        )
            
        history = pd.read_csv(loss_results_path)               
        
        print(f"\n ###  miou {val_miou} vs {history['val miou'].max()}\n  ###", flush = True) 
        if val_miou >= history['val miou'].max() or np.round(val_miou,4) >= np.round(history['val miou'].max(),4):
            best_epoch = epoch
            tmp_path = f'{cfg.SAVE_DIR}/best model temp.pth'
            best_model_path_dir = f'{cfg.SAVE_DIR}/best model.pth'
            best_checkpoint = {
                'epoch' : epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict() if scheduler is not None else None,
                }
            torch.save(best_checkpoint, tmp_path)  # Save to temp file
            os.replace(tmp_path, best_model_path_dir) # atomic replace
            print(f'\n***   Saving model of epoch {epoch} with best val miou {val_miou}   ***\n', flush = True)  
            
            plot_confmat(epoch, cfg, val_confmat, text='val')    
                 
        plot_results(cfg.SAVE_DIR, history)

        if cfg.EARLY_STOP == True:    
            if epoch >= 150 and (epoch - best_epoch) == 30:
                print(f'Early stopping at {epoch}', flush = True)
                break

def val_step(model: torch.nn.Module,
         loss_fn: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         device: torch.device,
         cfg,
         epoch: int,
         jaccard, 
         confmat_metric,
        f1_metric,
        f1_per_class, 
        recall_metric, 
        recall_per_class,
         ):
    
    jaccard.reset()
    confmat_metric.reset()
    f1_metric.reset()
    f1_per_class.reset()
    recall_metric.reset()
    recall_per_class.reset()
    
    start_time = datetime.datetime.now()  

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():

        total_batch = len(dataloader)
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        for batch, (img, rgb, label, ID) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)  
            model.to(device)
            
            # 1. Forward pass
            pred = model(img)            
            if cfg.LOSS_FUNCTION == 'composite_loss':
                sum_loss = composite_loss(pred, label, cfg)
            else:
                sum_loss = loss_fn(pred, label)

            total_loss += sum_loss.item()
                
            pred = torch.nn.Softmax(dim=1)(pred)    
            pred = pred.argmax(dim=1) 
            
            jaccard.update(pred, label)
            confmat_metric.update(pred, label)
            f1_metric.update(pred, label)
            f1_per_class.update(pred, label)
            recall_metric.update(pred, label)
            recall_per_class.update(pred, label)
            
            # For accuracy calculation
            correct = (pred == label).sum().item()
            pixels = label.numel()
            total_correct += correct
            total_pixels += pixels

        loss = total_loss/total_batch
        iou = jaccard.compute().tolist()
        miou = torch.nanmean(jaccard.compute()).item()
        acc = total_correct / total_pixels
        confmat = confmat_metric.compute()
        f1 = f1_metric.compute().item()
        per_class_f1 = f1_per_class.compute().tolist()
        recall = recall_metric.compute().item()
        per_class_recall = recall_per_class.compute().tolist()
            
    end_time = datetime.datetime.now()
    duration = end_time - start_time  
    
    return loss, iou, miou, acc, duration, \
        confmat, f1, recall, per_class_f1, per_class_recall



def print_images(dataloader,
                 model,
                 cfg,
                 device,
                 mode,
                    ):    


    # load best model
    best_model = torch.load(f'{cfg.SAVE_DIR}/best model.pth', weights_only=False)
    model.load_state_dict(best_model['model'])
    print(f'\nLoaded best model with epoch {best_model["epoch"]} for printing images', flush = True)
    
    images_folder_path = f'{cfg.SAVE_DIR}/images/{mode}_predictions/'
    if not os.path.exists(images_folder_path): 
        os.makedirs(images_folder_path)
     
    model.eval()
    with torch.inference_mode(): 
        
        if mode == 'val':
            for batch, (img, rgb, label, ID) in enumerate(dataloader): 
                img = img.to(device) 
                model.to(device)

                pred = model(img)       
                pred = torch.nn.Softmax(dim=1)(pred)    
                pred = pred.argmax(dim=1) 
                
                # Below is for saving predicted logits
                # for index in range(len(pred)):
                #     pred_np = pred[index].cpu().numpy().astype(np.uint8)
                #     Image.fromarray(pred_np).save(f'{images_folder_path}/{ID[index]}')
                
                # Below is for colored predictions
                for index in range(len(pred)):
                    pred_tmp = label[index].cpu()              
                    colour_pred = colorEncode(pred_tmp, cfg.class_color)    
                    plt.imshow(colour_pred.clip(0,1).cpu().permute(1, 2, 0))
                    plt.axis('off')
                    plt.savefig(f'{images_folder_path}/{ID[index]}', dpi=600, bbox_inches='tight', pad_inches=0)
                    plt.close('all') 
                    
        if mode == 'test':
            for batch, (img, rgb, ID) in enumerate(dataloader): 
                img = img.to(device) 
                model.to(device)

                pred = model(img)       
                pred = torch.nn.Softmax(dim=1)(pred)    
                pred = pred.argmax(dim=1) 
                
                # Below is for saving predicted logits
                for index in range(len(pred)):
                    pred_np = pred[index].cpu().numpy().astype(np.uint8)
                    Image.fromarray(pred_np).save(f'{images_folder_path}/{ID[index]}')
                
                # Below is for colored predictions
                # for index in range(len(pred)):             
                #     colour_pred = colorEncode(pred_tmp, cfg.class_color)    
                #     plt.imshow(colour_pred.clip(0,1).cpu().permute(1, 2, 0))
                #     plt.axis('off')
                #     plt.savefig(f'{images_folder_path}/{ID[index]}', dpi=600, bbox_inches='tight', pad_inches=0)
                #     plt.close('all') 
                
