import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns



def label_color_map(cfg):
    images_folder_path = f'{cfg.SAVE_DIR}/images/'
    if not os.path.exists(images_folder_path): 
        os.makedirs(images_folder_path)
    plt.figure(figsize=(15, 3))
    plt.tight_layout()   
    
    for index, classes in enumerate(cfg.class_color):
        img = Image.new('RGB', (7, 1), color = tuple(classes))   # width, height    
        plt.subplot(1, len(cfg.class_color), index+1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(f'{index} - {cfg.classes[index]}', fontsize=9)
        
        # alternatively,
        # img = torch.ones(3,3,1) # dim, width, height
        # img[0] = train_dataloader.dataset.classes_colour[index][0]/255
        # img[1] = train_dataloader.dataset.classes_colour[index][1]/255
        # img[2] = train_dataloader.dataset.classes_colour[index][2]/255
        # plt.subplot(1, len(train_dataloader.dataset.classes_colour), index+1)
        # plt.axis('off')
        # plt.imshow(img.permute(1,2,0))
        # plt.title(f'{index} - {train_dataloader.dataset.classes[index]}', fontsize=9)
    plt.savefig(f'{images_folder_path}/classes_color.png')
    plt.close('all')


def record_results(epoch : int, 
                    train_loss, val_loss,
                    train_duration, val_duration,
                    train_miou,
                    val_iou, val_miou, 
                    path,
                    train_acc, val_acc, 
                    val_f1, val_recall,
                    val_per_class_f1,
                    val_per_class_recall,
                    current_lr,
                    cfg,
                    ):
    
    results = pd.DataFrame({'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_duration' : train_duration,
            'val_duration' : val_duration,
            'val_iou [background]' : val_iou[0],
            'val_iou [building]' : val_iou[1],
            'val_iou [woodland]' : val_iou[2], 
            'val_iou [water]' : val_iou[3],
            'val_iou [road]' : val_iou[4], 
            'val miou': val_miou,
            'train miou': train_miou,
            'train acc' : train_acc,
            'val acc' : val_acc,
            'lr' : current_lr,
            }, index=[0])
    
    f1_path = f'{cfg.SAVE_DIR}/{cfg.MODULE} f1.csv'
    recall_path = f'{cfg.SAVE_DIR}/{cfg.MODULE} recall.csv'
    
    f1_results = pd.DataFrame({
                    'val_f1' : val_f1,
                    'val_f1 [background]' : val_per_class_f1[0],
                    'val_f1 [building]' : val_per_class_f1[1],
                    'val_f1 [woodland]' : val_per_class_f1[2],
                    'val_f1 [water]' : val_per_class_f1[3],
                    'val_f1 [road]' : val_per_class_f1[4],
                    }, index=[0]) 
    
    recall_results = pd.DataFrame({
                    'val_recall' : val_recall,
                    'val_recall [background]' : val_per_class_recall[0],
                    'val_recall [building]' : val_per_class_recall[1],
                    'val_recall [woodland]' : val_per_class_recall[2],
                    'val_recall [water]' : val_per_class_recall[3],
                    'val_recall [road]' : val_per_class_recall[4],                    
                    }, index=[0])
    
    if os.path.exists(path):
        results.to_csv(path, mode='a', header=False, index=False)
    else:
        results.to_csv(path, index=False)
    
    if os.path.exists(f1_path):
        f1_results.to_csv(f1_path, mode='a', header=False, index=False)
    else:
        f1_results.to_csv(f1_path, index=False)
        
    if os.path.exists(recall_path):
        recall_results.to_csv(recall_path, mode='a', header=False, index=False)
    else:
        recall_results.to_csv(recall_path, index=False)


def plot_confmat(epoch, cfg, confmat, text):
    confmat = confmat.cpu().numpy()
    # Optional: Normalize by true label counts (row-wise)
    confmat_normalized = np.nan_to_num(confmat.astype('float') / confmat.sum(axis=1, keepdims=True))
    # confmat_normalized = np.nan_to_num(confmat_normalized)  # handle division by zero if any class is missing

    cm = {
        'confmax': confmat,
        'confmat_normalized': confmat_normalized 
    }
    torch.save(cm, f'{cfg.SAVE_DIR}/confusion_matrix for {text}.pt')

    class_names = cfg.classes

    plt.figure(figsize=(10, 8))
    sns.heatmap(confmat_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix for epoch {epoch}')
    plt.tight_layout()
    
    plt.savefig(f'{cfg.SAVE_DIR}/confusion_matrix_{text}.png')
    plt.close('all')



def colorEncode(img: torch.Tensor, 
                classes_colour: torch.Tensor):
    # classes.shape (1, W, H)
    colored_img = torch.zeros(3, img.shape[0], img.shape[1])
    list1 = img.unique().tolist()
    # list2 = [ele for ele in list1 if ele != -1]        
    # print(f'list: {list1}, list after delete: {list2}')
    for c in list1:
        color = torch.tensor(classes_colour[c])               # if c=1, then color will be torch.size([3]) and has value of tensor([169.,169.,169.])
        color = color.unsqueeze(1).unsqueeze(1)   # torch.size([3,1,1])
        colored_img += color.expand(3, img.shape[0], img.shape[1]) * (img == c)
    return colored_img/255.0  
        
  
def plot_images(rgb_img,  
                input_img, 
                label_img, 
                colour_pred_img, 
                channels : str,
                ) -> None: 
        
    plt.figure(figsize=(15, 5))
    # # plt.rcParams["axes.edgecolor"] = "0.15"
    # # plt.rcParams["axes.linewidth"]  = 1.25
    num = 4
    
    plt.subplot(1, num, 1)
    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis('off')
        
    plt.subplot(1, num, 2)
    plt.imshow(input_img.permute(1, 2, 0))
    plt.title(f'Model Input ({channels})')
    plt.axis('off')
    
    plt.subplot(1, num, 3)
    plt.imshow(label_img.permute(1, 2, 0))
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, num, 4)
    plt.imshow(colour_pred_img.permute(1, 2, 0))
    plt.title('Prediction')
    plt.axis('off')



def plot_results(dir_path, history):

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss', color='purple')
    # plt.plot(history['epoch'], history['test_loss'], label='Test Loss', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, 
         color='gray', 
         linestyle='--', 
         linewidth=0.5,
         alpha=0.5)
    
    plt.subplot(3, 1, 2)
    plt.plot(history['epoch'], history['train acc'], label='Train accuracy', color='blue')
    plt.plot(history['epoch'], history['val acc'], label='Val accuracy', color='purple')
    # plt.plot(history['epoch'], history['test acc'], label='Test accuracy', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, 
         color='gray', 
         linestyle='--', 
         linewidth=0.5,
         alpha=0.5)
    
    plt.subplot(3, 1, 3)
    plt.plot(history['epoch'], history['train miou'], label='Train mIoU', color='blue')
    plt.plot(history['epoch'], history['val miou'], label='Val mIoU', color='purple')
    # plt.plot(history['epoch'], history['test miou'], label='Test mIoU', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True, 
         color='gray', 
         linestyle='--', 
         linewidth=0.5,
         alpha=0.5)

    plt.savefig(dir_path + '/loss graph.png')
    plt.close('all')
    
    plt.plot(history['lr'])
    plt.title("Learning Rate Schedule")
    plt.savefig(dir_path + '/lr.png')   
    plt.close('all') 
    
        