from data_setup import BaseDataset
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import shutil
from torchinfo import summary
import engine
import matplotlib.pyplot as plt
from defaults import _C as cfg
import random
from custom_model import CustomDeepLabV3
# from losses import *
import pandas as pd
from utils import *
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  
def main(): 

    if not os.path.exists(cfg.SAVE_DIR): 
        os.makedirs(cfg.SAVE_DIR)
        shutil.copytree(cfg.MODULE, cfg.SAVE_DIR+cfg.MODULE)
        shutil.copyfile(cfg.MODULE+'/defaults.py', cfg.SAVE_DIR+'configuration.yaml')
           
                   
    ''' Get model summary '''
    model = CustomDeepLabV3(cfg)
    model_summary = str(summary(model,
            input_size=(len(cfg.CHANNELS), 1024, 1024), batch_dim = 0,
            col_names=["input_size", "output_size", "num_params", 'trainable'],
            verbose=0,
            col_width=16,
            row_settings=["var_names"],
            ))
    txt_file = open(f'{cfg.SAVE_DIR}/model_summary.txt', 'w', encoding = 'utf-8')
    txt_file.write(model_summary)
    txt_file.close()
  
    assert cfg.TRAIN_OPTIMIZER in ['adam', 'sgd'], 'Invalid optimizer'
    if cfg.TRAIN_OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.to(device).parameters(), 
                                     lr=cfg.TRAIN_LR)
    else:
        optimizer = torch.optim.SGD(model.to(device).parameters(), 
                                    lr=cfg.TRAIN_LR, 
                                    momentum=cfg.TRAIN_OPTIMIZER_SDG_MOMENTUM)

    if cfg.LOSS_FUNCTION == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss().to(device) 
    else: 
        loss_fn = None
             
     
    train_data = BaseDataset(img_directory=cfg.IMG_PATH, 
                            channels = cfg.CHANNELS,
                            mode = 'train',
                            cfg = cfg,
                            )  

    val_data = BaseDataset(img_directory=cfg.IMG_PATH, 
                            channels = cfg.CHANNELS,
                            mode ='val',
                            cfg = cfg,
                            )   
    test_data = BaseDataset(img_directory=cfg.IMG_PATH, 
                            channels = cfg.CHANNELS,
                            mode ='test',
                            cfg = cfg,
                            )     
    
    if cfg.DEBUG:
        print('*** DEBUGGING MODE ***\n')
        cfg.BATCH_SIZE = 2
        cfg.NUM_EPOCHS = 3
        num_samples = 16
        # Create a subset of train dataset
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = Subset(train_data, random_indices)
        
        random_indices = random.sample(range(len(val_data)), num_samples)
        val_data = Subset(val_data, random_indices)
        
        random_indices = random.sample(range(len(test_data)), num_samples)
        test_data = Subset(test_data, random_indices)
        
    print(f'train set {len(train_data)}, val set {len(val_data)}, test set {len(test_data)}\n')
    
        
    train_dataloader = DataLoader(dataset=train_data, 
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True, )
                        
    val_dataloader = DataLoader(dataset=val_data, 
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True, )
    
    test_dataloader = DataLoader(dataset=test_data, 
                    batch_size=cfg.BATCH_SIZE,
                    shuffle=True, ) 


    if cfg.START_EPOCH == 0:
        label_color_map (cfg)
        
        
    engine.train(model=model, 
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = device,
        cfg = cfg,
        ) 
    

    # Print selected images using best model from above
    print_data = BaseDataset(img_directory=cfg.IMG_PATH, 
                        channels = cfg.CHANNELS,
                        mode = 'val',   # use 'val' or 'test'
                        cfg = cfg,
                        ) 
    
    if cfg.DEBUG == True:
        num_samples = 5
        random_indices = random.sample(range(len(print_data)), num_samples)
        subset_print_data = Subset(print_data, random_indices)
    
    print_dataloader = DataLoader(dataset=subset_print_data, 
                    batch_size = 8,
                    shuffle=True, )

    engine.print_images(dataloader = print_dataloader,
                 model = model,
                 cfg = cfg,
                 device = device,
                 mode = print_data.mode
                 )
         
                  
if __name__ == "__main__":
    main()
    