## This is for landcover.ai dataset
# Dataset is obtained from https://landcover.ai.linuxpolska.com/
# Paper with code : https://paperswithcode.com/dataset/landcover-ai

from yacs.config import CfgNode as CN

_C = CN()

_C.DEBUG = True

_C.CLASSES = 5                 # background (0), crop (1), weed(2), partial-crop (3), partial-crop (4)
_C.CHANNELS = 'rgb'             
_C.BATCH_SIZE = 4
_C.NUM_EPOCHS = 500
_C.EPOCH_TO_SAVE_MODEL = 1
_C.RESUME_TRAINING = False
_C.START_EPOCH = 0

_C.MODULE = 'PhenoBench'
_C.SAVE_DIR = f'results/{_C.MODULE}/'

_C.IMG_PATH = 'data/PhenoBench/'      # '/home/irene7/scratch/data/PhenoBench/'
_C.TRAIN_OPTIMIZER = 'sgd'                                # 'adam', 'sgd'
_C.TRAIN_LR = 0.01
_C.TRAIN_SCHEDULER = 'yes'                              # 'yes' or 'no'
_C.TRAIN_OPTIMIZER_SDG_MOMENTUM = 0.9
_C.LOSS_FUNCTION = 'cross_entropy'                      # ''cross_entropy', 'focal'
if _C.LOSS_FUNCTION == 'composite_loss':
    _C.w_ce = 1.0                                
    _C.w_dice = 1.0                              
    _C.w_focal = 1.0                          
    _C.w_lovasz = 1.0                        
  
_C.EARLY_STOP = True    



