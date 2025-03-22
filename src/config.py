from pathlib import Path
import torch

class Config:
    # Path Configuration
    DATA_DIR = Path("/home/oliraff/cvhackathon/aalto-cv/data")
    TRAIN_IMAGE_DIR = DATA_DIR / "train/train/images"
    TRAIN_MASK_DIR = DATA_DIR / "train/train/masks"
    TEST_IMAGE_DIR = DATA_DIR / "test/test/images"
    MODEL_DIR = Path("./models")
    
    # Hardware Settings
    DEVICE = "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = False
    
    # Model Architecture
    MODEL_NAME = "efficientnet-b0"
    ENCODER_WEIGHTS = None
    IN_CHANNELS = 3
    CLASSES = 1
    DROPOUT = 0.2
    
    # Training Parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    
    # Image Processing
    IMAGE_SIZE = 256
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Validation
    VAL_SPLIT = 0.2
    VAL_INTERVAL = 1
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    METRIC_MONITOR = "val_dice"

    def __init__(self):
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.DEVICE = torch.device(self.DEVICE)
        
config = Config()