from pathlib import Path
import torch

class Config:
    # Path Configuration
    DATA_DIR = Path("/home/oliraff/cvhackathon/aalto-cv/data")  # Absolute path to data
    MODEL_DIR = Path("./models")                                 # Model save directory
    
    # Hardware Settings
    DEVICE = "cpu"               # Force CPU mode
    NUM_WORKERS = 2              # DataLoader workers
    PIN_MEMORY = False           # Disable for CPU
    
    # Model Architecture
    MODEL_NAME = "efficientnet-b0"  # Lightweight variant
    ENCODER_WEIGHTS = None          # No ImageNet weights
    IN_CHANNELS = 3                 # RGB input
    CLASSES = 1                      # Binary segmentation
    DROPOUT = 0.2                    # Regularization
    
    # Training Parameters
    BATCH_SIZE = 8              # Reduced for CPU memory
    NUM_EPOCHS = 50             # Total epochs
    LR = 1e-4                    # Learning rate
    WEIGHT_DECAY = 1e-4          # AdamW regularization
    PATIENCE = 5                 # Early stopping
    
    # Image Processing
    IMAGE_SIZE = 256             # Input resolution
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Validation
    VAL_SPLIT = 0.2              # Train/val split ratio
    VAL_INTERVAL = 1             # Validate every epoch
    
    # Checkpointing
    SAVE_BEST_ONLY = True        # Only save best model
    METRIC_MONITOR = "val_dice"  # Monitoring metric

    def __init__(self):
        # Create directories if missing
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.DEVICE = torch.device(self.DEVICE)
        
config = Config()