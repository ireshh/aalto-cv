import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from config import config

class InpaintDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train
        self.transform = self.get_transforms()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mask_paths is not None:
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
            
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
    def get_transforms(self):
        if self.is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=25,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
