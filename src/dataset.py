import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class InpaintingDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, mode='train'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.mode = mode
        self.transform = self.get_transforms()

    def get_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                          std=(0.229, 0.224, 0.225)),
                A.ToFloat()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                A.ToFloat()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train':
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0
            
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).unsqueeze(0)
        
        else:
            transformed = self.transform(image=image)
            image = transformed["image"]
            return torch.tensor(image).permute(2, 0, 1)