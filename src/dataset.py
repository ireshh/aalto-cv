import albumentations as A
from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        
        if self.mask_paths:  # For training
            mask = load_mask(self.mask_paths[idx])
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            return image, mask
        else:  # For inference
            if self.transform:
                augmented = self.transform(image=image)
                return augmented['image']
            return image