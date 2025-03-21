import segmentation_models_pytorch as smp
import torch
def create_unet():
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )

class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        
    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        
        smooth = 1.0
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        
        return bce_loss + dice_loss