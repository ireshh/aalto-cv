import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class HybridLoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
        
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        denominator = (pred + target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class InpaintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
    def forward(self, x):
        return self.model(x)
