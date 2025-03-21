import segmentation_models_pytorch as smp

def create_model(encoder='efficientnet-b4'):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )