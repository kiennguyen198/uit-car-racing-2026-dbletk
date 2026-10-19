"""Local road segmentation model. Does not import or modify the BTC library."""

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


MODEL_NAME = "lraspp_mobilenet_v3_large"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 2  # 0=background, 1=drivable road


def build_model(pretrained_backbone: bool = False):
    backbone_weights = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    return lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=NUM_CLASSES,
    )


def load_checkpoint(path: str, device: torch.device):
    """Load a locally trained checkpoint without any network download."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME or checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("Checkpoint architecture or class count does not match model.py")
    model = build_model(pretrained_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
