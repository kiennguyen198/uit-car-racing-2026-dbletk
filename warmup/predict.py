"""Offline road-mask inference for a trained checkpoint.

Call only after measuring the actual GetRaw image shape and channel order.
The returned mask has the checkpoint's training resolution, not camera size.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from model import load_checkpoint


class RoadPredictor:
    def __init__(self, checkpoint_path: str | Path, *, color_order: str):
        if color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must explicitly be RGB or BGR")
        self.color_order = color_order
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CPU preprocessing is small; avoid scheduling eight CPU workers
        # per camera frame when convolutions already run on the GPU.
        if self.device.type == "cuda":
            torch.set_num_threads(1)
        self.model, checkpoint = load_checkpoint(str(checkpoint_path), self.device)
        self.size = tuple(checkpoint["image_size"])
        self.mean = tuple(checkpoint["mean"])
        self.std = tuple(checkpoint["std"])

    def predict(self, raw_image: np.ndarray) -> np.ndarray:
        if raw_image.dtype != np.uint8 or raw_image.ndim != 3 or raw_image.shape[2] != 3:
            raise ValueError("Expected uint8 camera image with shape H×W×3")
        rgb = raw_image if self.color_order == "RGB" else raw_image[:, :, ::-1]
        image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        image = image.resize(self.size, Image.Resampling.BILINEAR)
        image_tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(image_tensor)["out"]
            mask = logits.argmax(dim=1)[0].to(dtype=torch.uint8).cpu().numpy()
        return mask
