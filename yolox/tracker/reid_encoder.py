import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


class ReIDEncoder:
    """Lightweight appearance encoder used for ReID-style embeddings.

    The encoder is intentionally compact (MobileNetV2 backbone with a small
    projection head) and can optionally load user-provided weights. If no
    weights are supplied, ImageNet pretrained weights are used when available
    and the model falls back to randomly initialized weights otherwise.
    """

    def __init__(self, device=None, model_path=None, input_size=128, embedding_dim=256):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = int(input_size)
        self.embedding_dim = int(embedding_dim)
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        if model_path:
            self._load_weights(model_path)

    def _build_model(self):
        try:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = mobilenet_v2(weights=None)
        backbone.classifier = nn.Identity()
        projector = nn.Linear(1280, self.embedding_dim)
        return nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            projector,
        )

    def _load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"ReID encoder missing keys: {missing}")
        if unexpected:
            print(f"ReID encoder unexpected keys: {unexpected}")

    @staticmethod
    def _preprocess(crops):
        crops = np.asarray(crops, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crops = (crops - mean) / std
        crops = np.transpose(crops, (0, 3, 1, 2))
        return torch.from_numpy(crops)

    def _crop_images(self, image, tlbrs):
        h, w = image.shape[:2]
        crops = []
        valid_indices = []
        for idx, tlbr in enumerate(tlbrs):
            x0, y0, x1, y1 = tlbr
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = image[y0:y1, x0:x1]
            patch = cv2.resize(patch, (self.input_size, self.input_size))
            crops.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            valid_indices.append(idx)
        return crops, valid_indices

    @torch.inference_mode()
    def __call__(self, image, tlbrs):
        if image is None or len(tlbrs) == 0:
            return []
        crops, valid_indices = self._crop_images(image, tlbrs)
        if not crops:
            return []
        batch = self._preprocess(crops).to(self.device)
        feats = self.model(batch)
        feats = F.normalize(feats, dim=1)
        feats = feats.cpu().numpy()
        embeddings = [None] * len(tlbrs)
        for idx, feat in zip(valid_indices, feats):
            embeddings[idx] = feat
        return embeddings
