# dataeval/models/pi0_5_adapter.py
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image
from dataeval.paths import HF_CACHE_DIR

class Pi05Adapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384",
            use_fast=True,
            cache_dir=str(HF_CACHE_DIR),
        )
        self.model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            cache_dir=str(HF_CACHE_DIR),
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_batch(self, batch_frames):
        """
        batch_frames: List[np.ndarray], each [T, H, W, 3]
        return: np.ndarray [B, T, D]
        """
        B = len(batch_frames)
        T = batch_frames[0].shape[0]
        for frames in batch_frames:
            assert frames.shape[0] == T, "Inconsistent T within batch"

        images = []
        for frames in batch_frames:
            for f in frames:
                images.append(Image.fromarray(f.astype(np.uint8)))

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        feats = self.model.vision_model(**inputs).last_hidden_state.mean(dim=1)  # [B*T, D]
        feats = feats.view(B, T, -1)                                             # [B, T, D]

        return feats.cpu().numpy()


    def extract(self, frames):
        return self.extract_batch([frames])[0]
