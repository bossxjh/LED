# dataeval/models/clip_adapter.py
import clip
import torch
import numpy as np
from PIL import Image
from dataeval.paths import CLIP_CACHE_DIR

class CLIPAdapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=str(CLIP_CACHE_DIR))
        # self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    # def extract(self, frames):
    #     images = [
    #         self.preprocess(Image.fromarray(f.astype(np.uint8)))
    #         .unsqueeze(0)
    #         .to(self.device)
    #         for f in frames
    #     ]
    #     with torch.no_grad():
    #         feats = torch.cat(
    #             [self.model.encode_image(img) for img in images], dim=0
    #         )
    #         feats = feats / feats.norm(dim=-1, keepdim=True)
    #     return feats.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, batch_frames):
        """
        batch_frames: List[np.ndarray], each [T, H, W, 3]
        return: np.ndarray [B, D]
        """
        B = len(batch_frames)
        T = batch_frames[0].shape[0]

        images = []
        for frames in batch_frames:
            for f in frames:
                images.append(
                    self.preprocess(Image.fromarray(f.astype(np.uint8)))
                )

        images = torch.stack(images).to(self.device)  # [B*T, 3, H, W]

        feats = self.model.encode_image(images)        # [B*T, D]
        feats = feats / feats.norm(dim=-1, keepdim=True)

        feats = feats.view(B, T, -1)       # [B, T, D]
        return feats.cpu().numpy()

    def extract(self, frames):
        # 兼容旧接口
        return self.extract_batch([frames])[0]
