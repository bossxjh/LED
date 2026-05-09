import numpy as np
import os
from dataeval.paths import HF_CACHE_DIR

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device=None, normalize=True):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device, cache_folder=str(HF_CACHE_DIR))
        self.normalize = normalize

    def encode(self, texts, batch_size=8, show_progress_bar=False):
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return embs.astype(np.float32)
