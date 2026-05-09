from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RELEASE_ROOT / "checkpoint"
BGE_MODEL_PATH = CHECKPOINT_DIR / "bge-large-en-v1.5"
OPENVLA_CACHE_DIR = CHECKPOINT_DIR / "openvla_cache"
HF_CACHE_DIR = CHECKPOINT_DIR / "huggingface"
CLIP_CACHE_DIR = CHECKPOINT_DIR / "clip"
RESNET18_WEIGHT_PATH = CHECKPOINT_DIR / "resnet18-f37072fd.pth"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
