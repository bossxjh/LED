import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

def parse_franka_play(pickle_root, num_frames=3):
    """
    生成器，每次 yield np.array [num_frames, H, W, 3]
    适用于 franka_play 数据集。
    
    Args:
        pickle_root (str): pickle 文件所在目录
        num_frames (int): 每个 trajectory 抽取的帧数，默认 3 (首/中/尾)
    """
    def decode_image_bytes(image_bytes):
        try:
            return np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            print(f"[WARN] Failed to decode image: {e}")
            return None

    for root, dirs, files in os.walk(pickle_root):
        for fname in sorted(files):
            if not fname.endswith(".data.pickle") or fname.startswith("._"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load {fname}: {e}")
                    continue

            steps = data.get("steps", [])
            if len(steps) < 1:
                continue

            total_frames = len(steps)
            indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            frames = []
            for idx in indices:
                step = steps[idx]
                img_bytes = step.get("observation", {}).get("image", None)
                if img_bytes is not None:
                    img = decode_image_bytes(img_bytes)
                    if img is not None:
                        frames.append(img)

            if frames:
                yield np.stack(frames, axis=0)  # shape [num_frames, H, W, 3]
