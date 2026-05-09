import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

def decode_image_bytes(image_bytes):
    """把 bytes 转成 np.array RGB 图像"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Failed to decode image: {e}")
        return None

def parse_toto(dataset_path, num_frames=3):
    """
    Toto 数据集解析器（均匀抽帧）
    dataset_path: pickle 文件夹
    输出：生成器，每次 yield np.array [num_frames, H, W, 3]
    """
    # 收集所有 pickle 文件
    pickle_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".data.pickle") and not f.startswith("._"):
                pickle_files.append(os.path.join(root, f))
    pickle_files = sorted(pickle_files)
    print(f"Found {len(pickle_files)} pickle files in {dataset_path}")

    # 遍历每个 pickle 文件
    for fpath in pickle_files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        steps = data.get("steps", [])
        T = len(steps)
        if T < num_frames:
            continue

        # 均匀抽帧
        if num_frames == 1:
            idx = [0]
        elif num_frames == 2:
            idx = [0, T-1]
        else:
            idx = np.linspace(0, T-1, num_frames, dtype=int)

        frames = []
        for i in idx:
            step = steps[i]
            img_bytes = step.get("observation", {}).get("image", None)
            if img_bytes is not None:
                img_np = decode_image_bytes(img_bytes)
                if img_np is not None:
                    frames.append(img_np)

        if len(frames) == num_frames:
            yield np.stack(frames, axis=0)
