import os
from PIL import Image
import numpy as np

def parse_nyu_opening_door(folder_path, num_frames=3):
    """
    Parse NYU_opening_door 数据集，生成器版本，每次 yield 一个 run 的帧数组。

    Args:
        folder_path (str): 数据集根目录，例如 .../handle/val_all
        num_frames (int): 每个 run 抽取的帧数（默认首/中/尾3帧）

    Yields:
        np.array: shape [num_frames, H, W, 3], dtype=np.uint8
    """
    run_list = sorted([d for d in os.listdir(folder_path) if d.startswith("run_")])

    for run_name in run_list:
        run_path = os.path.join(folder_path, run_name)
        img_dir = os.path.join(run_path, "images_linear")
        if not os.path.isdir(img_dir):
            continue

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg") and not f.startswith("._")])
        total = len(img_files)
        if total == 0:
            continue

        # 均匀抽取 num_frames 帧
        indices = np.linspace(0, total-1, min(num_frames, total), dtype=int)
        frames = []
        for idx in indices:
            img_path = os.path.join(img_dir, img_files[idx])
            try:
                img = Image.open(img_path).convert("RGB")
                frames.append(np.array(img))  # HWC, RGB
            except Exception as e:
                print(f"[WARN] 跳过 {img_path}: {e}")
                continue

        if len(frames) == 0:
            continue

        yield np.stack(frames, axis=0)
