import os
import cv2
import numpy as np
from PIL import Image

def parse_route(root_dir, num_frames=3):
    """
    遍历 MP4 视频文件，均匀抽帧
    输出: yield np.array [num_frames, H, W, 3]
    """
    # 收集 MP4 文件
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.mp4') and not f.startswith('._'):
                mp4_files.append(os.path.join(dirpath, f))
    mp4_files = sorted(mp4_files)
    print(f"Found {len(mp4_files)} MP4 files in {root_dir}")

    for path in mp4_files:
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            continue

        # 均匀采样 num_frames 帧
        indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == len(indices):
            yield np.stack(frames, axis=0)  # [num_frames, H, W, 3]
        else:
            print(f"{path}: skipped, only {len(frames)} frames read")
