# dataeval/sampling.py
import numpy as np

def sample_frames(frames, num_frames):
    """
    均匀采样策略：
    - 1 帧：第一帧
    - 2 帧：首 + 尾
    - >=3：均匀采样（包含首尾）
    """
    T = len(frames)
    if T == 0:
        raise ValueError("No frames available.")

    if num_frames == 1:
        idx = [0]
    elif num_frames == 2:
        idx = [0, T - 1]
    else:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)

    return frames[idx]
