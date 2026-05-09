import os
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def parse_meta_libero(dataset_path, num_frames=3):
    """
    Parse Libero dataset.
    返回整个数据集的 demo 级特征列表，每个元素是一个字典:
        - 'frames': sampled frames, shape [num_frames, H, W, C]
        - 'task_id': int
        - 'task_length': int, avg length of this task
        - 'demo_length': int, length of this demo
        - 'task_description': str
    """
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith((".h5", ".hdf5"))])
    
    demo_features = []

    # 使用 tqdm 跟踪文件读取进度
    for task_idx, fname in tqdm(enumerate(files), desc="Reading files", total=len(files)):
        task_description = os.path.basename(fname).replace(".hdf5", "").replace("_", " ")
        task_id = task_idx

        all_demo_lengths_task = []
        demo_frames_data = []

        with h5py.File(os.path.join(dataset_path, fname), "r") as f:
            data_group = f["data"]

            # 为每个 demo 添加进度条
            for demo in tqdm(data_group.keys(), desc=f"Processing {task_description}", leave=False):
                obs_group = data_group[demo]["obs"]
                if "agentview_rgb" not in obs_group:
                    continue

                frames_h5 = obs_group["agentview_rgb"]
                T = frames_h5.shape[0]
                if T == 0 or T < num_frames:
                    continue

                all_demo_lengths_task.append(T)

                # 选帧索引
                if num_frames == 1:
                    idx = [0]
                elif num_frames == 2:
                    idx = [0, T-1]
                else:
                    idx = np.linspace(0, T-1, num_frames, dtype=int)

                sampled_frames = frames_h5[idx]
                demo_frames_data.append((sampled_frames, T))

        if not all_demo_lengths_task:
            continue

        avg_task_length = int(np.mean(all_demo_lengths_task))

        for sampled_frames, demo_length in demo_frames_data:
            demo_features.append({
                "frames": sampled_frames,
                "task_id": task_id,
                "task_length": avg_task_length,
                "demo_length": demo_length,
                "task_description": task_description
            })

    return demo_features

# import os
# import h5py
# import numpy as np
# from tqdm import tqdm

# def parse_meta_libero(dataset_path, num_frames=3):
#     """
#     Parse Libero dataset.
#     返回整个数据集的 demo 级特征列表，每个元素是一个字典:
#         - 'frames': sampled frames, shape [num_frames, H, W, C]
#         - 'task_id': int
#         - 'task_length': int, avg length of this task
#         - 'demo_length': int, length of this demo
#         - 'task_description': str
#     """
#     files = sorted([f for f in os.listdir(dataset_path) if f.endswith((".h5", ".hdf5"))])
    
#     demo_features = []

#     for task_idx, fname in tqdm(enumerate(files), desc="Reading files", total=len(files)):
#         task_description = os.path.basename(fname).replace(".hdf5", "").replace("_", " ")
#         task_id = task_idx

#         all_demo_lengths_task = []
#         demo_frames_data = []

#         with h5py.File(os.path.join(dataset_path, fname), "r") as f:
#             data_group = f["data"]
#             for demo in data_group.keys():
#                 obs_group = data_group[demo]["obs"]
#                 if "agentview_rgb" not in obs_group:
#                     continue

#                 frames_h5 = obs_group["agentview_rgb"]
#                 T = frames_h5.shape[0]
#                 if T == 0 or T < num_frames:
#                     continue

#                 all_demo_lengths_task.append(T)

#                 # 选帧索引
#                 if num_frames == 1:
#                     idx = [0]
#                 elif num_frames == 2:
#                     idx = [0, T-1]
#                 else:
#                     idx = np.linspace(0, T-1, num_frames, dtype=int)

#                 sampled_frames = frames_h5[idx]
#                 demo_frames_data.append((sampled_frames, T))

#         if not all_demo_lengths_task:
#             continue

#         avg_task_length = int(np.mean(all_demo_lengths_task))

#         for sampled_frames, demo_length in demo_frames_data:
#             demo_features.append({
#                 "frames": sampled_frames,
#                 "task_id": task_id,
#                 "task_length": avg_task_length,
#                 "demo_length": demo_length,
#                 "task_description": task_description
#             })

#     return demo_features
