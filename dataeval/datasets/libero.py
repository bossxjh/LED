import os
import h5py
import numpy as np


def parse_libero(dataset_path, num_frames=3):
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith((".h5", ".hdf5"))])

    for fname in files:
        with h5py.File(os.path.join(dataset_path, fname), "r") as f:
            data_group = f["data"]
            for demo in data_group.keys():
                if "agentview_rgb" not in data_group[demo]["obs"]:
                    continue

                frames_h5 = data_group[demo]["obs"]["agentview_rgb"]
                T = frames_h5.shape[0]
                if T == 0 or T < num_frames:
                    continue

                if num_frames == 1:
                    idx = [0]
                elif num_frames == 2:
                    idx = [0, T-1]
                else:
                    idx = np.linspace(0, T-1, num_frames, dtype=int)

                sampled_frames = frames_h5[idx]  # 直接从 HDF5 里读取
                yield sampled_frames