#/dataeval/datasets_meta/libero_rlds.py

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds


def _maybe_decode(x):
    """Decode tf.Tensor/bytes/np.bytes_ into python str if possible."""
    try:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
    except Exception:
        pass

    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _try_get_episode_native_id(ep, step0):
    """
    Best-effort: try to find a native ID in RLDS record.
    Different RLDS datasets may store id under different keys.
    Returns (id_key, id_value) or (None, None).
    """
    # 1) Try keys at episode level
    for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
        if k in ep:
            try:
                return k, _maybe_decode(ep[k])
            except Exception:
                pass

    # 2) Try keys at step0 level
    for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
        if k in step0:
            try:
                return k, _maybe_decode(step0[k])
            except Exception:
                pass

    # 3) Try nested fields sometimes used in RLDS
    # e.g., step0["metadata"]["episode_id"] etc.
    for root in ["metadata", "info"]:
        if root in ep:
            try:
                obj = ep[root]
                if isinstance(obj, tf.Tensor):
                    obj = obj.numpy()
            except Exception:
                obj = ep[root]
            # if it's a dict-like
            try:
                keys = list(obj.keys())
            except Exception:
                keys = []
            for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
                if k in keys:
                    try:
                        return f"{root}.{k}", _maybe_decode(obj[k])
                    except Exception:
                        pass

        if root in step0:
            obj = step0[root]
            try:
                keys = list(obj.keys())
            except Exception:
                keys = []
            for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
                if k in keys:
                    try:
                        return f"{root}.{k}", _maybe_decode(obj[k])
                    except Exception:
                        pass

    return None, None

def parse_meta_libero_rlds(
    dataset_path,
    num_frames=3,
    split="train",
    image_key="image",
    max_episodes=None,
    add_uid=True,
    # --- NEW ---
    action_key="action",
    compute_action_stats=True,
    eps_small=1e-3,
):
    """
    Parse Libero RLDS (TFDS) dataset with UNIQUE identifiers + action stats.

    Returns demo_features: List[dict], each:
      - frames: [num_frames, H, W, 3]
      - task_id: int
      - task_length: int
      - demo_length: int
      - task_description: str
      - action_jerk: float (optional)
      - action_smoothness: float (optional)
      - action_energy: float (optional)
      - action_small_ratio: float (optional)
      - unique ids if add_uid
    """
    dataset_path = os.path.abspath(dataset_path.rstrip("/"))
    data_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)

    ds = tfds.load(
        dataset_name,
        data_dir=data_dir,
        split=split,
        shuffle_files=False,
    )

    task2id = {}
    task_lengths = {}
    raw = []  # (task_id, task_desc, T, frames, ep_idx, native_k, native_v, action_stats)

    epi_iter = ds.take(max_episodes) if max_episodes is not None else ds
    pbar = tqdm(desc=f"Reading RLDS episodes ({dataset_name}/{split})", total=max_episodes)

    for ep_idx, ep in enumerate(epi_iter):
        # ---------- Pass 0: read step0 ----------
        steps0 = ep["steps"]
        if not isinstance(steps0, tf.data.Dataset):
            steps0 = tf.data.Dataset.from_tensor_slices(steps0)

        try:
            step0 = next(iter(steps0.take(1)))
        except StopIteration:
            if max_episodes is not None:
                pbar.update(1)
            continue

        task_desc = _maybe_decode(step0["language_instruction"])
        if task_desc not in task2id:
            task2id[task_desc] = len(task2id)
        task_id = task2id[task_desc]

        native_k, native_v = (None, None)
        if add_uid:
            native_k, native_v = _try_get_episode_native_id(ep, step0)

        # ---------- Pass 1: count length ----------
        steps1 = ep["steps"]
        if not isinstance(steps1, tf.data.Dataset):
            steps1 = tf.data.Dataset.from_tensor_slices(steps1)

        T = 0
        for _ in steps1:
            T += 1

        if T == 0 or T < num_frames:
            if max_episodes is not None:
                pbar.update(1)
            continue

        # frame indices
        if num_frames == 1:
            idxs = [0]
        elif num_frames == 2:
            idxs = [0, T - 1]
        else:
            idxs = np.linspace(0, T - 1, num_frames, dtype=int).tolist()

        idx_set = set(idxs)
        idx_to_pos = {i: p for p, i in enumerate(idxs)}
        cache = [None] * len(idxs)

        # ---------- Pass 2: extract frames + action stats ----------
        steps2 = ep["steps"]
        if not isinstance(steps2, tf.data.Dataset):
            steps2 = tf.data.Dataset.from_tensor_slices(steps2)

        # action accumulators
        sum_energy = 0.0
        sum_smooth = 0.0
        smooth_cnt = 0
        sum_jerk = 0.0
        jerk_cnt = 0
        small_cnt = 0
        act_cnt = 0

        a_prev2 = None
        a_prev1 = None

        for i, step in enumerate(steps2):
            # frames
            if i in idx_set:
                obs = step["observation"]
                if image_key not in obs:
                    raise KeyError(f"image_key='{image_key}' not in observation keys: {list(obs.keys())}")
                cache[idx_to_pos[i]] = obs[image_key].numpy()

            # action stats
            if compute_action_stats and (action_key in step):
                a = step[action_key]
                if isinstance(a, tf.Tensor):
                    a = a.numpy()
                a = np.asarray(a, dtype=np.float64).reshape(-1)

                nrm = float(np.linalg.norm(a))
                sum_energy += nrm
                act_cnt += 1
                if nrm < eps_small:
                    small_cnt += 1

                if a_prev1 is not None:
                    sum_smooth += float(np.linalg.norm(a - a_prev1))
                    smooth_cnt += 1

                if a_prev2 is not None and a_prev1 is not None:
                    j = a - 2.0 * a_prev1 + a_prev2
                    sum_jerk += float(np.linalg.norm(j))
                    jerk_cnt += 1

                a_prev2 = a_prev1
                a_prev1 = a

            # frames early break（但如果要 action stats，就不能提前停）
            if (not compute_action_stats) and all(x is not None for x in cache):
                break

        if any(x is None for x in cache):
            if max_episodes is not None:
                pbar.update(1)
            continue

        frames = np.stack(cache, axis=0)

        action_stats = {}
        if compute_action_stats and act_cnt > 0:
            action_stats = {
                "action_energy": float(sum_energy / max(act_cnt, 1)),
                "action_smoothness": float(sum_smooth / max(smooth_cnt, 1)) if smooth_cnt > 0 else 0.0,
                "action_jerk": float(sum_jerk / max(jerk_cnt, 1)) if jerk_cnt > 0 else 0.0,
                "action_small_ratio": float(small_cnt / max(act_cnt, 1)),
            }

        raw.append((task_id, task_desc, T, frames, int(ep_idx), native_k, native_v, action_stats))
        task_lengths.setdefault(task_id, []).append(T)

        if max_episodes is not None:
            pbar.update(1)

    if max_episodes is not None:
        pbar.close()

    task_avg_len = {tid: int(np.mean(lens)) for tid, lens in task_lengths.items()}

    demo_features = []
    for task_id, task_desc, demo_len, frames, episode_index, native_k, native_v, action_stats in raw:
        item = {
            "frames": frames,
            "task_id": int(task_id),
            "task_length": int(task_avg_len.get(task_id, demo_len)),
            "demo_length": int(demo_len),
            "task_description": str(task_desc),
            **action_stats,
        }
        if add_uid:
            item["episode_index"] = int(episode_index)
            item["demo_uid"] = f"{dataset_name}/{split}|ep={episode_index:06d}"
            item["native_episode_id_key"] = native_k
            item["native_episode_id"] = native_v
        demo_features.append(item)

    return demo_features

# import os
# import numpy as np
# from tqdm import tqdm
# import tensorflow as tf
# import tensorflow_datasets as tfds


# def _maybe_decode(x):
#     """Decode tf.Tensor/bytes/np.bytes_ into python str if possible."""
#     try:
#         if isinstance(x, tf.Tensor):
#             x = x.numpy()
#     except Exception:
#         pass

#     if isinstance(x, (bytes, np.bytes_)):
#         return x.decode("utf-8", errors="ignore")
#     return str(x)


# def _try_get_episode_native_id(ep, step0):
#     """
#     Best-effort: try to find a native ID in RLDS record.
#     Different RLDS datasets may store id under different keys.
#     Returns (id_key, id_value) or (None, None).
#     """
#     # 1) Try keys at episode level
#     for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
#         if k in ep:
#             try:
#                 return k, _maybe_decode(ep[k])
#             except Exception:
#                 pass

#     # 2) Try keys at step0 level
#     for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
#         if k in step0:
#             try:
#                 return k, _maybe_decode(step0[k])
#             except Exception:
#                 pass

#     # 3) Try nested fields sometimes used in RLDS
#     # e.g., step0["metadata"]["episode_id"] etc.
#     for root in ["metadata", "info"]:
#         if root in ep:
#             try:
#                 obj = ep[root]
#                 if isinstance(obj, tf.Tensor):
#                     obj = obj.numpy()
#             except Exception:
#                 obj = ep[root]
#             # if it's a dict-like
#             try:
#                 keys = list(obj.keys())
#             except Exception:
#                 keys = []
#             for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
#                 if k in keys:
#                     try:
#                         return f"{root}.{k}", _maybe_decode(obj[k])
#                     except Exception:
#                         pass

#         if root in step0:
#             obj = step0[root]
#             try:
#                 keys = list(obj.keys())
#             except Exception:
#                 keys = []
#             for k in ["episode_id", "trajectory_id", "demo_id", "id", "uuid", "key"]:
#                 if k in keys:
#                     try:
#                         return f"{root}.{k}", _maybe_decode(obj[k])
#                     except Exception:
#                         pass

#     return None, None


# def parse_meta_libero_rlds(
#     dataset_path,
#     num_frames=3,
#     split="train",
#     image_key="image",
#     max_episodes=None,
#     add_uid=True,
# ):
#     """
#     Parse Libero RLDS (TFDS) dataset with UNIQUE identifiers.

#     Returns:
#         demo_features: List[dict], each:
#           - 'frames': [num_frames, H, W, 3]
#           - 'task_id': int
#           - 'task_length': int
#           - 'demo_length': int
#           - 'task_description': str
#           - 'episode_index': int              (NEW, unique within this split)
#           - 'demo_uid': str                   (NEW, stable id string)
#           - 'native_episode_id': str|None     (NEW, best-effort if exists)
#           - 'native_episode_id_key': str|None (NEW)
#     """
#     dataset_path = os.path.abspath(dataset_path.rstrip("/"))
#     data_dir = os.path.dirname(dataset_path)
#     dataset_name = os.path.basename(dataset_path)

#     ds = tfds.load(
#         dataset_name,
#         data_dir=data_dir,
#         split=split,
#         shuffle_files=False,
#     )

#     task2id = {}
#     task_lengths = {}  # task_id -> list of demo lengths
#     raw = []           # list of (task_id, task_desc, demo_len, frames, episode_index, native_id_key, native_id)

#     epi_iter = ds.take(max_episodes) if max_episodes is not None else ds
#     pbar = tqdm(
#         desc=f"Reading RLDS episodes ({dataset_name}/{split})",
#         total=max_episodes
#     )

#     for ep_idx, ep in enumerate(epi_iter):
#         steps_ds = ep["steps"]
#         if not isinstance(steps_ds, tf.data.Dataset):
#             steps_ds = tf.data.Dataset.from_tensor_slices(steps_ds)

#         # step0: get language_instruction (and maybe native id)
#         try:
#             step0 = next(iter(steps_ds.take(1)))
#         except StopIteration:
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         instr = step0["language_instruction"]
#         task_desc = _maybe_decode(instr)

#         if task_desc not in task2id:
#             task2id[task_desc] = len(task2id)
#         task_id = task2id[task_desc]

#         native_k, native_v = (None, None)
#         if add_uid:
#             native_k, native_v = _try_get_episode_native_id(ep, step0)

#         # Count length T (first pass)
#         T = 0
#         for _ in steps_ds:
#             T += 1

#         if T == 0 or T < num_frames:
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         # Choose frame indices
#         if num_frames == 1:
#             idxs = [0]
#         elif num_frames == 2:
#             idxs = [0, T - 1]
#         else:
#             idxs = np.linspace(0, T - 1, num_frames, dtype=int).tolist()

#         idx_set = set(idxs)
#         idx_to_pos = {i: p for p, i in enumerate(idxs)}
#         cache = [None] * len(idxs)

#         # Second pass: extract frames
#         for i, step in enumerate(steps_ds):
#             if i not in idx_set:
#                 continue
#             obs = step["observation"]
#             if image_key not in obs:
#                 raise KeyError(f"image_key='{image_key}' not in observation keys: {list(obs.keys())}")
#             img = obs[image_key].numpy()
#             cache[idx_to_pos[i]] = img
#             if all(x is not None for x in cache):
#                 break

#         if any(x is None for x in cache):
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         frames = np.stack(cache, axis=0)
#         raw.append((task_id, task_desc, T, frames, int(ep_idx), native_k, native_v))
#         task_lengths.setdefault(task_id, []).append(T)

#         if max_episodes is not None:
#             pbar.update(1)

#     if max_episodes is not None:
#         pbar.close()

#     task_avg_len = {tid: int(np.mean(lens)) for tid, lens in task_lengths.items()}

#     demo_features = []
#     for task_id, task_desc, demo_len, frames, episode_index, native_k, native_v in raw:
#         item = {
#             "frames": frames,
#             "task_id": int(task_id),
#             "task_length": int(task_avg_len.get(task_id, demo_len)),
#             "demo_length": int(demo_len),
#             "task_description": str(task_desc),
#         }
#         if add_uid:
#             item["episode_index"] = int(episode_index)  # unique within split
#             # demo_uid is what you will use in reports, super convenient
#             item["demo_uid"] = f"{dataset_name}/{split}|ep={episode_index:06d}"
#             item["native_episode_id_key"] = native_k
#             item["native_episode_id"] = native_v
#         demo_features.append(item)

#     return demo_features





# import os
# import numpy as np
# from tqdm import tqdm
# import tensorflow as tf
# import tensorflow_datasets as tfds


# def parse_meta_libero_rlds(dataset_path, num_frames=3, split="train", image_key="image", max_episodes=None):
#     """
#     Parse Libero RLDS (TFDS) dataset.

#     Args:
#         dataset_path: 指向 TFDS 数据集目录，例如：
#             /mnt/.../modified_libero_rlds/libero_10_no_noops
#         num_frames: 每个 demo 采样帧数
#         split: "train"/"validation"/"test"
#         image_key: "image" 或 "wrist_image"
#         max_episodes: 只解析前 N 个 episode（用于调试），None 表示全量

#     Returns:
#         demo_features: List[dict], 每个元素：
#           - 'frames': [num_frames, H, W, 3]
#           - 'task_id': int
#           - 'task_length': int (该 task 的平均 demo 长度)
#           - 'demo_length': int
#           - 'task_description': str (language_instruction)
#     """

#     dataset_path = os.path.abspath(dataset_path.rstrip("/"))
#     data_dir = os.path.dirname(dataset_path)              # .../modified_libero_rlds
#     dataset_name = os.path.basename(dataset_path)         # libero_10_no_noops

#     ds = tfds.load(
#         dataset_name,
#         data_dir=data_dir,
#         split=split,
#         shuffle_files=False,
#     )

#     # instruction(str) -> task_id(int)
#     task2id = {}
#     task_lengths = {}   # task_id -> list of demo lengths
#     raw = []            # list of (task_id, task_desc, demo_len, frames)

#     epi_iter = ds.take(max_episodes) if max_episodes is not None else ds
#     pbar = tqdm(desc=f"Reading RLDS episodes ({dataset_name}/{split})", total=max_episodes)

#     for ep in epi_iter:
#         steps_ds = ep["steps"]
#         if not isinstance(steps_ds, tf.data.Dataset):
#             steps_ds = tf.data.Dataset.from_tensor_slices(steps_ds)

#         # step0 拿 language_instruction
#         try:
#             step0 = next(iter(steps_ds.take(1)))
#         except StopIteration:
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         instr = step0["language_instruction"].numpy()
#         if isinstance(instr, (bytes, np.bytes_)):
#             task_desc = instr.decode("utf-8", errors="ignore")
#         else:
#             task_desc = str(instr)

#         if task_desc not in task2id:
#             task2id[task_desc] = len(task2id)
#         task_id = task2id[task_desc]

#         # 先数长度 T（最稳，但会遍历一次 steps）
#         T = 0
#         for _ in steps_ds:
#             T += 1

#         if T == 0 or T < num_frames:
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         # 选帧索引（与你原逻辑一致）
#         if num_frames == 1:
#             idxs = [0]
#         elif num_frames == 2:
#             idxs = [0, T - 1]
#         else:
#             idxs = np.linspace(0, T - 1, num_frames, dtype=int).tolist()

#         idx_set = set(idxs)
#         idx_to_pos = {i: p for p, i in enumerate(idxs)}
#         cache = [None] * len(idxs)

#         # 第二次遍历：抽帧
#         for i, step in enumerate(steps_ds):
#             if i not in idx_set:
#                 continue
#             obs = step["observation"]
#             if image_key not in obs:
#                 # observation keys: image, wrist_image, joint_state, state
#                 raise KeyError(f"image_key='{image_key}' not in observation keys: {list(obs.keys())}")
#             img = obs[image_key].numpy()  # uint8 (256,256,3)
#             cache[idx_to_pos[i]] = img
#             if all(x is not None for x in cache):
#                 break

#         if any(x is None for x in cache):
#             if max_episodes is not None:
#                 pbar.update(1)
#             continue

#         frames = np.stack(cache, axis=0)
#         raw.append((task_id, task_desc, T, frames))
#         task_lengths.setdefault(task_id, []).append(T)

#         if max_episodes is not None:
#             pbar.update(1)

#     if max_episodes is not None:
#         pbar.close()

#     task_avg_len = {tid: int(np.mean(lens)) for tid, lens in task_lengths.items()}

#     demo_features = []
#     for task_id, task_desc, demo_len, frames in raw:
#         demo_features.append({
#             "frames": frames,
#             "task_id": task_id,
#             "task_length": task_avg_len.get(task_id, demo_len),
#             "demo_length": demo_len,
#             "task_description": task_desc,
#         })

#     return demo_features