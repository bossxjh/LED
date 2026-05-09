import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

def covariance_entropy_norm(X, eps=1e-12):
    """
    Entropy of covariance eigenvalues.
    X: [N, D]
    """
    if X.shape[0] < 2:
        return 0.0

    C = np.cov(X, rowvar=False)

    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, eps, None)

    p = eigvals / np.sum(eigvals)

    H = -np.sum(p * np.log(p))

    # normalize by log(D) so range ~ [0,1]
    D = len(eigvals)
    return float(H / np.log(D))

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

    # 3) Try nested fields
    for root in ["metadata", "info"]:
        if root in ep:
            try:
                obj = ep[root]
                if isinstance(obj, tf.Tensor):
                    obj = obj.numpy()
            except Exception:
                obj = ep[root]
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
    action_key="action",
    compute_action_stats=True,
    eps_small=1e-3,
):
    """
    Parse Libero RLDS (TFDS) dataset with metadata, sampled frames,
    sampled actions, and action statistics.

    Returns demo_features: List[dict], each contains:
      - frames: [num_frames, H, W, 3]
      - sampled_actions: [num_frames, A]
      - task_id: int
      - task_length: int
      - demo_length: int
      - task_description: str
      - action_jerk: float
      - action_smoothness: float
      - action_energy: float
      - action_small_ratio: float
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
    raw = []
    # raw item:
    # (task_id, task_desc, T, frames, sampled_actions, ep_idx, native_k, native_v, action_stats)

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
        frame_cache = [None] * len(idxs)
        action_cache = [None] * len(idxs)

        # ---------- Pass 2: extract frames + sampled actions + action stats ----------
        steps2 = ep["steps"]
        if not isinstance(steps2, tf.data.Dataset):
            steps2 = tf.data.Dataset.from_tensor_slices(steps2)

        # action accumulators
        sum_energy = 0.0
        sum_smooth = 0.0
        smooth_cnt = 0

        sum_jerk = 0.0
        jerk_cnt = 0
        jerks = []
        actions_full = []

        small_cnt = 0
        act_cnt = 0

        a_prev2 = None
        a_prev1 = None

        for i, step in enumerate(steps2):
            # sampled frames + sampled actions
            if i in idx_set:
                pos = idx_to_pos[i]
                obs = step["observation"]
                if image_key not in obs:
                    raise KeyError(f"image_key='{image_key}' not in observation keys: {list(obs.keys())}")
                frame_cache[pos] = obs[image_key].numpy()

                if action_key in step:
                    a_now = step[action_key]
                    if isinstance(a_now, tf.Tensor):
                        a_now = a_now.numpy()
                    action_cache[pos] = np.asarray(a_now, dtype=np.float32).reshape(-1)

            # action stats over full demo
            if compute_action_stats and (action_key in step):
                a = step[action_key]
                if isinstance(a, tf.Tensor):
                    a = a.numpy()
                a = np.asarray(a, dtype=np.float64).reshape(-1)
                a = a[:6]

                nrm = float(np.linalg.norm(a))
                sum_energy += nrm
                act_cnt += 1
                if nrm < eps_small:
                    small_cnt += 1

                if a_prev1 is not None:
                    sum_smooth += float(np.linalg.norm(a - a_prev1))
                    smooth_cnt += 1

                # if a_prev2 is not None and a_prev1 is not None:
                #     j = a - 2.0 * a_prev1 + a_prev2
                #     jerk_energy = np.sum(j ** 2)
                #     sum_jerk += jerk_energy
                #     jerk_cnt += 1
                if a_prev2 is not None and a_prev1 is not None:
                    j = a - 2.0 * a_prev1 + a_prev2
                    jerk = np.linalg.norm(j)
                    jerks.append(jerk)

                a_prev2 = a_prev1
                a_prev1 = a
            

            if action_key in step:
                a = step[action_key]
                if isinstance(a, tf.Tensor):
                    a = a.numpy()

                a = np.asarray(a, dtype=np.float32).reshape(-1)

                # a = a[:6]   # remove gripper
                actions_full.append(a)

            # early break only if not computing action stats
            if (not compute_action_stats) and all(x is not None for x in frame_cache):
                break

        if any(x is None for x in frame_cache):
            if max_episodes is not None:
                pbar.update(1)
            continue

        # fill missing sampled actions if any
        if any(a is None for a in action_cache):
            valid = [a for a in action_cache if a is not None]
            if len(valid) == 0:
                if max_episodes is not None:
                    pbar.update(1)
                continue
            act_dim = valid[0].shape[0]
            action_cache = [
                np.zeros(act_dim, dtype=np.float32) if a is None else a
                for a in action_cache
            ]

        frames = np.stack(frame_cache, axis=0)
        sampled_actions = np.stack(action_cache, axis=0)

        action_stats = {}
        if compute_action_stats and act_cnt > 0:
            action_stats = {
                "actions": np.stack(actions_full, axis=0),   # 新增
                "action_energy": float(sum_energy / max(act_cnt, 1)),
                "action_smoothness": float(sum_smooth / max(smooth_cnt, 1)) if smooth_cnt > 0 else 0.0,
                # "action_jerk": float(sum_jerk / max(jerk_cnt, 1)) if jerk_cnt > 0 else 0.0,
                "action_jerk": float(np.percentile(jerks, 90)) if len(jerks) > 0 else 0.0,
                "action_small_ratio": float(small_cnt / max(act_cnt, 1)),
            }

        raw.append((
            task_id,
            task_desc,
            T,
            frames,
            sampled_actions,
            int(ep_idx),
            native_k,
            native_v,
            action_stats,
        ))
        task_lengths.setdefault(task_id, []).append(T)

        if max_episodes is not None:
            pbar.update(1)

    if max_episodes is not None:
        pbar.close()

    task_avg_len = {tid: int(np.mean(lens)) for tid, lens in task_lengths.items()}

    demo_features = []
    for (
        task_id,
        task_desc,
        demo_len,
        frames,
        sampled_actions,
        episode_index,
        native_k,
        native_v,
        action_stats,
    ) in raw:
        item = {
            "frames": frames,
            "sampled_actions": sampled_actions,
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