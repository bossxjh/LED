import os
import json
import cv2
import numpy as np
from tqdm import tqdm


def covariance_entropy_norm(X, eps=1e-12):
    """
    Entropy of covariance eigenvalues.
    X: [N, D]
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0

    C = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, eps, None)

    p = eigvals / np.sum(eigvals)
    H = -np.sum(p * np.log(p))

    D = len(eigvals)
    if D <= 1:
        return 0.0
    return float(H / np.log(D))


def _safe_npz_load(npz_path):
    """
    Robustly load the main array from an .npz file.
    Priority:
      1) key named 'arr_0'
      2) if only one key exists, use that
      3) otherwise raise error
    """
    data = np.load(npz_path, allow_pickle=True)

    if "arr_0" in data.files:
        arr = data["arr_0"]
    elif len(data.files) == 1:
        arr = data[data.files[0]]
    else:
        raise KeyError(
            f"Cannot determine which key to use in {npz_path}. "
            f"Available keys: {data.files}"
        )
    return arr


def _read_video_frames(video_path):
    """
    Read all frames from mp4.
    Returns:
        frames: list of RGB uint8 images, each [H, W, 3]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def _sample_indices(T, num_frames):
    if T <= 0:
        return []
    if num_frames <= 1:
        return [0]
    if T < num_frames:
        return None
    if num_frames == 2:
        return [0, T - 1]
    return np.linspace(0, T - 1, num_frames, dtype=int).tolist()


def _infer_task_description(task_dir_name, config_path=None):
    """
    Try to infer task description from config.json first,
    otherwise use folder name.
    """
    if config_path is not None and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # try several possible keys
            for k in [
                "task_description",
                "language_instruction",
                "instruction",
                "task_name",
                "task",
                "description",
            ]:
                if k in cfg and cfg[k] is not None:
                    return str(cfg[k])
        except Exception:
            pass

    return str(task_dir_name).replace("_", " ")


def parse_meta_local_demo_dataset(
    dataset_path,
    num_frames=3,
    third_person_video="camera_250122075871.mp4",
    action_file="action.npz",
    max_episodes=None,
    add_uid=True,
    compute_action_stats=True,
    eps_small=1e-3,
    action_dims_for_stats=6,
):
    """
    Parse local embodied dataset organized like:

        dataset_path/
            run1/
                action.npz
                camera_250122075871.mp4
                ...
            run2/
                ...

    or

        dataset_path/
            put_orange_on_plate/
                run1/
                run2/
                ...

    Returns demo_features: List[dict], each contains:
      - frames: [num_frames, H, W, 3]
      - sampled_actions: [num_frames, A]
      - task_id: int
      - task_length: int
      - demo_length: int
      - task_description: str
      - actions: [T, A]
      - action_jerk: float
      - action_smoothness: float
      - action_energy: float
      - action_small_ratio: float
      - episode_index
      - demo_uid
    """

    dataset_path = os.path.abspath(dataset_path.rstrip("/"))

    # --------------------------------------------------
    # 1) collect runs
    # --------------------------------------------------
    run_dirs = []

    # case A: dataset_path itself contains run1, run2, ...
    direct_subdirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    has_direct_runs = any(d.startswith("run") for d in direct_subdirs)

    if has_direct_runs:
        task_name = os.path.basename(dataset_path)
        for d in direct_subdirs:
            if d.startswith("run"):
                run_dirs.append((task_name, os.path.join(dataset_path, d)))
    else:
        # case B: dataset_path contains task folders, each task folder contains runs
        for task_name in direct_subdirs:
            task_dir = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_dir):
                continue

            subruns = sorted([
                d for d in os.listdir(task_dir)
                if os.path.isdir(os.path.join(task_dir, d)) and d.startswith("run")
            ])
            for r in subruns:
                run_dirs.append((task_name, os.path.join(task_dir, r)))

    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f"No run directories found under {dataset_path}. "
            f"Expected run1/run2/... or task_name/run1/..."
        )

    if max_episodes is not None:
        run_dirs = run_dirs[:max_episodes]

    # --------------------------------------------------
    # 2) parse all demos
    # --------------------------------------------------
    task2id = {}
    task_lengths = {}
    raw = []

    pbar = tqdm(run_dirs, desc="Reading local demos")

    for ep_idx, (task_dir_name, run_dir) in enumerate(pbar):
        action_path = os.path.join(run_dir, action_file)
        video_path = os.path.join(run_dir, third_person_video)
        config_path = os.path.join(run_dir, "config.json")

        if not os.path.isfile(action_path):
            print(f"[Skip] Missing action file: {action_path}")
            continue
        if not os.path.isfile(video_path):
            print(f"[Skip] Missing third-person video: {video_path}")
            continue

        task_desc = _infer_task_description(task_dir_name, config_path=config_path)

        if task_desc not in task2id:
            task2id[task_desc] = len(task2id)
        task_id = task2id[task_desc]

        # ---------- load actions ----------
        try:
            actions = _safe_npz_load(action_path)
        except Exception as e:
            print(f"[Skip] Failed loading actions from {action_path}: {e}")
            continue

        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[:, None]
        elif actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)

        # ---------- load frames ----------
        try:
            video_frames = _read_video_frames(video_path)
        except Exception as e:
            print(f"[Skip] Failed reading video from {video_path}: {e}")
            continue

        T_video = len(video_frames)
        T_action = actions.shape[0]

        if T_video == 0 or T_action == 0:
            print(f"[Skip] Empty video/actions in {run_dir}")
            continue

        # Align by shortest length
        T = min(T_video, T_action)
        video_frames = video_frames[:T]
        actions = actions[:T]

        if T < num_frames:
            print(f"[Skip] Too short demo ({T} < {num_frames}) in {run_dir}")
            continue

        idxs = _sample_indices(T, num_frames)
        if idxs is None:
            print(f"[Skip] Cannot sample {num_frames} frames from T={T} in {run_dir}")
            continue

        frames = np.stack([video_frames[i] for i in idxs], axis=0)
        sampled_actions = np.stack([actions[i] for i in idxs], axis=0).astype(np.float32)

        # ---------- action stats ----------
        action_stats = {}
        if compute_action_stats:
            a_full = np.asarray(actions, dtype=np.float64)

            if a_full.ndim != 2:
                a_full = a_full.reshape(a_full.shape[0], -1)

            # optionally exclude gripper or extra dims
            if action_dims_for_stats is not None:
                a_stats = a_full[:, :action_dims_for_stats]
            else:
                a_stats = a_full

            if len(a_stats) > 0:
                norms = np.linalg.norm(a_stats, axis=1)
                action_energy = float(np.mean(norms))
                action_small_ratio = float(np.mean(norms < eps_small))

                if len(a_stats) >= 2:
                    diffs = a_stats[1:] - a_stats[:-1]
                    smoothness = float(np.mean(np.linalg.norm(diffs, axis=1)))
                else:
                    smoothness = 0.0

                if len(a_stats) >= 3:
                    jerks = a_stats[2:] - 2.0 * a_stats[1:-1] + a_stats[:-2]
                    jerk_norms = np.linalg.norm(jerks, axis=1)
                    action_jerk = float(np.percentile(jerk_norms, 90))
                else:
                    action_jerk = 0.0

                action_stats = {
                    "actions": np.asarray(actions, dtype=np.float32),
                    "action_energy": action_energy,
                    "action_smoothness": smoothness,
                    "action_jerk": action_jerk,
                    "action_small_ratio": action_small_ratio,
                }

        native_k = "run_dir"
        native_v = os.path.basename(run_dir)

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
            run_dir,
        ))

        task_lengths.setdefault(task_id, []).append(T)

    # --------------------------------------------------
    # 3) build final output
    # --------------------------------------------------
    task_avg_len = {tid: int(np.mean(lens)) for tid, lens in task_lengths.items()}

    demo_features = []
    dataset_name = os.path.basename(dataset_path)

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
        run_dir,
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
            item["demo_uid"] = f"{dataset_name}|ep={episode_index:06d}"
            item["native_episode_id_key"] = native_k
            item["native_episode_id"] = native_v
            item["run_dir"] = run_dir

        demo_features.append(item)

    return demo_features