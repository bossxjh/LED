# # # dataeval/metric/baselines.py
# dataeval/metric/baselines.py
import numpy as np

def l2_normalize(X, axis=-1, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    denom = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / denom

def vision_action_consistency_score(V, A, eps=1e-12):
    """
    V: (M, F) visual features
    A: (M, 6) action vectors

    consistency = corr( cos(v_i,v_j), cos(a_i,a_j) )
    """

    M = V.shape[0]
    if M < 3:
        return 0.0

    # normalize
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)

    # cosine matrices
    Sv = Vn @ Vn.T
    Sa = An @ An.T

    # remove diagonal
    mask = ~np.eye(M, dtype=bool)

    vals_v = Sv[mask]
    vals_a = Sa[mask]

    if vals_v.size < 10:
        return 0.0

    # Pearson correlation
    corr = np.corrcoef(vals_v, vals_a)[0, 1]

    if np.isnan(corr):
        return 0.0

    return float(corr)

def covariance_entropy_norm(X, eps=1e-12):
    """
    X: (N, D)
    normalized covariance spectrum entropy in [0,1]
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    if N < 3:
        return 0.0

    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
        eigvals = (s * s) / max(N - 1, 1)
    except np.linalg.LinAlgError:
        return 0.0

    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= eps:
        return 0.0

    p = eigvals / (total + eps)
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    Hn = H / np.log(len(p) + eps)
    return float(np.clip(Hn, 0.0, 1.0))

def demo_level_trajectory_entropy(actions_list, eps=1e-12):
    """
    Compute demo-level trajectory entropy safely
    actions_list: list of demos, each [T, 7]
    """
    # 找最短 demo 长度
    lengths = [a.shape[0] for a in actions_list if a is not None and a.shape[0] > 0]
    if len(lengths) < 2:
        return 0.0
    min_len = min(lengths)

    traj_vecs = []
    for a in actions_list:
        if a is None or a.shape[0] < 1:
            continue
        a = np.asarray(a)
        # 截断到最短长度，flatten
        traj_vecs.append(a[:min_len, :7].reshape(-1))

    if len(traj_vecs) < 2:
        return 0.0

    traj_vecs = np.stack(traj_vecs, axis=0)
    return covariance_entropy_norm(traj_vecs, eps=eps)


def demo_level_visual_entropy(sample_features, idx, eps=1e-12):
    """
    Compute demo-level visual entropy safely
    sample_features: [N, K, F]
    idx: indices of demos to consider
    """
    # 找最短 demo 帧数
    lengths = [sample_features[i].shape[0] for i in idx if sample_features[i] is not None and sample_features[i].shape[0] > 0]
    if len(lengths) < 2:
        return 0.0
    min_len = min(lengths)

    V_demo = []
    for i in idx:
        v = sample_features[i]
        if v is None or v.shape[0] < 1:
            continue
        # 截断到最短长度，flatten
        V_demo.append(v[:min_len].reshape(-1))

    if len(V_demo) < 2:
        return 0.0

    V_demo = np.stack(V_demo, axis=0)
    return covariance_entropy_norm(V_demo, eps=eps)
def compute_baselines_from_npzdata(
    npzdata,
    *,
    eps=1e-12,
    max_pair_samples=50000,
    rng_seed=0
):
    """
    Baselines:

    1. dataset_scale
    2. trajectory_quality (demo-level entropy)
    3. visual_coverage   (demo-level entropy)
    4. vision_action_consistency (ALL sampled frames)

    Required npz keys:

    task_ids
    demo_lengths
    actions
    sample_features   [N, K, F]
    sample_actions    [N, K, A]
    """

    rng = np.random.default_rng(rng_seed)

    task_ids = np.asarray(npzdata["task_ids"]).astype(np.int64)
    demo_lengths = np.asarray(npzdata["demo_lengths"]).astype(np.float64)
    actions = npzdata["actions"]   # object array: N demos, each [T,A]
    sample_features = np.asarray(npzdata["sample_features"])  # [N,K,F]
    sample_actions = np.asarray(npzdata["sample_actions"])    # [N,K,A]

    unique_tids = sorted(set(task_ids.tolist()))
    per_task = {}

    for tid in unique_tids:
        idx = np.where(task_ids == tid)[0]
        if len(idx) == 0:
            continue

        # --------------------------------------------------
        # (2) Trajectory quality (demo-level entropy)
        # --------------------------------------------------
        trajs = [actions[i] for i in idx]
        trajectory_quality = demo_level_trajectory_entropy(trajs, eps=eps)

        # --------------------------------------------------
        # (3) Visual coverage (demo-level entropy)
        # --------------------------------------------------
        visual_coverage = demo_level_visual_entropy(sample_features, idx, eps=eps)

        # --------------------------------------------------
        # (4) Vision-action consistency
        # use ALL sampled frames
        # --------------------------------------------------
        V = sample_features[idx].reshape(-1, sample_features.shape[-1])
        A = sample_actions[idx].reshape(-1, sample_actions.shape[-1])
        A = A[:, :7]  # only first 7 dims

        # subsample for efficiency
        if V.shape[0] > max_pair_samples:
            sel = rng.choice(V.shape[0], size=max_pair_samples, replace=False)
            V = V[sel]
            A = A[sel]

        vision_action_consistency = vision_action_consistency_score(V, A, eps=eps)

        # --------------------------------------------------
        # (1) Dataset scale
        # --------------------------------------------------
        dataset_scale = float(len(idx))

        per_task[int(tid)] = {
            "dataset_scale": dataset_scale,
            "trajectory_quality": trajectory_quality,
            "visual_coverage": visual_coverage,
            "vision_action_consistency": vision_action_consistency,
            "num_demos": int(len(idx)),
        }

    return {
        "baseline_per_task": per_task,
        "baseline_name_list": [
            "dataset_scale",
            "trajectory_quality",
            "visual_coverage",
            "vision_action_consistency",
        ],
    }
# import numpy as np


# def l2_normalize(X, axis=-1, eps=1e-12):
#     X = np.asarray(X, dtype=np.float64)
#     denom = np.linalg.norm(X, axis=axis, keepdims=True) + eps
#     return X / denom

# def demo_level_trajectory_entropy(actions_list, eps=1e-12):
#     """
#     actions_list: list of demos, each [T, 7]
#     Returns: normalized covariance spectrum entropy across demos
#     """
#     traj_vecs = []
#     for a in actions_list:
#         if a is None or a.shape[0] < 1:
#             continue
#         a = np.asarray(a)
#         traj_vecs.append(a[:, :7].reshape(-1))  # flatten demo

#     if len(traj_vecs) < 2:
#         return 0.0

#     traj_vecs = np.stack(traj_vecs, axis=0)
#     return covariance_entropy_norm(traj_vecs, eps=eps)


# def demo_level_visual_entropy(sample_features, idx, eps=1e-12):
#     """
#     sample_features: [N, K, F]
#     idx: indices of demos to consider
#     """
#     V_demo = []
#     for i in idx:
#         v = sample_features[i]  # [K, F]
#         v_flat = v.reshape(-1)
#         V_demo.append(v_flat)

#     if len(V_demo) < 2:
#         return 0.0

#     V_demo = np.stack(V_demo, axis=0)
#     return covariance_entropy_norm(V_demo, eps=eps)

# def pairwise_visual_diversity(features_list, eps=1e-12, L=None):
#     """
#     features_list: list of demos, each [T, F] or [K, F]
#     flatten每个demo的frame特征 -> demo-level向量，然后计算pairwise cosine diversity
#     """
#     demo_vecs = []

#     # 可选：用最短demo的frame数做截断
#     min_len = min([f.shape[0] for f in features_list if f is not None])
#     use_len = min_len if L is None else min(L, min_len)

#     for f in features_list:
#         if f is None or f.shape[0] < 1:
#             continue
#         f = np.asarray(f)
#         f_flat = f[:use_len, :].reshape(-1)  # flatten demo内所有frame
#         demo_vecs.append(f_flat)

#     if len(demo_vecs) < 2:
#         return 0.0

#     demo_vecs = np.stack(demo_vecs, axis=0)
#     demo_vecs = demo_vecs / (np.linalg.norm(demo_vecs, axis=1, keepdims=True) + eps)

#     S = demo_vecs @ demo_vecs.T
#     N = S.shape[0]
#     mask = np.triu(np.ones((N, N), dtype=bool), k=1)
#     vals = S[mask]
#     diversity = float(np.mean(1.0 - vals))
#     return diversity

# def pairwise_trajectory_diversity(actions_list, eps=1e-12, L=None):
#     traj_vecs = []
#     # 可选：取最短轨迹长度
#     min_len = min([a.shape[0]-1 for a in actions_list if a is not None])
#     use_len = min_len if L is None else min(L, min_len)

#     for a in actions_list:
#         if a is None or a.shape[0] < 2:
#             continue
#         a = np.asarray(a)
#         traj = a[:use_len, :7]  # 取前7维，按 use_len 截断
#         traj_vecs.append(traj.reshape(-1))

#     if len(traj_vecs) < 2:
#         return 0.0

#     traj_vecs = np.stack(traj_vecs, axis=0)
#     traj_vecs = traj_vecs / (np.linalg.norm(traj_vecs, axis=1, keepdims=True) + eps)

#     S = traj_vecs @ traj_vecs.T
#     N = S.shape[0]
#     mask = np.triu(np.ones((N, N), dtype=bool), k=1)
#     vals = S[mask]
#     diversity = float(np.mean(1.0 - vals))
#     return diversity
# def visual_coverage_score(V, eps=1e-12):
#     """
#     V: (N, F) demo-level visual features
#     return average pairwise cosine distance
#     """
#     N = V.shape[0]
#     if N < 2:
#         return 0.0

#     # L2 normalize
#     V = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)

#     # cosine similarity matrix
#     S = V @ V.T

#     # convert to cosine distance
#     D = 1.0 - S

#     mask = ~np.eye(N, dtype=bool)
#     return float(np.mean(D[mask]))

# def covariance_entropy_norm(X, eps=1e-12):
#     """
#     X: (N, D)
#     normalized covariance spectrum entropy in [0,1]
#     """
#     X = np.asarray(X, dtype=np.float64)

#     N, D = X.shape
#     if N < 3:
#         return 0.0

#     Xc = X - X.mean(axis=0, keepdims=True)

#     try:
#         s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
#         eigvals = (s * s) / max(N - 1, 1)
#     except np.linalg.LinAlgError:
#         return 0.0

#     eigvals = np.clip(eigvals, 0.0, None)

#     total = eigvals.sum()
#     if total <= eps:
#         return 0.0

#     p = eigvals / (total + eps)
#     p = np.clip(p, eps, 1.0)

#     H = -np.sum(p * np.log(p))
#     Hn = H / np.log(len(p) + eps)

#     return float(np.clip(Hn, 0.0, 1.0))

# def vision_action_consistency_score(V, A, eps=1e-12):
#     """
#     V: (M, F) visual features
#     A: (M, 6) action vectors

#     consistency = corr( cos(v_i,v_j), cos(a_i,a_j) )
#     """

#     M = V.shape[0]
#     if M < 3:
#         return 0.0

#     # normalize
#     Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
#     An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)

#     # cosine matrices
#     Sv = Vn @ Vn.T
#     Sa = An @ An.T

#     # remove diagonal
#     mask = ~np.eye(M, dtype=bool)

#     vals_v = Sv[mask]
#     vals_a = Sa[mask]

#     if vals_v.size < 10:
#         return 0.0

#     # Pearson correlation
#     corr = np.corrcoef(vals_v, vals_a)[0, 1]

#     if np.isnan(corr):
#         return 0.0

#     return float(corr)


# def action_entropy_score(actions, eps=1e-12):
#     """
#     actions: (T, A)

#     compute entropy of delta-action covariance
#     """

#     actions = np.asarray(actions, dtype=np.float64)

#     if actions.shape[0] < 3:
#         return 0.0

#     # motion signal
#     dA = actions[1:] - actions[:-1]

#     return covariance_entropy_norm(dA)


# # def vision_action_consistency_score(V, A, eps=1e-12):
# #     """
# #     V: (M, F) visual features
# #     A: (M, 6) action vectors (only first 6 dims)

# #     score = mean_{i!=j} cos(v_i,v_j) * cos(a_i,a_j)
# #     """

# #     M = V.shape[0]
# #     if M < 2:
# #         return 0.0

# #     Vn = l2_normalize(V, axis=1, eps=eps)
# #     An = l2_normalize(A, axis=1, eps=eps)

# #     Sv = Vn @ Vn.T
# #     Sa = An @ An.T

# #     mask = ~np.eye(M, dtype=bool)

# #     vals = (Sv * Sa)[mask]

# #     if vals.size == 0:
# #         return 0.0

# #     return float(np.mean(vals))


# def compute_baselines_from_npzdata(
#     npzdata,
#     *,
#     eps=1e-12,
#     max_pair_samples=50000,
#     rng_seed=0
# ):
#     """
#     Baselines:

#     1. dataset_scale
#     2. trajectory_quality
#     3. visual_coverage  (FIRST FRAME ONLY)
#     4. vision_action_consistency (ALL SAMPLED FRAMES)

#     Required npz keys:

#     task_ids
#     demo_lengths
#     action_jerk
#     sample_features   [N, K, F]
#     sample_actions    [N, K, A]
#     """

#     rng = np.random.default_rng(rng_seed)

#     task_ids = np.asarray(npzdata["task_ids"]).astype(np.int64)
#     demo_lengths = np.asarray(npzdata["demo_lengths"]).astype(np.float64)

#     action_jerk = np.asarray(npzdata["action_jerk"]).astype(np.float64)
#     action_smoothness = np.asarray(npzdata["action_smoothness"]).astype(np.float64)
#     action_small_ratio =np.asarray(npzdata["action_small_ratio"]).astype(np.float64)

#     actions = npzdata["actions"]   # object array: N demos, each [T,A]

#     sample_features = np.asarray(npzdata["sample_features"])  # [N,K,F]
#     sample_actions = np.asarray(npzdata["sample_actions"])    # [N,K,A]

#     unique_tids = sorted(set(task_ids.tolist()))

#     per_task = {}

#     for tid in unique_tids:

#         idx = np.where(task_ids == tid)[0]

#         if len(idx) == 0:
#             continue


#         # --------------------------------------------------
#         # (2) Trajectory quality
#         # --------------------------------------------------

#         # trajectory_quality = -float(np.mean(action_jerk[idx]))

#         # all_dA = []

#         # for i in idx:
#         #     a = actions[i]

#         #     if a.shape[0] < 2:
#         #         continue

#         #     dA = a[1:] - a[:-1]

#         #     all_dA.append(dA[:, :7])

#         # if len(all_dA) > 0:
#         #     all_dA = np.concatenate(all_dA, axis=0)
#         #     trajectory_quality = covariance_entropy_norm(all_dA)
#         # else:
#         #     trajectory_quality = 0.0
#         # all_A = [actions[i] for i in idx if actions[i] is not None]
#         # trajectory_quality = pairwise_trajectory_diversity(all_A)
#         trajectory_quality = demo_level_trajectory_entropy([actions[i] for i in idx])

#         # all_A = []

#         # for i in idx:

#         #     a = actions[i]

#         #     if a is None:
#         #         continue

#         #     a = np.asarray(a)

#         #     if a.ndim != 2:
#         #         continue

#         #     all_A.append(a[:, :7])   # 只用 arm action


#         # if len(all_A) > 0:

#         #     all_A = np.concatenate(all_A, axis=0)

#         #     trajectory_quality = covariance_entropy_norm(all_A)

#         # else:

#         #     trajectory_quality = 0.0
#         # divs = []

#         # for i in idx:
#         #     a = actions[i]

#         #     if a is None:
#         #         continue

#         #     a = np.asarray(a)

#         #     if a.ndim != 2:
#         #         continue

#         #     div = action_entropy_score(a[:, :7])
#         #     divs.append(div)

#         # trajectory_quality = float(np.mean(divs)) if len(divs) > 0 else 0.0

#         # --------------------------------------------------
#         # (3) Visual coverage
#         # FIRST FRAME ONLY
#         # --------------------------------------------------
#         # demo-level视觉特征
#         # all_V = [sample_features[i] for i in idx]  # 每个demo的 [K, F]

#         # visual_coverage = pairwise_visual_diversity(all_V)
#         visual_coverage = demo_level_visual_entropy(sample_features, idx)

#         # V0 = sample_features[idx, 0, :]  # [Nt, F]
#         # # V0 = sample_features[idx].reshape(-1, sample_features.shape[-1])

#         # V0 = l2_normalize(V0, axis=1, eps=eps)

#         # visual_coverage = covariance_entropy_norm(V0)

#         # --------------------------------------------------
#         # (4) Vision-action consistency
#         # use ALL sampled frames
#         # --------------------------------------------------

#         V = sample_features[idx].reshape(-1, sample_features.shape[-1])

#         A = sample_actions[idx].reshape(-1, sample_actions.shape[-1])

#         # only first 6 dims
#         A = A[:, :7]

#         # subsample for efficiency
#         if V.shape[0] > max_pair_samples:

#             sel = rng.choice(V.shape[0], size=max_pair_samples, replace=False)

#             V = V[sel]
#             A = A[sel]

#         vision_action_consistency = vision_action_consistency_score(V, A, eps=eps)


#         # --------------------------------------------------
#         # (1) Dataset scale
#         # --------------------------------------------------

#         dataset_scale = float(len(idx))

#         per_task[int(tid)] = {
#             "dataset_scale": dataset_scale,
#             "trajectory_quality": trajectory_quality,
#             "visual_coverage": visual_coverage,
#             "vision_action_consistency": vision_action_consistency,
#             "num_demos": int(len(idx)),
#         }

#     return {
#         "baseline_per_task": per_task,
#         "baseline_name_list": [
#             "dataset_scale",
#             "trajectory_quality",
#             "visual_coverage",
#             "vision_action_consistency",
#         ],
#     }
# # import numpy as np

# # def _to_1d_feature(f: np.ndarray) -> np.ndarray:
# #     """
# #     Make sure each demo feature is a 1D vector (F,).
# #     Supports:
# #       - (F,)
# #       - (K, F)  -> mean over K
# #       - (F, K)  -> mean over K
# #       - (K, ..., F): flatten per-slice then mean over first axis if it looks like multi-frame.
# #     """
# #     f = np.asarray(f)

# #     if f.ndim == 1:
# #         return f.astype(np.float64)

# #     if f.ndim == 2:
# #         a, b = f.shape
# #         if a <= 8 and b > a:
# #             return f.mean(axis=0).astype(np.float64)   # (F,)
# #         if b <= 8 and a > b:
# #             return f.mean(axis=1).astype(np.float64)   # (F,)
# #         return f.reshape(-1).astype(np.float64)

# #     if f.ndim >= 3 and f.shape[0] <= 8:
# #         K = f.shape[0]
# #         ff = f.reshape(K, -1).mean(axis=0)
# #         return ff.astype(np.float64)

# #     return f.reshape(-1).astype(np.float64)


# # def l2_normalize(X, axis=-1, eps=1e-12):
# #     X = np.asarray(X, dtype=np.float64)
# #     denom = np.linalg.norm(X, axis=axis, keepdims=True) + eps
# #     return X / denom


# # def covariance_entropy_norm(X, eps=1e-12):
# #     """
# #     X: (N, D) (can be normalized)
# #     Return normalized covariance spectrum entropy in [0,1].
# #     """
# #     X = np.asarray(X, dtype=np.float64)
# #     N, D = X.shape
# #     if N < 3:
# #         return 0.0

# #     Xc = X - X.mean(axis=0, keepdims=True)
# #     try:
# #         s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
# #         eigvals = (s * s) / max(N - 1, 1)
# #     except np.linalg.LinAlgError:
# #         return 0.0

# #     eigvals = np.clip(eigvals, 0.0, None)
# #     total = float(eigvals.sum())
# #     if total <= eps:
# #         return 0.0

# #     p = eigvals / (total + eps)
# #     p = np.clip(p, eps, 1.0)
# #     H = float(-np.sum(p * np.log(p)))
# #     Hn = H / float(np.log(len(p) + eps))
# #     return float(max(0.0, min(1.0, Hn)))


# # def mean_distance_to_centroid_cos(Xn):
# #     """
# #     Xn: (N,D) L2 normalized
# #     Use cosine distance to centroid: 1 - cos(x_i, mu)
# #     """
# #     N = Xn.shape[0]
# #     if N < 2:
# #         return 0.0
# #     mu = Xn.mean(axis=0, keepdims=True)
# #     mu = l2_normalize(mu, axis=1)
# #     cos = (Xn * mu).sum(axis=1)
# #     d = 1.0 - cos
# #     return float(np.mean(d))


# # def mean_top1_cos_sim(Xn):
# #     """
# #     Xn: (N,D) L2 normalized
# #     Redundancy = mean_i max_{j!=i} cos(x_i, x_j)
# #     """
# #     N = Xn.shape[0]
# #     if N < 2:
# #         return 0.0
# #     S = Xn @ Xn.T
# #     np.fill_diagonal(S, -np.inf)
# #     top1 = np.max(S, axis=1)
# #     top1 = np.where(np.isfinite(top1), top1, 0.0)
# #     return float(np.mean(top1))


# # def compute_baselines_from_npzdata(npzdata, *, eps=1e-12):
# #     """
# #     Compute task-level dataset-stat baselines from an NPZ.

# #     Requires:
# #       - features: (N, ...) demo-level features (may be (F,), (K,F), (K,*,*))
# #       - task_ids: (N,)
# #       - demo_lengths: (N,)  (steps per demo)

# #     Optional:
# #       - action_jerk: (N,) per-demo jerk (if missing -> zeros)

# #     Returns:
# #       {
# #         "baseline_per_task": {tid: {...}},
# #         "baseline_name_list": [...],
# #       }
# #     """
# #     feats = np.asarray(npzdata["features"])
# #     task_ids = np.asarray(npzdata["task_ids"]).astype(np.int64)

# #     # steps per demo
# #     demo_lengths = np.asarray(npzdata["demo_lengths"]).astype(np.float64)

# #     action_jerk = npzdata["action_jerk"] if "action_jerk" in npzdata.files else None
# #     if action_jerk is None:
# #         action_jerk = np.zeros_like(demo_lengths, dtype=np.float64)
# #     else:
# #         action_jerk = np.asarray(action_jerk).astype(np.float64)

# #     unique_tids = sorted(set(task_ids.tolist()))
# #     per_task = {}

# #     for tid in unique_tids:
# #         idx = np.where(task_ids == tid)[0]
# #         if len(idx) == 0:
# #             per_task[int(tid)] = {
# #                 "dataset_steps": 0.0,
# #                 "mean_action_jerk": 0.0,
# #                 "traj_consistency": 0.0,
# #                 "visual_redundancy": 0.0,
# #                 "cov_entropy": 0.0,
# #                 "num_demos": 0,
# #             }
# #             continue

# #         # demo embeddings -> (Nt, F)
# #         X = np.stack([_to_1d_feature(feats[i]) for i in idx], axis=0)
# #         Xn = l2_normalize(X, axis=1, eps=eps)

# #         # (A) Dataset scale baseline: total steps in this task subset
# #         # dataset_steps = float(np.sum(demo_lengths[idx])/task_lengths[idx])
# #         dataset_steps = float(np.size(idx))

# #         # (B) Motion statistic
# #         mean_jerk = float(np.mean(action_jerk[idx]))

# #         # (C) Visual structure stats (on embeddings)
# #         traj_cons = mean_distance_to_centroid_cos(Xn)
# #         redundancy = mean_top1_cos_sim(Xn)
# #         cov_ent = covariance_entropy_norm(Xn, eps=eps)

# #         per_task[int(tid)] = {
# #             "dataset_steps": dataset_steps,
# #             "mean_action_jerk": mean_jerk,
# #             "traj_consistency": traj_cons,
# #             "visual_redundancy": redundancy,
# #             "cov_entropy": cov_ent,
# #             "num_demos": int(len(idx)),
# #         }

# #     return {
# #         "baseline_per_task": per_task,
# #         "baseline_name_list": [
# #             "dataset_steps",
# #             "mean_action_jerk",
# #             "traj_consistency",
# #             "visual_redundancy",
# #             "cov_entropy",
# #         ],
# #     }