# 我的特征层面：where i am可以就是第一帧没错。how i act需要对轨迹序列建模。what i accomplish可以是最后一帧，也可以是对task instruction的文本建模。
# 算法层面就是，in-task内部就是用图像+轨迹建模（可以是那个的输入输出的互信息，也可以是），task之间用文本形式clip举例建模（或者图像）
# 一个是在计算in-task：
# 记忆点的时候，样本内相似度，可以从最小的（输入state--输出action）这个单元去计算。【需要显示的计算输入和输出】
# R的时候可以，可以参考那个输入和输出的互信息（这个表明了这个演示点的知识质量，或者其实表示了这个空间点的概率密度）；是一个具有知识的点，那么这个点的迁移性怎么样？那我就需要考虑这些有效点的覆盖面（输入输出面）怎么样，理论上在一个全输入的空间里，是最佳的学习场（也就是完整的表示了所有输入对应所有输出的映射概率分布图，这里可以加入概率分布的离散带宽的东西（）也就是从原有的东西来恢复出缘由的表示。）。【btw这个点也可以用来筛选出关键帧】。

#task之间的迁移性：本质上是不同任务之间的范式差别的可迁移性。可以是两个任务之间的的互信息。或者是看两个任务在空间中的（state2action）概率分布的规律差异？或者是两个任务空间概率分布图的重叠率？可以是task文本上的clip举例？或者是任务类别之间的轨迹范式差别。

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import numpy as np
from sklearn.metrics import pairwise_distances

# ----------------------------
# Utils
# ----------------------------
def l2_normalize(X, axis=-1, eps=1e-12):
    denom = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / denom

def covariance_entropy_norm(X, eps=1e-12):
    """
    X: (N, D)
    return: normalized covariance spectrum entropy in [0,1]
    """
    N, D = X.shape
    if N <= 1:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)  # (D, D)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, eps)
    p = eigvals / (eigvals.sum() + eps)
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(D + eps) + eps)  # normalize to [0,1]
    return float(np.clip(H_norm, 0.0, 1.0))

def estimate_sigma_median(X, min_sigma=1e-4, max_sigma=10.0, eps=1e-12):
    """
    per-task adaptive bandwidth based on median pairwise distance (not squared).
    """
    N = X.shape[0]
    if N <= 1:
        return min_sigma
    d = pairwise_distances(X, X, metric="euclidean")
    tri = d[np.triu_indices(N, 1)]
    if tri.size == 0:
        return min_sigma
    med = np.median(tri)
    sigma = float(np.clip(med, min_sigma, max_sigma))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = min_sigma
    return sigma

def row_normalize(W, eps=1e-12):
    s = W.sum(axis=1, keepdims=True)
    return W / (s + eps)

# ----------------------------
# Stable v2
# ----------------------------
def compute_learning_ease_with_task_transfer(
        task_groups,
        beta=0.5,
        sigma_task=0.001,      # still supported as fallback / floor
        sigma_center=0.001,  #这个参数其实已经没有用了，可以删掉
        pi_scale=0.01698373,
        transfer_mode="visual_center",   # "semantic" or "visual_center"
        # ---- v2 knobs ----
        adaptive_sigma=True,
        sigma_floor=1e-4,      # lower bound for adaptive sigma
        sigma_ceiling=10.0,    # upper bound
        use_self_loop=True,    # keep self similarity (recommended)
        length_penalty=True,
        eps=1e-12,
        debug=False,
):
    """
    Returns:
      dataset_score: float
      task_scores: dict[task_id] -> float
    """
    task_ids = [g["task_id"] for g in task_groups]
    n_tasks = len(task_groups)
    if n_tasks == 0:
        return 0.0, {}

    # ----------------------------
    # Step 1: per-task L_t_raw + task center for transfer
    # ----------------------------
    L_t_raw = {}
    task_centers = {}

    # precompute demo counts for pi_t
    total_demo_count = sum(len(g.get("demo_lengths", [])) for g in task_groups)
    total_demo_count = max(total_demo_count, 1)

    for g in task_groups:
        tid = g["task_id"]
        X_t = np.asarray(g["features"])
        N_t = X_t.shape[0]

        # define task center (for transfer)
        if transfer_mode == "semantic":
            te = np.asarray(g["task_embeddings"])
            if te.ndim == 2:
                te = te.mean(axis=0)
            task_centers[tid] = te.astype(np.float32)
        else:
            task_centers[tid] = X_t.mean(axis=0).astype(np.float32) if N_t > 0 else None

        if N_t <= 1:
            L_t_raw[tid] = 0.0
            continue

        # --- adaptive sigma per task (stabilizes RBF) ---
        if adaptive_sigma:
            sigma_t = estimate_sigma_median(X_t, min_sigma=max(sigma_floor, sigma_task),
                                            max_sigma=sigma_ceiling, eps=eps)
        else:
            sigma_t = float(max(sigma_task, sigma_floor))

        # --- in-task similarity via RBF on squared distances ---
        d2 = pairwise_distances(X_t, X_t, metric="euclidean") ** 2
        S_t = np.exp(-d2 / (2.0 * (sigma_t ** 2) + eps))

        rho_t = S_t.mean(axis=1)
        E_t = float(rho_t.mean())

        if length_penalty:
            tl = float(g.get("task_length", 0.0))
            E_t = E_t / (np.log10(1.0 + max(tl, 0.0)) + 1.0)  # +1.0 avoid over-penalize & div0

        # expressiveness: entropy * spread
        d = np.sqrt(d2)
        tri = d[np.triu_indices(N_t, 1)]
        d_avg = float(tri.mean()) if tri.size > 0 else 0.0

        H = covariance_entropy_norm(X_t, eps=eps)  # [0,1]
        spread = np.tanh(d_avg / (sigma_t + eps))  # [0,1)
        R_t = float(np.clip(H * spread, 0.0, 1.0))

        # stable clamp
        E_t = float(np.clip(E_t, 0.0, 1.0))

        L = (R_t ** beta) * ((E_t + eps) ** (1.0 - beta))
        L_t_raw[tid] = float(L)

        if debug:
            print(f"[Task {tid}] Nt={N_t} sigma_t={sigma_t:.6g} "
                  f"E_t={E_t:.4f} H={H:.4f} spread={spread:.4f} R_t={R_t:.4f} L_raw={L:.4f}")

    # ----------------------------
    # Step 2: task-to-task similarity S_task (true cosine -> [0,1])
    # ----------------------------
    centers = []
    valid_tids = []
    for tid in task_ids:
        c = task_centers.get(tid, None)
        if c is None:
            continue
        centers.append(np.asarray(c, dtype=np.float32))
        valid_tids.append(tid)

    if len(valid_tids) == 0:
        return 0.0, {}

    C = np.stack(centers, axis=0)  # (T, D)
    C = l2_normalize(C, axis=1, eps=eps)         # ✅ real cosine
    cos = C @ C.T                                # [-1,1]
    S_task = (cos + 1.0) * 0.5                   # ✅ map to [0,1]

    if not use_self_loop:
        np.fill_diagonal(S_task, 0.0)

    # ✅ row-normalize for weighted average propagation
    W = row_normalize(S_task, eps=eps)

    if debug:
        print("[S_task stats] cos min/mean/max:",
              float(cos.min()), float(cos.mean()), float(cos.max()))
        print("[S_task stats] S in [0,1] min/mean/max:",
              float(S_task.min()), float(S_task.mean()), float(S_task.max()))
        print("[W stats] row-sum min/mean/max:",
              float(W.sum(axis=1).min()), float(W.sum(axis=1).mean()), float(W.sum(axis=1).max()))

    # ----------------------------
    # Step 3: transfer-adjust + pi_t weighting
    # ----------------------------
    # vectorize L_raw in valid order
    L_raw_vec = np.array([L_t_raw.get(tid, 0.0) for tid in valid_tids], dtype=np.float32)

    # propagate: L_adj = W @ L_raw (weighted average)
    L_adj_vec = W @ L_raw_vec

    task_scores = {}
    for i, tid in enumerate(valid_tids):
        g = next(x for x in task_groups if x["task_id"] == tid)
        Nt = len(g.get("demo_lengths", []))
        pi_t = Nt / total_demo_count
        pi_t = float(np.tanh(pi_t / (pi_scale + eps)))

        task_scores[tid] = float(L_adj_vec[i] * pi_t)

        if debug:
            print(f"[Task {tid}] L_adj={float(L_adj_vec[i]):.4f} pi_t={pi_t:.4f} score={task_scores[tid]:.4f}")

    dataset_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    return dataset_score, task_scores