import numpy as np
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
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, eps)
    p = eigvals / (eigvals.sum() + eps)
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(D + eps) + eps)
    return float(np.clip(H_norm, 0.0, 1.0))

def row_normalize(W, eps=1e-12):
    s = W.sum(axis=1, keepdims=True)
    return W / (s + eps)

def median_cosine_distance(Xn, eps=1e-12):
    """
    Xn: (N,D) normalized
    return median of (1 - cos) over upper triangle
    """
    N = Xn.shape[0]
    if N <= 1:
        return 0.1
    cos = Xn @ Xn.T
    d = 1.0 - cos
    tri = d[np.triu_indices(N, 1)]
    if tri.size == 0:
        return 0.1
    med = float(np.median(tri))
    if not np.isfinite(med):
        med = 0.1
    return max(med, eps)

# ----------------------------
# Pure cosine version
# ----------------------------
def compute_learning_ease_with_task_transfer(
        task_groups,
        beta=0.5,
        pi_scale=0.01698373,
        sigma_task=0.001,      # still supported as fallback / floor
        sigma_center=0.001,  #这个参数其实已经没有用了，可以删掉
        transfer_mode="semantic",  # "semantic" or "visual_center"
        # ---- intra-task cosine knobs ----
        intra_sim="exp",         # "exp" or "linear"
        tau_intra=0.05,          # used if intra_sim="exp"
        adaptive_tau=True,       # per-task tau based on median(1-cos)
        tau_floor=1e-3,
        tau_ceiling=0.5,
        length_penalty=True,
        # ---- inter-task transfer knobs ----
        use_self_loop=True,
        alpha=1.0,               # residual mix: L_adj = (1-a)L_raw + a*(W@L_raw). set <1 to avoid oversmooth
        eps=1e-12,
        debug=False,
):
    """
    Pure cosine geometry for SigLIP embeddings:
      - all features are L2-normalized
      - in-task similarity uses cosine (no euclidean distance)
      - task-to-task similarity uses cosine
    Returns:
      dataset_score: float
      task_scores: dict[task_id] -> float
    """
    task_ids = [g["task_id"] for g in task_groups]
    n_tasks = len(task_groups)
    if n_tasks == 0:
        return 0.0, {}

    L_t_raw = {}
    task_centers = {}

    total_demo_count = sum(len(g.get("demo_lengths", [])) for g in task_groups)
    total_demo_count = max(total_demo_count, 1)

    # ----------------------------
    # Step 1: compute L_raw per task (cosine intra-task)
    # ----------------------------
    for g in task_groups:
        tid = g["task_id"]
        X_t = np.asarray(g["features"], dtype=np.float32)
        N_t = X_t.shape[0]

        # define transfer center
        if transfer_mode == "semantic":
            te = np.asarray(g["task_embeddings"])
            if te.ndim == 2:
                te = te.mean(axis=0)
            task_centers[tid] = np.asarray(te, dtype=np.float32)
        else:
            task_centers[tid] = X_t.mean(axis=0).astype(np.float32) if N_t > 0 else None

        if N_t <= 1:
            L_t_raw[tid] = 0.0
            continue

        # normalize demos (SigLIP: cosine geometry)
        Xn = l2_normalize(X_t, axis=1, eps=eps)

        cos = Xn @ Xn.T  # [-1,1]
        # cosine distance in [0,2]
        d_cos = 1.0 - cos

        # pick tau per task if needed
        if intra_sim == "exp":
            if adaptive_tau:
                tau_t = median_cosine_distance(Xn, eps=eps)
                tau_t = float(np.clip(tau_t, tau_floor, tau_ceiling))
            else:
                tau_t = float(np.clip(tau_intra, tau_floor, tau_ceiling))

            S_t = np.exp(-d_cos / (tau_t + eps))  # (N,N) in (0,1]
        elif intra_sim == "linear":
            # map cosine to [0,1] directly
            S_t = np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
            tau_t = None
        else:
            raise ValueError(f"Unknown intra_sim={intra_sim}")

        rho_t = S_t.mean(axis=1)
        E_t = float(rho_t.mean())

        if length_penalty:
            tl = float(g.get("task_length", 0.0))
            E_t = E_t / (np.log10(1.0 + max(tl, 0.0)) + 1.0)

        # expressiveness: entropy * spread (spread also in cosine distance)
        tri = d_cos[np.triu_indices(N_t, 1)]
        d_avg = float(tri.mean()) if tri.size > 0 else 0.0

        H = covariance_entropy_norm(Xn, eps=eps)  # use normalized vectors for stable covariance
        # spread: larger avg cosine distance -> more diverse
        spread = np.tanh(d_avg / (tau_t + eps)) if (intra_sim == "exp" and tau_t is not None) else np.tanh(d_avg)
        R_t = float(np.clip(H * spread, 0.0, 1.0))

        E_t = float(np.clip(E_t, 0.0, 1.0))
        L = (R_t ** beta) * ((E_t + eps) ** (1.0 - beta))
        L_t_raw[tid] = float(L)

        if debug:
            if intra_sim == "exp":
                print(f"[Task {tid}] Nt={N_t} tau_t={tau_t:.5f} E={E_t:.4f} H={H:.4f} spread={spread:.4f} R={R_t:.4f} L_raw={L:.4f}")
            else:
                print(f"[Task {tid}] Nt={N_t} E={E_t:.4f} H={H:.4f} spread={spread:.4f} R={R_t:.4f} L_raw={L:.4f}")

    # ----------------------------
    # Step 2: task-to-task cosine similarity (centers normalized)
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

    C = l2_normalize(np.stack(centers, axis=0), axis=1, eps=eps)
    cos_task = C @ C.T  # [-1,1]
    # keep in [0,1] for weights
    S_task = (cos_task + 1.0) * 0.5

    if not use_self_loop:
        np.fill_diagonal(S_task, 0.0)

    W = row_normalize(S_task, eps=eps)

    if debug:
        print("[S_task stats] cos min/mean/max:",
              float(cos_task.min()), float(cos_task.mean()), float(cos_task.max()))
        print("[S_task stats] S in [0,1] min/mean/max:",
              float(S_task.min()), float(S_task.mean()), float(S_task.max()))

    # ----------------------------
    # Step 3: propagate + (optional) residual mix + pi weighting
    # ----------------------------
    L_raw_vec = np.array([L_t_raw.get(tid, 0.0) for tid in valid_tids], dtype=np.float32)
    L_prop = W @ L_raw_vec
    L_adj_vec = (1.0 - alpha) * L_raw_vec + alpha * L_prop

    task_scores = {}
    for i, tid in enumerate(valid_tids):
        g = next(x for x in task_groups if x["task_id"] == tid)
        Nt = len(g.get("demo_lengths", []))
        pi_t = Nt / total_demo_count
        pi_t = float(np.tanh(pi_t / (pi_scale + eps)))
        task_scores[tid] = float(L_adj_vec[i] * pi_t)

        if debug:
            print(f"[Task {tid}] L_raw={float(L_raw_vec[i]):.4f} "
                  f"L_prop={float(L_prop[i]):.4f} L_adj={float(L_adj_vec[i]):.4f} "
                  f"pi={pi_t:.4f} score={task_scores[tid]:.4f}")

    dataset_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    return dataset_score, task_scores