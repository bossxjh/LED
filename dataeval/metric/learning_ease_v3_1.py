import numpy as np
from sklearn.metrics import pairwise_distances

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

def estimate_sigma_median(X, min_sigma=1e-4, max_sigma=10.0, eps=1e-12):
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

def topk_sparsify(W, k, include_self=True):
    """
    Keep top-k entries per row (by value). Others -> 0.
    If include_self, diagonal is always kept.
    """
    T = W.shape[0]
    out = np.zeros_like(W)
    for i in range(T):
        row = W[i].copy()
        if include_self:
            row[i] = row[i] + 1e9  # force keep
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = W[i, idx]
        if include_self:
            out[i, i] = W[i, i]  # restore true diag
    return out

# ----------------------------
# v2.1
# ----------------------------
def compute_learning_ease_with_task_transfer(
        task_groups,
        beta=0.5,
        sigma_task=0.001,
        sigma_center=0.001,  #这个参数其实已经没有用了，可以删掉
        pi_scale=0.01698373,
        transfer_mode="visual_center",   # "semantic" or "visual_center"
        # ---- Step1 stability ----
        adaptive_sigma=True,
        sigma_floor=1e-4,
        sigma_ceiling=10.0,
        length_penalty=True,
        # ---- Transfer control (NEW) ----
        alpha=0.2,          # residual mix strength: 0=no transfer, 1=all transfer
        tau_cos=0.95,       # cosine threshold: only very similar tasks transfer
        gamma=4.0,          # sharpness: larger -> only extremely similar keep weight
        topk=3,             # sparsify neighbors; set None to disable
        include_self=True,  # keep self-loop
        eps=1e-12,
        debug=False,
):
    task_ids = [g["task_id"] for g in task_groups]
    n_tasks = len(task_groups)
    if n_tasks == 0:
        return 0.0, {}

    # ----------------------------
    # Step 1: L_raw + centers
    # ----------------------------
    L_t_raw = {}
    task_centers = {}

    total_demo_count = sum(len(g.get("demo_lengths", [])) for g in task_groups)
    total_demo_count = max(total_demo_count, 1)

    for g in task_groups:
        tid = g["task_id"]
        X_t = np.asarray(g["features"])
        N_t = X_t.shape[0]

        # center for transfer
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

        # adaptive sigma
        if adaptive_sigma:
            sigma_t = estimate_sigma_median(
                X_t,
                min_sigma=max(sigma_floor, sigma_task),
                max_sigma=sigma_ceiling,
                eps=eps
            )
        else:
            sigma_t = float(max(sigma_task, sigma_floor))

        d2 = pairwise_distances(X_t, X_t, metric="euclidean") ** 2
        S_t = np.exp(-d2 / (2.0 * (sigma_t ** 2) + eps))

        E_t = float(S_t.mean())
        if length_penalty:
            tl = float(g.get("task_length", 0.0))
            E_t = E_t / (np.log10(1.0 + max(tl, 0.0)) + 1.0)

        d = np.sqrt(d2)
        tri = d[np.triu_indices(N_t, 1)]
        d_avg = float(tri.mean()) if tri.size > 0 else 0.0

        H = covariance_entropy_norm(X_t, eps=eps)  # [0,1]
        spread = np.tanh(d_avg / (sigma_t + eps))  # [0,1)
        R_t = float(np.clip(H * spread, 0.0, 1.0))

        E_t = float(np.clip(E_t, 0.0, 1.0))
        L = (R_t ** beta) * ((E_t + eps) ** (1.0 - beta))
        L_t_raw[tid] = float(L)

        if debug:
            print(f"[Task {tid}] Nt={N_t} sigma_t={sigma_t:.6g} E={E_t:.4f} R={R_t:.4f} L_raw={L:.4f}")

    # ----------------------------
    # Step 2: cosine -> gate+sharpen -> sparse -> row-normalize
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

    C = l2_normalize(np.stack(centers, axis=0), axis=1, eps=eps)  # (T,D)
    cos = C @ C.T  # [-1,1]

    # gate: keep only cos >= tau_cos
    gated = np.maximum(cos - tau_cos, 0.0)  # [0, 1-tau]
    denom = (1.0 - tau_cos) + eps
    # normalize to [0,1] then sharpen
    W = (gated / denom) ** gamma

    if include_self:
        np.fill_diagonal(W, 1.0)  # self always strongest
    else:
        np.fill_diagonal(W, 0.0)

    # optional top-k sparsify (very helpful when many cos are high)
    if topk is not None and topk > 0:
        W = topk_sparsify(W, k=topk, include_self=include_self)

    W = row_normalize(W, eps=eps)

    if debug:
        # 看看过不过平滑：每行的有效邻居数、最大权重等
        nnz = (W > 0).sum(axis=1)
        print("[cos stats] min/mean/max:", float(cos.min()), float(cos.mean()), float(cos.max()))
        print("[W stats] nnz min/mean/max:", int(nnz.min()), float(nnz.mean()), int(nnz.max()))
        print("[W stats] max weight mean:", float(W.max(axis=1).mean()))

    # ----------------------------
    # Step 3: residual transfer + pi weighting
    # ----------------------------
    L_raw_vec = np.array([L_t_raw.get(tid, 0.0) for tid in valid_tids], dtype=np.float32)
    L_nei = W @ L_raw_vec
    L_adj = (1.0 - alpha) * L_raw_vec + alpha * L_nei  # ✅ prevent "all same"

    task_scores = {}
    for i, tid in enumerate(valid_tids):
        g = next(x for x in task_groups if x["task_id"] == tid)
        Nt = len(g.get("demo_lengths", []))
        pi_t = Nt / total_demo_count
        pi_t = float(np.tanh(pi_t / (pi_scale + eps)))
        task_scores[tid] = float(L_adj[i] * pi_t)

        if debug:
            print(f"[Task {tid}] L_raw={float(L_raw_vec[i]):.4f} L_nei={float(L_nei[i]):.4f} "
                  f"L_adj={float(L_adj[i]):.4f} pi={pi_t:.4f} score={task_scores[tid]:.4f}")

    dataset_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    return dataset_score, task_scores