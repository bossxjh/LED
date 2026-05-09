# dataeval/metric/learning_ease_v5.py
import numpy as np

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

def harmonic_mean(a, b, eps=1e-12):
    """
    Harmonic mean in [0,1] if a,b in [0,1]:
        H = 2ab/(a+b+eps)
    Strongly enforces "both must be high".
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (2.0 * a * b) / (a + b + eps)

def topk_mask(S, k: int, self_loop: bool = True):
    """
    Keep top-k per row of S; set others to -inf for softmax masking.
    If self_loop=False, diagonal will not be selected.
    """
    n = S.shape[0]
    A = S.copy()

    if not self_loop:
        np.fill_diagonal(A, -np.inf)

    if k is None or k <= 0 or k >= n:
        return A  # keep all

    # indices of top-k per row
    idx = np.argpartition(A, -k, axis=1)[:, -k:]
    mask = np.zeros_like(A, dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True

    A[~mask] = -np.inf
    return A

def row_softmax(A, temp=0.07, eps=1e-12):
    """
    Row-wise softmax with temperature; supports -inf masking.
    Smaller temp -> sharper (more peaky) distribution.
    """
    t = max(float(temp), eps)
    A = A / t
    row_max = np.max(A, axis=1, keepdims=True)
    A = A - row_max
    expA = np.exp(A)
    expA[~np.isfinite(expA)] = 0.0  # exp(-inf)=0
    Z = expA.sum(axis=1, keepdims=True)
    return expA / (Z + eps)


# ----------------------------
# Pure cosine version (harmonic + topk+softmax task transfer)
# ----------------------------
def compute_learning_ease_with_task_transfer(
        task_groups,
        beta=0.5,
        pi_scale=0.01698373,
        sigma_task=0.001,      # kept for compat (not used in pure-cos path)
        sigma_center=0.001,    # kept for compat (not used)
        transfer_mode="harmonic",  # "semantic" | "visual_center" | "harmonic"
        # ---- intra-task cosine knobs ----
        intra_sim="exp",         # "exp" or "linear"
        tau_intra=0.05,          # used if intra_sim="exp" and adaptive_tau=False
        adaptive_tau=True,       # per-task tau based on median(1-cos)
        tau_floor=1e-3,
        tau_ceiling=0.5,
        length_penalty=True,
        # ---- inter-task transfer knobs ----
        use_self_loop=True,
        alpha=1.0,               # residual mix: L_adj = (1-a)L_raw + a*(W@L_raw)
        task_knn=7,              # NEW: keep top-k neighbors per task row
        task_temp=0.11,          # NEW: softmax temperature for task graph
        eps=1e-12,
        debug=False,
):
    """
    Pure cosine geometry for SigLIP embeddings:
      - all features are L2-normalized
      - in-task similarity uses cosine (no euclidean distance)
      - task-to-task similarity uses cosine
      - task transfer modes:
          * semantic:      graph from task_embeddings
          * visual_center: graph from visual feature mean
          * harmonic:      graph weight = harmonic_mean(S_visual, S_semantic)
                           (enforces both visual & semantic similarity)

    Graph stabilization:
      - top-k sparsification (task_knn)
      - row-wise softmax with temperature (task_temp)
    This avoids fully-connected averaging / oversmoothing.

    Returns:
      dataset_score: float
      task_scores: dict[task_id] -> float
    """
    task_ids = [g["task_id"] for g in task_groups]
    if len(task_ids) == 0:
        return 0.0, {}

    L_t_raw = {}

    # always compute both centers
    task_centers_visual = {}
    task_centers_semantic = {}

    total_demo_count = sum(len(g.get("demo_lengths", [])) for g in task_groups)
    total_demo_count = max(total_demo_count, 1)

    # ----------------------------
    # Step 1: compute L_raw per task + centers
    # ----------------------------
    for g in task_groups:
        tid = g["task_id"]
        X_t = np.asarray(g["features"], dtype=np.float32)
        N_t = int(X_t.shape[0])

        # centers
        task_centers_visual[tid] = X_t.mean(axis=0).astype(np.float32) if N_t > 0 else None
        te = np.asarray(g["task_embeddings"])
        if te.ndim == 2:
            te = te.mean(axis=0)
        task_centers_semantic[tid] = np.asarray(te, dtype=np.float32)

        if N_t <= 1:
            L_t_raw[tid] = 0.0
            continue

        # normalize demos
        Xn = l2_normalize(X_t, axis=1, eps=eps)

        cos = Xn @ Xn.T  # [-1,1]
        d_cos = 1.0 - cos

        # pick tau
        if intra_sim == "exp":
            if adaptive_tau:
                tau_t = median_cosine_distance(Xn, eps=eps)
                tau_t = float(np.clip(tau_t, tau_floor, tau_ceiling))
            else:
                tau_t = float(np.clip(tau_intra, tau_floor, tau_ceiling))
            S_t = np.exp(-d_cos / (tau_t + eps))
        elif intra_sim == "linear":
            S_t = np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
            tau_t = None
        else:
            raise ValueError(f"Unknown intra_sim={intra_sim}")

        rho_t = S_t.mean(axis=1)
        E_t = float(rho_t.mean())

        if length_penalty:
            tl = float(g.get("task_length", 0.0))
            E_t = E_t / (np.log10(1.0 + max(tl, 0.0)) + 1.0)

        tri = d_cos[np.triu_indices(N_t, 1)]
        d_avg = float(tri.mean()) if tri.size > 0 else 0.0

        H = covariance_entropy_norm(Xn, eps=eps)
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
    # Step 2: build task graph W (sparse + softmax)
    # ----------------------------
    valid_tids = []
    Cv = []
    Cs = []

    for tid in task_ids:
        cv = task_centers_visual.get(tid, None)
        cs = task_centers_semantic.get(tid, None)
        if cv is None or cs is None:
            continue
        valid_tids.append(tid)
        Cv.append(np.asarray(cv, dtype=np.float32))
        Cs.append(np.asarray(cs, dtype=np.float32))

    if len(valid_tids) == 0:
        return 0.0, {}

    Cv = l2_normalize(np.stack(Cv, axis=0), axis=1, eps=eps)
    Cs = l2_normalize(np.stack(Cs, axis=0), axis=1, eps=eps)

    # similarities in [0,1]
    S_vis = (Cv @ Cv.T + 1.0) * 0.5
    S_sem = (Cs @ Cs.T + 1.0) * 0.5

    tm = str(transfer_mode).lower()
    if tm == "semantic":
        S_task = S_sem
    elif tm in ("visual_center", "visual", "vision"):
        S_task = S_vis
    elif tm in ("harmonic", "both", "and"):
        S_task = harmonic_mean(S_vis, S_sem, eps=eps)
    else:
        raise ValueError(f"Unknown transfer_mode={transfer_mode} (use semantic/visual_center/harmonic)")

    # mask top-k then softmax -> W (row-stochastic, sparse/peaky)
    A = topk_mask(S_task, k=int(task_knn), self_loop=bool(use_self_loop))
    W = row_softmax(A, temp=float(task_temp), eps=eps)

    if debug:
        nnz = (W > 0).sum(axis=1)
        ent = -np.sum(W * np.log(W + eps), axis=1)
        print("[TaskGraph] transfer_mode:", transfer_mode,
              "knn:", int(task_knn), "temp:", float(task_temp))
        print("[TaskGraph] avg nnz/row:", float(nnz.mean()),
              "entropy mean/min/max:", float(ent.mean()), float(ent.min()), float(ent.max()))
        print("[S_task stats] min/mean/max:", float(S_task.min()), float(S_task.mean()), float(S_task.max()))

    # ----------------------------
    # Step 3: propagate + residual mix + pi weighting
    # ----------------------------
    L_raw_vec = np.array([L_t_raw.get(tid, 0.0) for tid in valid_tids], dtype=np.float32)
    L_prop = W @ L_raw_vec
    L_adj_vec = (1.0 - float(alpha)) * L_raw_vec + float(alpha) * L_prop

    task_scores = {}
    for i, tid in enumerate(valid_tids):
        g = next(x for x in task_groups if x["task_id"] == tid)
        Nt = len(g.get("demo_lengths", []))
        pi_t = Nt / total_demo_count
        pi_t = float(np.tanh(pi_t / (pi_scale + eps)))
        # task_scores[tid] = float(L_adj_vec[i] * pi_t)
        task_scores[tid] = float(L_adj_vec[i])

        if debug:
            print(f"[Task {tid}] L_raw={float(L_raw_vec[i]):.4f} "
                  f"L_prop={float(L_prop[i]):.4f} L_adj={float(L_adj_vec[i]):.4f} "
                  f"pi={pi_t:.4f} score={task_scores[tid]:.4f}")

    dataset_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    return dataset_score, task_scores