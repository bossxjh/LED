import math
import numpy as np
import torch


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
# Torch utils
# ----------------------------
def _torch_dtype(dtype: str):
    dtype = str(dtype).lower()
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32

def _l2_normalize_t(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def _row_normalize_t(W: torch.Tensor, eps: float = 1e-12):
    return W / (W.sum(dim=1, keepdim=True) + eps)

@torch.no_grad()
def _covariance_entropy_norm_t(Xn: torch.Tensor, eps: float = 1e-12):
    """
    Xn: (N, D) torch tensor on device
    return: float in [0,1]
    Note: uses torch.linalg.eigvalsh on covariance (D,D). D=4608 can be heavy.
          If this is too slow, you can optionally compute in numpy on CPU once
          per task (or approximate with a low-rank method).
    """
    N, D = Xn.shape
    if N <= 1:
        return 0.0
    # compute covariance on CPU might be cheaper when D is large
    # but here we keep it on device for simplicity
    Xc = Xn - Xn.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / (max(N - 1, 1))  # (D,D)
    eigvals = torch.linalg.eigvalsh(cov)  # (D,)
    eigvals = torch.clamp(eigvals, min=eps)
    p = eigvals / (eigvals.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    H_norm = H / (math.log(D) + eps)
    H_norm = float(torch.clamp(H_norm, 0.0, 1.0).item())
    return H_norm

@torch.no_grad()
def _median_cosine_distance_chunked(Xn: torch.Tensor, chunk_size: int = 1024, eps: float = 1e-12):
    """
    Compute median of (1 - cos) over upper triangle WITHOUT storing full NxN.
    Xn: (N,D) normalized
    Returns float.
    """
    N, D = Xn.shape
    if N <= 1:
        return 0.1

    # Collect distances in a list of tensors (still can be big if N large).
    # But for your N~400-500, this is fine.
    # If N becomes huge, use approximate median (random sampling).
    d_list = []
    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        Xi = Xn[i0:i1]  # (b,D)
        # compute cos to ALL columns
        cos_block = Xi @ Xn.T  # (b,N)
        d_block = 1.0 - cos_block  # (b,N)
        # keep only upper triangle positions (i < j)
        # for row r in [i0,i1), valid cols are > r
        rows = torch.arange(i0, i1, device=Xn.device).unsqueeze(1)  # (b,1)
        cols = torch.arange(0, N, device=Xn.device).unsqueeze(0)    # (1,N)
        mask = cols > rows  # (b,N)
        vals = d_block[mask]
        if vals.numel() > 0:
            d_list.append(vals)

    if not d_list:
        return 0.1
    d_all = torch.cat(d_list, dim=0)
    med = torch.median(d_all).item()
    if not np.isfinite(med):
        med = 0.1
    return float(max(med, eps))

@torch.no_grad()
def _mean_exp_sim_chunked(d_cos: torch.Tensor, tau: float, chunk_size: int = 1024, eps: float = 1e-12):
    """
    Compute mean of exp(-d_cos/tau) over all pairs in NxN matrix, chunked.
    d_cos is not explicitly provided here; we compute on the fly using matmul.
    We'll instead accept Xn and do exp directly (more memory efficient).
    """
    raise NotImplementedError  # kept for clarity


# ----------------------------
# GPU-accelerated cosine version
# ----------------------------
@torch.no_grad()
def compute_learning_ease_with_task_transfer(
        task_groups,
        beta=0.5,
        pi_scale=0.01698373,
        transfer_mode="semantic",   # "semantic" or "visual_center"
        intra_sim="exp",            # "exp" or "linear"
        tau_intra=0.05,
        adaptive_tau=True,
        tau_floor=1e-3,
        tau_ceiling=0.5,
        length_penalty=True,
        use_self_loop=True,
        alpha=1.0,
        # ---- GPU knobs ----
        device="cuda",
        dtype="fp16",               # "fp16"|"bf16"|"fp32"
        chunk_size=1024,            # chunk for NxN computations
        compute_entropy_on="cpu",   # "cpu" or "gpu" (cpu often faster for D=4608)
        eps=1e-12,
        debug=False,
):
    """
    Drop-in replacement for your numpy version, but GPU-accelerated with torch.
    Returns:
      dataset_score: float
      task_scores: dict[task_id] -> float
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    tdtype = _torch_dtype(dtype)

    task_ids = [g["task_id"] for g in task_groups]
    if len(task_ids) == 0:
        return 0.0, {}

    # for pi_t
    total_demo_count = sum(len(g.get("demo_lengths", [])) for g in task_groups)
    total_demo_count = max(total_demo_count, 1)

    L_t_raw = {}
    task_centers = {}

    # ----------------------------
    # Step 1: per-task L_raw (GPU)
    # ----------------------------
    for g in task_groups:
        tid = g["task_id"]
        X_t = np.asarray(g["features"], dtype=np.float32)
        N_t = X_t.shape[0]

        # centers for transfer
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

        # move to torch
        Xt = torch.from_numpy(X_t).to(dev, dtype=tdtype)
        Xn = _l2_normalize_t(Xt, dim=1, eps=eps)  # (N,D)

        # ---- tau selection (median of 1-cos) ----
        if intra_sim == "exp":
            if adaptive_tau:
                # median in fp32 for stability
                Xn_med = Xn.to(torch.float32) if Xn.dtype != torch.float32 else Xn
                tau_t = _median_cosine_distance_chunked(Xn_med, chunk_size=chunk_size, eps=eps)
                tau_t = float(np.clip(tau_t, tau_floor, tau_ceiling))
            else:
                tau_t = float(np.clip(tau_intra, tau_floor, tau_ceiling))
        else:
            tau_t = None

        # ---- E_t: mean similarity ----
        # We compute mean over all entries of S_t = exp(-(1-cos)/tau) or linear map
        # in chunks to reduce peak memory
        sum_sim = 0.0
        cnt = 0

        # For spread: need mean of d_cos over upper triangle (i<j)
        sum_d = 0.0
        cnt_d = 0

        # (optional) covariance entropy
        if compute_entropy_on == "cpu":
            H = covariance_entropy_norm(np.asarray(Xn.to(torch.float32).cpu().numpy()), eps=eps)
        else:
            Xn_e = Xn.to(torch.float32) if Xn.dtype != torch.float32 else Xn
            H = _covariance_entropy_norm_t(Xn_e, eps=eps)

        N = N_t
        Xn_full = Xn  # (N,D)

        for i0 in range(0, N, chunk_size):
            i1 = min(N, i0 + chunk_size)
            Xi = Xn_full[i0:i1]           # (b,D)
            cos_block = Xi @ Xn_full.T    # (b,N)
            if intra_sim == "exp":
                d_block = 1.0 - cos_block
                S_block = torch.exp(-d_block / (tau_t + eps))
            elif intra_sim == "linear":
                S_block = torch.clamp((cos_block + 1.0) * 0.5, 0.0, 1.0)
                d_block = 1.0 - cos_block
            else:
                raise ValueError(f"Unknown intra_sim={intra_sim}")

            sum_sim += float(S_block.sum().item())
            cnt += int(S_block.numel())

            # upper triangle stats for d_avg
            rows = torch.arange(i0, i1, device=dev).unsqueeze(1)
            cols = torch.arange(0, N, device=dev).unsqueeze(0)
            mask = cols > rows
            vals = d_block[mask]
            if vals.numel() > 0:
                sum_d += float(vals.sum().item())
                cnt_d += int(vals.numel())

        E_t = sum_sim / max(cnt, 1)
        if length_penalty:
            tl = float(g.get("task_length", 0.0))
            E_t = E_t / (np.log10(1.0 + max(tl, 0.0)) + 1.0)
        E_t = float(np.clip(E_t, 0.0, 1.0))

        d_avg = (sum_d / max(cnt_d, 1)) if cnt_d > 0 else 0.0
        if intra_sim == "exp" and tau_t is not None:
            spread = float(np.tanh(d_avg / (tau_t + eps)))
        else:
            spread = float(np.tanh(d_avg))

        R_t = float(np.clip(H * spread, 0.0, 1.0))

        L = (R_t ** beta) * ((E_t + eps) ** (1.0 - beta))
        L_t_raw[tid] = float(L)

        if debug:
            if intra_sim == "exp":
                print(f"[Task {tid}] Nt={N_t} tau_t={tau_t:.5f} E={E_t:.4f} H={H:.4f} spread={spread:.4f} R={R_t:.4f} L_raw={L:.4f}")
            else:
                print(f"[Task {tid}] Nt={N_t} E={E_t:.4f} H={H:.4f} spread={spread:.4f} R={R_t:.4f} L_raw={L:.4f}")

    # ----------------------------
    # Step 2: task-to-task cosine (small, do on GPU quickly)
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

    C_np = np.stack(centers, axis=0).astype(np.float32)
    Ct = torch.from_numpy(C_np).to(dev, dtype=tdtype)
    Ct = _l2_normalize_t(Ct, dim=1, eps=eps)
    cos_task = Ct @ Ct.T
    S_task = (cos_task + 1.0) * 0.5

    if not use_self_loop:
        S_task.fill_diagonal_(0.0)

    W = _row_normalize_t(S_task, eps=eps)

    if debug:
        cos_min = float(cos_task.min().item())
        cos_mean = float(cos_task.mean().item())
        cos_max = float(cos_task.max().item())
        print("[S_task stats] cos min/mean/max:", cos_min, cos_mean, cos_max)

    # ----------------------------
    # Step 3: propagate + residual + pi
    # ----------------------------
    L_raw_vec = torch.tensor([L_t_raw.get(tid, 0.0) for tid in valid_tids], device=dev, dtype=torch.float32)
    L_prop = W.to(torch.float32) @ L_raw_vec
    L_adj = (1.0 - alpha) * L_raw_vec + alpha * L_prop

    task_scores = {}
    for i, tid in enumerate(valid_tids):
        g = next(x for x in task_groups if x["task_id"] == tid)
        Nt = len(g.get("demo_lengths", []))
        pi_t = Nt / total_demo_count
        pi_t = float(np.tanh(pi_t / (pi_scale + eps)))
        task_scores[tid] = float(L_adj[i].item() * pi_t)

        if debug:
            print(f"[Task {tid}] L_raw={float(L_raw_vec[i].item()):.4f} "
                  f"L_prop={float(L_prop[i].item()):.4f} L_adj={float(L_adj[i].item()):.4f} "
                  f"pi={pi_t:.4f} score={task_scores[tid]:.4f}")

    dataset_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0
    return dataset_score, task_scores