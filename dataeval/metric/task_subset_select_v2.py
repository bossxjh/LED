#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upgraded selector for per-task subsets under fixed budget m=ceil(ratio*N):

  (A) BEST subset: maximize L_raw
      best = max( multi-start greedy , random-max search )
      -> output: filtered_r{ratio}.npz

IMPORTANT (per your definition):
  task_length(S) = mean(demo_length of selected demos in this task)

Input npz expected keys (per-demo arrays length N):
  - features (N,D) or (N,T,D)
  - task_ids (N,)
  - task_lengths (N,)                (will be overwritten in filtered npz)
  - demo_lengths (N,)
  - task_descriptions (N,) (object)
  - task_embeddings (N,E) (optional, not used here)

Optional unique id keys:
  - episode_indices (N,)
  - demo_uids (N,)

Outputs in --out_dir:
  - filtered_r{ratio}.npz
  - selection_report.json

Example:
  python -m dataeval.metric.select_subset_cli \
    --in_npz /path/in.npz \
    --out_dir /path/out \
    --ratios 0.8,0.6,0.4,0.2 \
    --best_restarts 10 \
    --random_max_samples 200000 \
    --random_max_patience 30000 \
    --seed 0 \
    --use_fixed_tau_for_search \
    --beta 0.5 \
    --tau_floor 0.03
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dataeval.metric.utils.task_grouping import group_by_task
from dataeval.metric.learning_ease_v5 import l2_normalize, median_cosine_distance


# =========================================================
# Helpers
# =========================================================
def subset_task_length_from_demo_lengths(demo_lengths_task: np.ndarray,
                                        idx_local: List[int] | np.ndarray) -> float:
    """task_length(S) = mean(demo_length of selected demos)."""
    idx = np.asarray(idx_local, dtype=np.int64)
    if idx.size == 0:
        return 0.0
    vals = np.asarray(demo_lengths_task, dtype=np.float32)[idx]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _cov_entropy_fast(Xn: np.ndarray, eps: float = 1e-12) -> float:
    """
    Fast covariance spectrum entropy using Gram matrix (N,N),
    normalized by log(D) to match your original definition.
    """
    N, D = Xn.shape
    if N <= 1:
        return 0.0
    Xc = Xn - Xn.mean(axis=0, keepdims=True)
    G = (Xc @ Xc.T) / max(N - 1, 1)
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, eps)
    p = eigvals / (eigvals.sum() + eps)
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(D + eps) + eps)
    return float(np.clip(H_norm, 0.0, 1.0))


# =========================================================
# 1) L_raw for subset
# =========================================================
def compute_L_raw_for_subset(
    X: np.ndarray,
    idx_local: List[int] | np.ndarray,
    *,
    beta: float = 0.5,
    intra_sim: str = "exp",         # "exp" | "linear"
    adaptive_tau: bool = True,
    tau_intra: float = 0.05,
    tau_floor: float = 0.03,
    tau_ceiling: float = 0.5,
    length_penalty: bool = True,
    task_length: float = 0.0,
    fixed_tau: Optional[float] = None,
    eps: float = 1e-12,
) -> float:
    idx = np.asarray(idx_local, dtype=np.int64)
    Xs = np.asarray(X, dtype=np.float32)[idx]
    N, D = Xs.shape
    if N <= 1:
        return 0.0

    Xn = l2_normalize(Xs, axis=1, eps=eps)
    cos = Xn @ Xn.T
    d_cos = 1.0 - cos

    if intra_sim == "exp":
        if fixed_tau is not None:
            tau_t = float(fixed_tau)
        else:
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
        E_t = E_t / (np.log10(1.0 + max(float(task_length), 0.0)) + 1.0)

    tri = d_cos[np.triu_indices(N, 1)]
    d_avg = float(tri.mean()) if tri.size > 0 else 0.0

    H = _cov_entropy_fast(Xn, eps=eps)
    if intra_sim == "exp" and tau_t is not None:
        spread = float(np.tanh(d_avg / (tau_t + eps)))
    else:
        spread = float(np.tanh(d_avg))

    R_t = float(np.clip(H * spread, 0.0, 1.0))
    E_t = float(np.clip(E_t, 0.0, 1.0))

    L = (R_t ** beta) * ((E_t + eps) ** (1.0 - beta))
    return float(L)


# =========================================================
# 2) Greedy selection per task (maximize L_raw)
# =========================================================
def greedy_select_task_subset(
    X: np.ndarray,
    demo_lengths_task: np.ndarray,
    m: int,
    *,
    beta: float = 0.5,
    intra_sim: str = "exp",
    adaptive_tau: bool = True,
    tau_intra: float = 0.05,
    tau_floor: float = 0.03,
    tau_ceiling: float = 0.5,
    length_penalty: bool = True,
    use_fixed_tau_for_search: bool = True,
    init: str = "best_pair",                 # "best_pair" | "random"
    candidate_subset: Optional[int] = 300,   # speed-up for large N
    rng_seed: int = 0,
    eps: float = 1e-12,
) -> List[int]:
    X = np.asarray(X, dtype=np.float32)
    demo_lengths_task = np.asarray(demo_lengths_task, dtype=np.float32)
    N, _ = X.shape

    if m <= 0:
        return []
    if N <= m:
        return list(range(N))
    if N == 1:
        return [0]
    m = int(m)

    rng = np.random.default_rng(rng_seed)

    Xn_all = l2_normalize(X, axis=1, eps=eps)
    fixed_tau = None
    if use_fixed_tau_for_search and intra_sim == "exp":
        fixed_tau = median_cosine_distance(Xn_all, eps=eps)
        fixed_tau = float(np.clip(fixed_tau, tau_floor, tau_ceiling))

    def L_of(S: List[int]) -> float:
        tl = subset_task_length_from_demo_lengths(demo_lengths_task, S)
        return compute_L_raw_for_subset(
            X, S,
            beta=beta,
            intra_sim=intra_sim,
            adaptive_tau=adaptive_tau,
            tau_intra=tau_intra,
            tau_floor=tau_floor,
            tau_ceiling=tau_ceiling,
            length_penalty=length_penalty,
            task_length=tl,
            fixed_tau=fixed_tau,  # fixed during greedy search if enabled
            eps=eps,
        )

    # ---- init ----
    if init == "random":
        S = [int(rng.integers(0, N))]
    elif init == "best_pair":
        best_val = -1.0
        best_pair = (0, 1)

        if N > 1200:
            K = 20000
            ii = rng.integers(0, N, size=K)
            jj = rng.integers(0, N, size=K)
            for a, b in zip(ii, jj):
                a, b = int(a), int(b)
                if a == b:
                    continue
                val = L_of([a, b])
                if val > best_val:
                    best_val = val
                    best_pair = (a, b)
        else:
            for i in range(N):
                for j in range(i + 1, N):
                    val = L_of([i, j])
                    if val > best_val:
                        best_val = val
                        best_pair = (i, j)

        S = [best_pair[0], best_pair[1]]
    else:
        raise ValueError(f"Unknown init={init}")

    S = list(dict.fromkeys(S))
    remaining = set(range(N)) - set(S)
    best_L = L_of(S)

    # ---- greedy add ----
    while len(S) < m and remaining:
        rem = np.array(list(remaining), dtype=np.int64)

        # candidate pruning by diversity from current center
        if candidate_subset is not None and rem.size > candidate_subset:
            center = Xn_all[S].mean(axis=0)
            center = center / (np.linalg.norm(center) + eps)
            sims = Xn_all[rem] @ center
            order = np.argsort(1.0 - sims)[::-1][:candidate_subset]
            cand = rem[order]
        else:
            cand = rem

        best_gain = -1e18
        best_i = int(cand[0])

        for i in cand:
            i = int(i)
            L_new = L_of(S + [i])
            gain = L_new - best_L
            if gain > best_gain:
                best_gain = gain
                best_i = i

        S.append(best_i)
        remaining.remove(best_i)
        best_L = L_of(S)

    return S


def best_select_task_subset_multistart(
    X: np.ndarray,
    demo_lengths_task: np.ndarray,
    m: int,
    *,
    best_restarts: int = 10,
    beta: float = 0.5,
    intra_sim: str = "exp",
    adaptive_tau: bool = True,
    tau_intra: float = 0.05,
    tau_floor: float = 0.03,
    tau_ceiling: float = 0.5,
    length_penalty: bool = True,
    use_fixed_tau_for_search: bool = True,
    candidate_subset: Optional[int] = 300,
    seed: int = 0,
    eps: float = 1e-12,
) -> Tuple[List[int], float]:
    """
    Multi-start greedy:
      - 1 run with best_pair init
      - best_restarts runs with random init
    Return subset with MAX L_raw (evaluated with fixed_tau=None).
    """
    def eval_truth(idx_local: List[int]) -> float:
        tl = subset_task_length_from_demo_lengths(demo_lengths_task, idx_local)
        return compute_L_raw_for_subset(
            X, idx_local,
            beta=beta, intra_sim=intra_sim,
            adaptive_tau=adaptive_tau, tau_intra=tau_intra,
            tau_floor=tau_floor, tau_ceiling=tau_ceiling,
            length_penalty=length_penalty,
            task_length=tl,
            fixed_tau=None,
            eps=eps,
        )

    best_idx = None
    best_L = -1.0

    # best_pair init
    idx0 = greedy_select_task_subset(
        X, demo_lengths_task, m,
        beta=beta, intra_sim=intra_sim,
        adaptive_tau=adaptive_tau, tau_intra=tau_intra,
        tau_floor=tau_floor, tau_ceiling=tau_ceiling,
        length_penalty=length_penalty,
        use_fixed_tau_for_search=use_fixed_tau_for_search,
        init="best_pair",
        candidate_subset=candidate_subset,
        rng_seed=seed,
        eps=eps,
    )
    L0 = float(eval_truth(idx0))
    best_idx, best_L = idx0, L0

    # random restarts
    for r in range(int(best_restarts)):
        idx = greedy_select_task_subset(
            X, demo_lengths_task, m,
            beta=beta, intra_sim=intra_sim,
            adaptive_tau=adaptive_tau, tau_intra=tau_intra,
            tau_floor=tau_floor, tau_ceiling=tau_ceiling,
            length_penalty=length_penalty,
            use_fixed_tau_for_search=use_fixed_tau_for_search,
            init="random",
            candidate_subset=candidate_subset,
            rng_seed=seed + 1000 + r,
            eps=eps,
        )
        L = float(eval_truth(idx))
        if L > best_L:
            best_L = L
            best_idx = idx

    return best_idx, best_L


# =========================================================
# 3) Random search for MAX/MIN L_raw
# =========================================================
def random_extreme_select_task_subset(
    X: np.ndarray,
    demo_lengths_task: np.ndarray,
    m: int,
    *,
    samples: int,
    mode: str,  # "max" | "min"
    patience: int = 0,  # early-stop: stop if no improvement for patience steps (0 disables)
    beta: float = 0.5,
    intra_sim: str = "exp",
    adaptive_tau: bool = True,
    tau_intra: float = 0.05,
    tau_floor: float = 0.03,
    tau_ceiling: float = 0.5,
    length_penalty: bool = True,
    seed: int = 0,
    eps: float = 1e-12,
) -> Tuple[List[int], float, int]:
    """
    Randomly sample many subsets of size m; return extreme (max/min) L_raw.
    Early-stop supported by patience.

    Returns:
      idx_local, best_L, num_evaluated
    """
    X = np.asarray(X, dtype=np.float32)
    demo_lengths_task = np.asarray(demo_lengths_task, dtype=np.float32)
    N = int(X.shape[0])
    m = int(max(2, min(m, N)))
    rng = np.random.default_rng(seed)

    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    if mode == "max":
        best_L = -float("inf")
        better = lambda L, best: L > best
    else:
        best_L = float("inf")
        better = lambda L, best: L < best

    best_idx = None
    since_improve = 0

    for t in range(int(samples)):
        idx = rng.choice(N, size=m, replace=False)
        tl = float(np.mean(demo_lengths_task[idx])) if idx.size > 0 else 0.0
        L = compute_L_raw_for_subset(
            X, idx,
            beta=beta, intra_sim=intra_sim,
            adaptive_tau=adaptive_tau, tau_intra=tau_intra,
            tau_floor=tau_floor, tau_ceiling=tau_ceiling,
            length_penalty=length_penalty,
            task_length=tl,
            fixed_tau=None,
            eps=eps,
        )

        if better(L, best_L):
            best_L = float(L)
            best_idx = idx
            since_improve = 0
        else:
            since_improve += 1

        if patience and since_improve >= int(patience):
            return best_idx.tolist(), float(best_L), (t + 1)

    return best_idx.tolist(), float(best_L), int(samples)


# =========================================================
# 4) Build filtered npz (global mask)
# =========================================================
def _build_task_global_index_map(task_ids: np.ndarray) -> Dict[int, np.ndarray]:
    task_ids = task_ids.astype(int)
    mp: Dict[int, np.ndarray] = {}
    for tid in np.unique(task_ids):
        tid = int(tid)
        mp[tid] = np.where(task_ids == tid)[0]
    return mp


def build_filtered_npz_arrays(npzdata, selected_local_by_task: Dict[int, List[int]]) -> Dict[str, np.ndarray]:
    task_ids = npzdata["task_ids"].astype(int)
    N_total = int(task_ids.shape[0])
    global_map = _build_task_global_index_map(task_ids)

    keep = np.zeros(N_total, dtype=bool)
    for tid, local_idx in selected_local_by_task.items():
        tid = int(tid)
        if tid not in global_map:
            continue
        gidx = global_map[tid]
        local_idx = np.asarray(local_idx, dtype=np.int64)
        if local_idx.size == 0:
            continue
        local_idx = np.clip(local_idx, 0, len(gidx) - 1)
        keep[gidx[local_idx]] = True

    out = {}
    for k in npzdata.files:
        arr = npzdata[k]
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] == N_total:
            out[k] = arr[keep]
        else:
            out[k] = arr

    out["selected_mask"] = keep
    out["selected_global_indices"] = np.where(keep)[0].astype(np.int64)
    return out


def update_task_lengths_in_filtered_npz(arrays_f: Dict[str, np.ndarray]) -> None:
    """Overwrite task_lengths so each task has mean(demo_lengths) within the filtered dataset."""
    if "task_lengths" not in arrays_f or "task_ids" not in arrays_f or "demo_lengths" not in arrays_f:
        return
    t_ids = np.asarray(arrays_f["task_ids"]).astype(int)
    d_lens = np.asarray(arrays_f["demo_lengths"]).astype(np.float32)
    t_lens = np.asarray(arrays_f["task_lengths"]).astype(np.float32)

    for tid in np.unique(t_ids):
        m = (t_ids == tid)
        new_tl = float(d_lens[m].mean()) if np.any(m) else 0.0
        t_lens[m] = new_tl

    arrays_f["task_lengths"] = t_lens


def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 5) CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser("Select LED-scored filtered subsets and export npz files plus a report.")

    p.add_argument("--in_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--ratios", type=str, default="0.8,0.6,0.4,0.2")
    p.add_argument("--min_keep", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    # ---- L_raw knobs (USED) ----
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--intra_sim", type=str, default="exp", choices=["exp", "linear"])
    p.add_argument("--adaptive_tau", action="store_true")
    p.add_argument("--no_adaptive_tau", action="store_true")
    p.add_argument("--tau_intra", type=float, default=0.05)
    p.add_argument("--tau_floor", type=float, default=0.03)
    p.add_argument("--tau_ceiling", type=float, default=0.5)
    p.add_argument("--length_penalty", action="store_true")
    p.add_argument("--no_length_penalty", action="store_true")

    # ---- greedy-best knobs ----
    p.add_argument("--use_fixed_tau_for_search", action="store_true")
    p.add_argument("--no_fixed_tau_for_search", action="store_true")
    p.add_argument("--candidate_subset", type=int, default=300, help="<=0 means no pruning")
    p.add_argument("--best_restarts", type=int, default=10)

    # ---- random-max knobs (NEW) ----
    p.add_argument("--random_max_samples", type=int, default=200000,
                   help="How many random subsets per task for MAX search")
    p.add_argument("--random_max_patience", type=int, default=30000,
                   help="Early stop if no improvement for this many tries (0 disables)")

    # ---- optional random-min comparison knobs ----
    p.add_argument("--save_randommin", action="store_true",
                   help="Also save random-min comparison subsets. Not needed for the standard LED release workflow.")
    p.add_argument("--random_min_samples", type=int, default=5000)

    # ---- debugging ----
    p.add_argument("--debug", action="store_true")

    # ---- Extra params (NOT used in L_raw-only, but recorded for consistency) ----
    p.add_argument("--task_knn", type=int, default=7)
    p.add_argument("--task_temp", type=float, default=0.07)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--pi_scale", type=float, default=0.02)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ratios = tuple(float(x) for x in args.ratios.split(",") if x.strip() != "")
    if not ratios:
        raise ValueError("Empty --ratios")

    adaptive_tau = True
    if args.no_adaptive_tau:
        adaptive_tau = False
    elif args.adaptive_tau:
        adaptive_tau = True

    length_penalty = True
    if args.no_length_penalty:
        length_penalty = False
    elif args.length_penalty:
        length_penalty = True

    use_fixed_tau_for_search = True
    if args.no_fixed_tau_for_search:
        use_fixed_tau_for_search = False
    elif args.use_fixed_tau_for_search:
        use_fixed_tau_for_search = True

    candidate_subset: Optional[int] = args.candidate_subset
    if candidate_subset is not None and candidate_subset <= 0:
        candidate_subset = None

    npzdata = np.load(args.in_npz, allow_pickle=True)

    # Optional unique ids
    has_ep_idx = "episode_indices" in npzdata.files
    has_uid = "demo_uids" in npzdata.files
    episode_indices_all = npzdata["episode_indices"].astype(int) if has_ep_idx else None
    demo_uids_all = npzdata["demo_uids"] if has_uid else None

    task_groups = group_by_task(npzdata)
    task_ids = npzdata["task_ids"].astype(int)
    global_map = _build_task_global_index_map(task_ids)

    print(
        "[INFO] L_raw-only selection. Recorded but NOT used here: "
        f"task_knn={args.task_knn}, task_temp={args.task_temp}, alpha={args.alpha}, pi_scale={args.pi_scale}"
    )

    report: Dict[str, Any] = {
        "input_npz": os.path.abspath(args.in_npz),
        "out_dir": os.path.abspath(args.out_dir),
        "ratios": [float(r) for r in ratios],
        "meta": {
            # used
            "beta": float(args.beta),
            "intra_sim": args.intra_sim,
            "adaptive_tau": bool(adaptive_tau),
            "tau_intra": float(args.tau_intra),
            "tau_floor": float(args.tau_floor),
            "tau_ceiling": float(args.tau_ceiling),
            "length_penalty": bool(length_penalty),
            "task_length_definition": "mean(demo_lengths of selected subset)",
            # greedy-best
            "use_fixed_tau_for_search": bool(use_fixed_tau_for_search),
            "candidate_subset": candidate_subset,
            "best_restarts": int(args.best_restarts),
            # random-max
            "random_max_samples": int(args.random_max_samples),
            "random_max_patience": int(args.random_max_patience),
            # optional random-min comparison
            "save_randommin": bool(args.save_randommin),
            "random_min_samples": int(args.random_min_samples),
            "seed": int(args.seed),
            # recorded (unused)
            "task_knn": int(args.task_knn),
            "task_temp": float(args.task_temp),
            "alpha": float(args.alpha),
            "pi_scale": float(args.pi_scale),
        },
        "per_ratio": {}
    }

    for r in ratios:
        r_key = f"{r:.4g}"

        best_local_by_task: Dict[int, List[int]] = {}
        bad_local_by_task: Dict[int, List[int]] = {}
        per_task: Dict[str, Any] = {}

        for g in task_groups:
            tid = int(g["task_id"])
            X = np.asarray(g["features"], dtype=np.float32)
            demo_lens_task = np.asarray(g["demo_lengths"], dtype=np.float32)

            N = int(X.shape[0])
            m = int(np.ceil(float(r) * N))
            m = max(int(args.min_keep), min(m, N))

            # ---------- candidate BEST via greedy multi-start ----------
            greedy_idx_local, greedy_L = best_select_task_subset_multistart(
                X, demo_lens_task, m,
                best_restarts=args.best_restarts,
                beta=args.beta, intra_sim=args.intra_sim,
                adaptive_tau=adaptive_tau, tau_intra=args.tau_intra,
                tau_floor=args.tau_floor, tau_ceiling=args.tau_ceiling,
                length_penalty=length_penalty,
                use_fixed_tau_for_search=use_fixed_tau_for_search,
                candidate_subset=candidate_subset,
                seed=args.seed + tid * 17,
            )

            # ---------- candidate BEST via random-max search ----------
            randmax_idx_local, randmax_L, randmax_evals = random_extreme_select_task_subset(
                X, demo_lens_task, m,
                samples=args.random_max_samples,
                mode="max",
                patience=max(int(args.random_max_patience), 0),
                beta=args.beta, intra_sim=args.intra_sim,
                adaptive_tau=adaptive_tau, tau_intra=args.tau_intra,
                tau_floor=args.tau_floor, tau_ceiling=args.tau_ceiling,
                length_penalty=length_penalty,
                seed=args.seed + 123456 + tid * 101,
            )

            # ---------- choose BEST = max(greedy, randmax) ----------
            if randmax_L >= greedy_L:
                best_src = "random_max"
                best_idx_local = randmax_idx_local
                best_L = randmax_L
            else:
                best_src = "greedy_multistart"
                best_idx_local = greedy_idx_local
                best_L = greedy_L

            best_local_by_task[tid] = best_idx_local
            best_tl = subset_task_length_from_demo_lengths(demo_lens_task, best_idx_local)

            # ---------- local -> global ----------
            gidx_all = global_map[tid]
            best_idx_global = gidx_all[np.asarray(best_idx_local, dtype=np.int64)].astype(int)

            # ---------- map to episode_index / demo_uid ----------
            best_ep = None
            best_uid = None
            if episode_indices_all is not None:
                best_ep = episode_indices_all[best_idx_global].astype(int).tolist()
            if demo_uids_all is not None:
                best_uid = [str(x) for x in demo_uids_all[best_idx_global].tolist()]

            per_task[str(tid)] = {
                "N": N,
                "m": m,
                "task_description": str(g.get("task_description", "")),
                "best": {
                    "source": best_src,
                    "task_length_selected": float(best_tl),
                    "L_sel": float(best_L),
                    "selected_local_indices": [int(x) for x in best_idx_local],
                    "selected_global_indices": best_idx_global.astype(int).tolist(),
                    "selected_episode_indices": best_ep,
                    "selected_demo_uids": best_uid,

                    "greedy_multistart_L": float(greedy_L),
                    "greedy_multistart_selected_local_indices": [int(x) for x in greedy_idx_local],

                    "random_max_L": float(randmax_L),
                    "random_max_evaluated": int(randmax_evals),
                    "random_max_selected_local_indices": [int(x) for x in randmax_idx_local],
                },
            }

            if args.save_randommin:
                bad_idx_local, bad_L, bad_evals = random_extreme_select_task_subset(
                    X, demo_lens_task, m,
                    samples=args.random_min_samples,
                    mode="min",
                    patience=0,
                    beta=args.beta, intra_sim=args.intra_sim,
                    adaptive_tau=adaptive_tau, tau_intra=args.tau_intra,
                    tau_floor=args.tau_floor, tau_ceiling=args.tau_ceiling,
                    length_penalty=length_penalty,
                    seed=args.seed + 999 + tid * 31,
                )
                bad_local_by_task[tid] = bad_idx_local
                bad_tl = subset_task_length_from_demo_lengths(demo_lens_task, bad_idx_local)
                bad_idx_global = gidx_all[np.asarray(bad_idx_local, dtype=np.int64)].astype(int)
                bad_ep = episode_indices_all[bad_idx_global].astype(int).tolist() if episode_indices_all is not None else None
                bad_uid = [str(x) for x in demo_uids_all[bad_idx_global].tolist()] if demo_uids_all is not None else None

                per_task[str(tid)]["randommin"] = {
                    "task_length_selected": float(bad_tl),
                    "L_sel": float(bad_L),
                    "selected_local_indices": [int(x) for x in bad_idx_local],
                    "selected_global_indices": bad_idx_global.astype(int).tolist(),
                    "selected_episode_indices": bad_ep,
                    "selected_demo_uids": bad_uid,
                    "random_min_evaluated": int(bad_evals),
                }
                per_task[str(tid)]["gap_best_minus_randommin"] = float(best_L - bad_L)

            if args.debug:
                msg = (
                    f"[Task {tid}] N={N} m={m} "
                    f"greedyL={greedy_L:.6f} randmaxL={randmax_L:.6f} -> best({best_src})={best_L:.6f}"
                )
                if args.save_randommin:
                    msg += f" badL={bad_L:.6f} gap={best_L - bad_L:.6f}"
                print(msg)

        # ---- save BEST filtered npz ----
        arrays_best = build_filtered_npz_arrays(npzdata, best_local_by_task)
        update_task_lengths_in_filtered_npz(arrays_best)
        out_best = os.path.join(args.out_dir, f"filtered_r{r_key}.npz")
        save_npz(out_best, arrays_best)

        per_task["_filtered_npz_best"] = out_best
        per_task["_kept_best_total"] = int(arrays_best["selected_global_indices"].shape[0])

        if args.save_randommin:
            arrays_bad = build_filtered_npz_arrays(npzdata, bad_local_by_task)
            update_task_lengths_in_filtered_npz(arrays_bad)
            out_bad = os.path.join(args.out_dir, f"randommin_r{r_key}.npz")
            save_npz(out_bad, arrays_bad)
            per_task["_filtered_npz_randommin"] = out_bad
            per_task["_kept_randommin_total"] = int(arrays_bad["selected_global_indices"].shape[0])

        report["per_ratio"][r_key] = per_task

        print(f"[OK] ratio={r_key} BEST saved: {out_best} kept={per_task['_kept_best_total']}")
        if args.save_randommin:
            print(f"[OK] ratio={r_key} random-min saved: {out_bad} kept={per_task['_kept_randommin_total']}")

    out_report = os.path.join(args.out_dir, "selection_report.json")
    save_json(out_report, report)
    print(f"[OK] saved report: {out_report}")

    if not has_ep_idx or not has_uid:
        print("[WARN] npz missing episode_indices/demo_uids. Report will only contain indices.")


if __name__ == "__main__":
    main()
