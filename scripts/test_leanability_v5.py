#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt

from dataeval.metric.leanability import compute_leanability_from_npzdata


# =========================
# 0) Correlation helper
# =========================
def _corr(pred, gt):
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Length mismatch: pred={len(pred)} gt={len(gt)}")
    srcc, _ = spearmanr(pred, gt)
    plcc, _ = pearsonr(pred, gt)
    krcc, _ = kendalltau(pred, gt)
    return float(srcc), float(plcc), float(krcc)


# =========================
# 0.5) Plot helper
# =========================
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau


def bootstrap_fit_ci(x, y, degree=3, n_boot=1000):

    x = np.asarray(x)
    y = np.asarray(y)

    x_grid = np.linspace(x.min(), x.max(), 200)

    boot_preds = []

    n = len(x)

    for _ in range(n_boot):

        idx = np.random.choice(n, n, replace=True)

        xb = x[idx]
        yb = y[idx]

        coefs = np.polyfit(xb, yb, degree)
        poly = np.poly1d(coefs)

        boot_preds.append(poly(x_grid))

    boot_preds = np.array(boot_preds)

    mean = np.mean(boot_preds, axis=0)
    lower = np.percentile(boot_preds, 2.5, axis=0)
    upper = np.percentile(boot_preds, 97.5, axis=0)

    return x_grid, mean, lower, upper


def plot_fit_vs_gt_multi_from_bench(
    y_trues,
    y_preds,
    labels,
    out_png,
    title="",
    fontsize=18,
    fit_degree=3,
    y_scale_range=(20, 100),
):

    markers = ['o','s','D','^','v','*','x','+','p','H','8'] * 5

    base_colors = plt.get_cmap('tab10').colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    all_gt = np.concatenate(y_trues)
    all_pred = np.concatenate(y_preds)

    # scale prediction
    y_lo, y_hi = y_scale_range
    pmin, pmax = all_pred.min(), all_pred.max()

    all_pred_scaled = y_lo + (all_pred - pmin)/(pmax - pmin)*(y_hi-y_lo)

    # correlations
    srcc,_ = spearmanr(all_gt, all_pred_scaled)
    krcc,_ = kendalltau(all_gt, all_pred_scaled)
    plcc,_ = pearsonr(all_gt, all_pred_scaled)

    # bootstrap CI
    x_fit, y_fit, ci_lo, ci_hi = bootstrap_fit_ci(
        all_gt,
        all_pred_scaled,
        degree=fit_degree,
        n_boot=1000
    )

    plt.figure(figsize=(11,7))

    start = 0

    for i, gt_i in enumerate(y_trues):

        gt_i = np.asarray(gt_i)
        n = len(gt_i)

        pred_scaled_i = all_pred_scaled[start:start+n]

        plt.scatter(
            gt_i,
            pred_scaled_i,
            label=labels[i],
            marker=markers[i],
            color=colors[i],
            s=80,
            alpha=0.85,
        )

        start += n

    # CI shadow
    plt.fill_between(
        x_fit,
        ci_lo,
        ci_hi,
        color="red",
        alpha=0.18
    )

    # fit line
    plt.plot(
        x_fit,
        y_fit,
        color="red",
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Ground Truth", fontsize=fontsize)
    plt.ylabel("Predicted Score", fontsize=fontsize)

    plt.title(title, fontsize=fontsize+2)

    plt.grid(True, linestyle="--", alpha=0.35)

    # correlation text
    text = (
        f"SRCC = {srcc:.3f}\n"
        f"KRCC = {krcc:.3f}\n"
        f"PLCC = {plcc:.3f}"
    )

    plt.text(
        0.03,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=fontsize-2,
        verticalalignment='top',
        bbox=dict(
            facecolor="white",
            alpha=0.7,
            edgecolor="none"
        )
    )

    plt.legend(
        fontsize=fontsize-4,
        loc="lower right",
        bbox_to_anchor=(1.35,0)
    )

    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)

# def plot_fit_vs_gt_multi_from_bench(
#     y_trues, y_preds, labels,
#     out_png,
#     title="Learning Ease vs GT (ALL Points)",
#     fontsize=18,
#     fit_degree=3,
#     y_scale_range=(20, 100),
# ):
#     markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'H', '8', '<', '>', '|', '_', '.', ','] * 6
#     base_colors = plt.get_cmap('tab10').colors
#     colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

#     all_gt = np.concatenate([np.asarray(a, dtype=float) for a in y_trues], axis=0)
#     all_pred = np.concatenate([np.asarray(a, dtype=float) for a in y_preds], axis=0)

#     y_lo, y_hi = y_scale_range
#     pmin, pmax = float(all_pred.min()), float(all_pred.max())
#     if abs(pmax - pmin) < 1e-12:
#         all_pred_scaled = np.full_like(all_pred, (y_lo + y_hi) / 2.0, dtype=float)
#     else:
#         all_pred_scaled = y_lo + (all_pred - pmin) / (pmax - pmin) * (y_hi - y_lo)

#     deg = int(fit_degree)
#     coefs = np.polyfit(all_gt, all_pred_scaled, deg)
#     poly = np.poly1d(coefs)
#     x_fit = np.linspace(float(all_gt.min()), float(all_gt.max()), 200)
#     y_fit = poly(x_fit)

#     plt.figure(figsize=(11, 5))
#     start = 0
#     for i, gt_i in enumerate(y_trues):
#         gt_i = np.asarray(gt_i, dtype=float)
#         n = len(gt_i)
#         pred_scaled_i = all_pred_scaled[start:start + n]

#         plt.scatter(
#             gt_i, pred_scaled_i,
#             label=labels[i],
#             marker=markers[i],
#             color=colors[i],
#             s=55,
#             alpha=0.75,
#             edgecolor='None',
#         )
#         start += n

#     plt.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label=f'Poly Fit (deg={deg})')

#     srcc, plcc, krcc = _corr(all_pred, all_gt)
#     plt.text(
#         0.02, 0.98,
#         f"SRCC={srcc:.3f}  KRCC={krcc:.3f}  PLCC={plcc:.3f}\nN={len(all_gt)}",
#         transform=plt.gca().transAxes,
#         va='top',
#         fontsize=fontsize - 2,
#         bbox=dict(facecolor='white', alpha=0.65, edgecolor='gray'),
#     )

#     plt.xlabel("Ground Truth", fontsize=fontsize)
#     plt.ylabel("Predicted (scaled)", fontsize=fontsize)
#     plt.title(title, fontsize=fontsize + 2)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=fontsize - 4, loc='lower right', bbox_to_anchor=(1.45, 0.0))

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#     plt.savefig(out_png, dpi=300)
#     plt.close()
#     return dict(srcc=srcc, plcc=plcc, krcc=krcc, fit_coefs=coefs)


# =========================
# 1) GT + Task order (use yours)
# =========================
GTS = {
    "goal":    np.array([40, 52, 90, 82, 96, 96, 78, 40, 88, 94], dtype=float),
    "object":  np.array([88, 70, 90, 84, 72, 86, 88, 94, 90, 86], dtype=float),
    "spatial": np.array([94, 96, 76, 96, 82, 84, 88, 90, 80, 72], dtype=float),
    "ten":     np.array([52, 34, 48, 20, 62, 58, 66, 44, 38, 68], dtype=float),

    # 20D combos
    "ten-object":     np.array([54, 50, 24, 14, 48, 48, 74, 36, 30, 64, 84, 66, 94, 88, 76, 76, 92, 88, 76, 92], dtype=float),
    "goal-object":    np.array([64, 48, 88, 86, 94, 94, 74, 50, 70, 96, 90, 80, 90, 86, 82, 92, 96, 92, 92, 92], dtype=float),
    "spatial-10":     np.array([94, 92, 82, 92, 84, 90, 90, 100, 72, 76, 50, 42, 38, 32, 54, 44, 66, 44, 60, 72], dtype=float),
    "spatial-goal":   np.array([90, 90, 72, 96, 84, 86, 86, 94, 84, 66, 60, 44, 88, 90, 90, 94, 72, 42, 90, 94], dtype=float),
    "spatial-object": np.array([94, 76, 68, 94, 80, 86, 94, 72, 88, 68, 88, 72, 96, 88, 78, 84, 92, 96, 94, 92], dtype=float),
    "goal+ten":       np.array([58, 52, 82, 84, 94, 98, 64, 38, 92, 98, 52, 52, 58, 28, 70, 54, 78, 44, 56, 64], dtype=float),
}

BENCH_ALIAS = {"10": "ten", "Goal": "goal", "Spatial": "spatial"}

TASK_ORDER_TEXT = {
    "goal": [
        "open the middle drawer of the cabinet demo",
        "open the top drawer and put the bowl inside demo",
        "push the plate to the front of the stove demo",
        "put the bowl on the plate demo",
        "put the bowl on the stove demo",
        "put the bowl on top of the cabinet demo",
        "put the cream cheese in the bowl demo",
        "put the wine bottle on the rack demo",
        "put the wine bottle on top of the cabinet demo",
        "turn on the stove demo",
    ],
    "object": [
        "pick up the alphabet soup and place it in the basket demo",
        "pick up the bbq sauce and place it in the basket demo",
        "pick up the butter and place it in the basket demo",
        "pick up the chocolate pudding and place it in the basket demo",
        "pick up the cream cheese and place it in the basket demo",
        "pick up the ketchup and place it in the basket demo",
        "pick up the milk and place it in the basket demo",
        "pick up the orange juice and place it in the basket demo",
        "pick up the salad dressing and place it in the basket demo",
        "pick up the tomato sauce and place it in the basket demo",
    ],
    "spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate demo",
        "pick up the black bowl from table center and place it on the plate demo",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate demo",
        "pick up the black bowl next to the cookie box and place it on the plate demo",
        "pick up the black bowl next to the plate and place it on the plate demo",
        "pick up the black bowl next to the ramekin and place it on the plate demo",
        "pick up the black bowl on the cookie box and place it on the plate demo",
        "pick up the black bowl on the ramekin and place it on the plate demo",
        "pick up the black bowl on the stove and place it on the plate demo",
        "pick up the black bowl on the wooden cabinet and place it on the plate demo",
    ],
    "ten": [
        "KITCHEN SCENE3 turn on the stove and put the moka pot on it demo",
        "KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it demo",
        "KITCHEN SCENE6 put the yellow and white mug in the microwave and close it demo",
        "KITCHEN SCENE8 put both moka pots on the stove demo",
        "LIVING ROOM SCENE1 put both the alphabet soup and the cream cheese box in the basket demo",
        "LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket demo",
        "LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket demo",
        "LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate demo",
        "LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate demo",
        "STUDY SCENE1 pick up the book and place it in the back compartment of the caddy demo",
    ],
}


# =========================
# 2) Text normalize + fuzzy scoring
# =========================
def _norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" demo", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _fuzzy_score(gold_norm: str, cand_norm: str) -> float:
    gtok = set(gold_norm.split())
    ctok = set(cand_norm.split())
    if not gtok and not ctok:
        return 0.0
    inter = len(gtok & ctok)
    union = len(gtok | ctok) + 1e-9
    jacc = inter / union
    sub = 1.0 if gold_norm in cand_norm else 0.0
    len_bonus = min(len(cand_norm), 200) / 200.0
    return 2.0 * jacc + 0.8 * sub + 0.2 * len_bonus


# =========================
# 3) Align 10 tasks with bench-restricted candidate set
# =========================
def _load_obj_dict(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, np.ndarray) and x.shape == ():
        try:
            v = x.item()
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    try:
        v = x.item()
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _align_1x10(per_task: dict, bench: str, *, task_desc_by_id=None, allowed_task_ids=None, verbose=False) -> np.ndarray:
    bench = bench.lower()
    gold_order = TASK_ORDER_TEXT[bench]

    if task_desc_by_id is None:
        raise ValueError(f"[{bench}] task_descriptions_by_id is required for fuzzy align.")
    if isinstance(task_desc_by_id, np.ndarray) and task_desc_by_id.shape == ():
        task_desc_by_id = task_desc_by_id.item()

    if allowed_task_ids is None:
        allowed = set(int(k) for k in per_task.keys())
    else:
        allowed = set(int(x) for x in allowed_task_ids)
        allowed &= set(int(k) for k in per_task.keys())

    tid2nd = {}
    for tid, desc in task_desc_by_id.items():
        tid = int(tid)
        if tid in allowed:
            tid2nd[tid] = _norm_text(desc)

    if len(tid2nd) == 0:
        raise ValueError(f"[{bench}] allowed task id set empty after intersection; check combo meta.")

    nd2tids = {}
    for tid, nd in tid2nd.items():
        nd2tids.setdefault(nd, []).append(tid)

    chosen_ids = []
    for gold in gold_order:
        g = _norm_text(gold)

        exact = nd2tids.get(g, [])
        if len(exact) == 1:
            chosen_ids.append(exact[0])
            continue

        best_tid = None
        best_score = -1e9
        for tid, nd in tid2nd.items():
            sc = _fuzzy_score(g, nd)
            if sc > best_score + 1e-12:
                best_score = sc
                best_tid = tid
            elif abs(sc - best_score) <= 1e-12:
                if best_tid is None or tid < best_tid:
                    best_tid = tid

        if best_tid is None:
            raise ValueError(f"[{bench}] cannot match gold='{gold}' in allowed pool.")
        chosen_ids.append(best_tid)

    if verbose:
        print(f"[ALIGN {bench}] chosen task_ids (gold order): {chosen_ids}")

    pred = []
    for tid in chosen_ids:
        if tid in per_task:
            pred.append(per_task[tid])
        elif str(tid) in per_task:
            pred.append(per_task[str(tid)])
        else:
            raise KeyError(f"[{bench}] chosen tid {tid} missing in per_task.")
    return np.asarray(pred, dtype=float)


def compute_correlation(result, benchmark_name, *, combo_meta=None, verbose=False):
    bench = BENCH_ALIAS.get(benchmark_name, benchmark_name).lower()
    gt = GTS.get(bench)
    if gt is None:
        raise ValueError(f"Benchmark '{benchmark_name}' mapped to '{bench}' not found in GTS.")

    per_task = result["leanability_per_task"]
    tdesc = result.get("task_descriptions_by_id", None)

    if bench in ("goal", "object", "spatial", "ten"):
        pred = _align_1x10(per_task, bench, task_desc_by_id=tdesc, allowed_task_ids=None, verbose=verbose)
        return pred, _corr(pred, gt)

    if "-" in bench or "+" in bench:
        if combo_meta is None:
            raise ValueError(f"[{bench}] combo_meta is required for combo alignment.")

        if "-" in bench:
            left, right = bench.split("-", 1)
        else:
            left, right = bench.split("+", 1)

        left = BENCH_ALIAS.get(left, left).lower()
        right = BENCH_ALIAS.get(right, right).lower()

        left_ids = combo_meta["left_task_ids"] if combo_meta["left_name"] == left else combo_meta["right_task_ids"]
        right_ids = combo_meta["right_task_ids"] if combo_meta["right_name"] == right else combo_meta["left_task_ids"]

        pred_left = _align_1x10(per_task, left, task_desc_by_id=tdesc, allowed_task_ids=left_ids, verbose=verbose)
        pred_right = _align_1x10(per_task, right, task_desc_by_id=tdesc, allowed_task_ids=right_ids, verbose=verbose)
        pred = np.concatenate([pred_left, pred_right], axis=0)
        return pred, _corr(pred, gt)

    raise ValueError(f"Unsupported benchmark name: {benchmark_name} (mapped to {bench})")


# =========================
# 4) NPZ merge with COMBO META
# =========================
def _npz_to_dict(npzfile) -> dict:
    return {k: npzfile[k] for k in npzfile.files}


def _get_task_id_key(d: dict) -> str:
    for k in ["task_id", "task_ids", "taskid", "task_index", "task_indices"]:
        if k in d:
            return k
    raise KeyError(f"Cannot find task id key in npz. keys={list(d.keys())}")


def remap_task_ids_in_npz_dict(npz_dict: dict, offset: int):
    tid_key = _get_task_id_key(npz_dict)
    npz_dict[tid_key] = npz_dict[tid_key].astype(np.int64) + int(offset)

    if "task_descriptions_by_id" in npz_dict:
        desc_map = _load_obj_dict(npz_dict["task_descriptions_by_id"])
        if isinstance(desc_map, dict):
            new_map = {int(k) + int(offset): v for k, v in desc_map.items()}
            npz_dict["task_descriptions_by_id"] = np.array(new_map, dtype=object)

    return tid_key


def _should_concat_by_sample_dim(va, vb, Na, Nb):
    if not (isinstance(va, np.ndarray) and isinstance(vb, np.ndarray)):
        return False
    if va.ndim < 1 or vb.ndim < 1:
        return False
    if va.shape[0] != Na or vb.shape[0] != Nb:
        return False
    if va.ndim != vb.ndim:
        return False
    if va.shape[1:] != vb.shape[1:]:
        return False
    return True


def merge_two_npz_for_combo(npz_path_a: str, npz_path_b: str, *, save_path: str, left_name: str, right_name: str):
    a = _npz_to_dict(np.load(npz_path_a, allow_pickle=True))
    b = _npz_to_dict(np.load(npz_path_b, allow_pickle=True))

    tid_key_a = _get_task_id_key(a)
    tid_key_b = _get_task_id_key(b)
    Na = int(a[tid_key_a].shape[0])
    Nb = int(b[tid_key_b].shape[0])

    left_task_ids = sorted(set(a[tid_key_a].astype(np.int64).tolist()))
    right_task_ids_raw = sorted(set(b[tid_key_b].astype(np.int64).tolist()))

    max_a = int(np.max(a[tid_key_a].astype(np.int64)))
    offset = max_a + 1
    remap_task_ids_in_npz_dict(b, offset=offset)
    right_task_ids = [int(x) + int(offset) for x in right_task_ids_raw]

    merged = {}
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    for k in all_keys:
        in_a, in_b = (k in a), (k in b)
        if in_a and in_b:
            va, vb = a[k], b[k]

            if k == "task_descriptions_by_id":
                da = _load_obj_dict(va)
                db = _load_obj_dict(vb)
                if isinstance(da, dict) and isinstance(db, dict):
                    out = dict(da)
                    out.update(db)
                    merged[k] = np.array(out, dtype=object)
                else:
                    merged[k] = va
                continue

            if _should_concat_by_sample_dim(va, vb, Na, Nb):
                merged[k] = np.concatenate([va, vb], axis=0)
            else:
                merged[k] = va
        elif in_a:
            merged[k] = a[k]
        else:
            merged[k] = b[k]

    combo_meta = {
        "left_name": left_name.lower(),
        "right_name": right_name.lower(),
        "left_task_ids": left_task_ids,
        "right_task_ids": right_task_ids,
        "offset": int(offset),
    }
    merged["_combo_meta"] = np.array(combo_meta, dtype=object)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, **merged)
    return save_path, combo_meta


# =========================
# 5) Leanability wrapper (takes **lean_kwargs)
# =========================
def compute_metrics_for_npz(npz_path, benchmark_name, *, lean_kwargs=None, verbose=False):
    npzdata = np.load(npz_path, allow_pickle=True)

    combo_meta = None
    if "_combo_meta" in npzdata.files:
        combo_meta = _load_obj_dict(npzdata["_combo_meta"])

    lean_kwargs = dict(lean_kwargs or {})

    # ---- Backward-compat: drop unsupported kwargs if needed ----
    try:
        result = compute_leanability_from_npzdata(npzdata=npzdata, **lean_kwargs)
    except TypeError as e:
        msg = str(e)
        bad_keys = []
        for k in list(lean_kwargs.keys()):
            if k in msg:
                bad_keys.append(k)
        if not bad_keys:
            for k in ["task_knn", "task_temp"]:
                if k in lean_kwargs:
                    bad_keys.append(k)

        if bad_keys:
            for k in bad_keys:
                lean_kwargs.pop(k, None)
            print(f"[WARN] compute_leanability_from_npzdata does not accept {bad_keys}, dropped for compat.")
            result = compute_leanability_from_npzdata(npzdata=npzdata, **lean_kwargs)
        else:
            raise

    pred, (srcc, plcc, krcc) = compute_correlation(
        result, benchmark_name,
        combo_meta=combo_meta,
        verbose=verbose
    )
    return pred, (srcc, plcc, krcc)


# =========================
# 6) CLI
# =========================
def _parse_skip_list(s: str):
    if s is None:
        return set()
    s = s.strip()
    if not s:
        return set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = set()
    for p in parts:
        out.add(BENCH_ALIAS.get(p, p).lower())
    return out


def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate Leanability against GT for LIBERO benchmarks (+ combos).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # paths (make all optional; we enforce required-by-selection later)
    p.add_argument("--npz_goal", type=str, default=None)
    p.add_argument("--npz_object", type=str, default=None)
    p.add_argument("--npz_spatial", type=str, default=None)
    p.add_argument("--npz_ten", type=str, default=None)

    # NEW: general skip list
    p.add_argument(
        "--skip_benches",
        type=str,
        default="",
        help="Comma-separated benches to skip. Examples: 'ten' or 'goal,ten'. Aliases supported: 10->ten.",
    )

    # output
    p.add_argument("--out_dir", type=str, default="./plots")
    p.add_argument("--tmp_dir", type=str, default="./tmp")
    p.add_argument("--plot", action="store_true", help="Save one ALL-points plot.")
    p.add_argument("--plot_name", type=str, default="learning_ease_fit_ALL_points.png")
    p.add_argument("--fit_degree", type=int, default=3)
    p.add_argument("--y_lo", type=float, default=20.0)
    p.add_argument("--y_hi", type=float, default=100.0)

    # debug
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verbose_align", action="store_true")

    # ---- leanability params ----
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--sigma_task", type=float, default=1e-7)
    p.add_argument("--sigma_center", type=float, default=1e-6)
    p.add_argument("--pi_scale", type=float, default=0.02)

    p.add_argument("--transfer_mode", type=str, default="harmonic",
                   choices=["semantic", "visual_center", "harmonic"])

    p.add_argument("--intra_sim", type=str, default="exp", choices=["exp", "linear"])
    p.add_argument("--tau_intra", type=float, default=0.05)
    p.add_argument("--adaptive_tau", action="store_true")
    p.add_argument("--no_adaptive_tau", dest="adaptive_tau", action="store_false")
    p.set_defaults(adaptive_tau=True)

    p.add_argument("--tau_floor", type=float, default=0.03)
    p.add_argument("--tau_ceiling", type=float, default=0.5)

    p.add_argument("--length_penalty", action="store_true")
    p.add_argument("--no_length_penalty", dest="length_penalty", action="store_false")
    p.set_defaults(length_penalty=True)

    p.add_argument("--use_self_loop", action="store_true")
    p.add_argument("--no_use_self_loop", dest="use_self_loop", action="store_false")
    p.set_defaults(use_self_loop=True)

    p.add_argument("--alpha", type=float, default=0.35)

    p.add_argument("--task_knn", type=int, default=5,
                   help="Keep top-k neighbors per task row for task transfer graph.")
    p.add_argument("--task_temp", type=float, default=0.07,
                   help="Softmax temperature for task transfer graph (smaller -> sharper).")

    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--debug", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()

    # canonical bench set
    ALL_BENCHES = ["goal", "object", "spatial", "ten"]

    paths = {
        "goal": args.npz_goal,
        "object": args.npz_object,
        "spatial": args.npz_spatial,
        "ten": args.npz_ten,
    }

    skip = _parse_skip_list(args.skip_benches)
    unknown = sorted([b for b in skip if b not in ALL_BENCHES])
    if unknown:
        raise ValueError(f"Unknown benches in --skip_benches: {unknown}. Allowed: {ALL_BENCHES} (aliases: 10->ten).")

    benches = [b for b in ALL_BENCHES if b not in skip]
    if not benches:
        raise ValueError("After applying --skip_benches, no benchmarks left to evaluate.")

    lean_kwargs = dict(
        beta=args.beta,
        sigma_task=args.sigma_task,
        sigma_center=args.sigma_center,
        pi_scale=args.pi_scale,

        transfer_mode=args.transfer_mode,

        intra_sim=args.intra_sim,
        tau_intra=args.tau_intra,
        adaptive_tau=args.adaptive_tau,
        tau_floor=args.tau_floor,
        tau_ceiling=args.tau_ceiling,
        length_penalty=args.length_penalty,

        use_self_loop=args.use_self_loop,
        alpha=args.alpha,

        task_knn=args.task_knn,
        task_temp=args.task_temp,

        eps=args.eps,
        debug=args.debug,
    )

    print("==== Leanability kwargs ====")
    for k in sorted(lean_kwargs.keys()):
        print(f"{k:16s}: {lean_kwargs[k]}")
    print(f"==== Selected benches ====\n{benches}")
    if skip:
        print(f"==== Skipped benches ====\n{sorted(skip)}")

    if args.dry_run:
        return

    # full combo list; we will auto-drop those that involve skipped benches
    combos_all = [
        ("goal", "object", "goal-object"),
        ("spatial", "goal", "spatial-goal"),
        ("spatial", "ten", "spatial-10"),
        ("ten", "object", "ten-object"),
        ("spatial", "object", "spatial-object"),
        ("goal", "ten", "goal+ten"),
    ]
    combos = [c for c in combos_all if (c[0] not in skip and c[1] not in skip)]

    # safety: required npz exist for selected benches
    for b in benches:
        if paths.get(b) is None:
            raise ValueError(f"Missing --npz_{b} for selected bench '{b}'. Provide it or add '{b}' to --skip_benches.")
        if not os.path.exists(paths[b]):
            raise FileNotFoundError(f"npz for '{b}' not found: {paths[b]}")

    # collect ALL points
    y_trues_all, y_preds_all, labels_all = [], [], []

    print("\n===== Per-benchmark Metrics (10D each) =====")
    for bench in benches:
        pred, (srcc, plcc, krcc) = compute_metrics_for_npz(
            paths[bench],
            benchmark_name=bench,
            lean_kwargs=lean_kwargs,
            verbose=args.verbose_align,
        )
        gt = GTS[bench]
        print(f"{bench:8s}  SRCC={srcc:.6f}  PLCC={plcc:.6f}  KRCC={krcc:.6f}")
        y_trues_all.append(gt)
        y_preds_all.append(pred)
        labels_all.append(bench)

    print("\n===== 20D Combos (merge npz then compute once) =====")
    os.makedirs(args.tmp_dir, exist_ok=True)

    for left, right, combo_name in combos:
        combo_key = combo_name.lower()
        gt = GTS[combo_key]

        merged_path = os.path.join(args.tmp_dir, f"merged_{left}_{right}.npz")
        merge_two_npz_for_combo(
            paths[left], paths[right],
            save_path=merged_path,
            left_name=left, right_name=right
        )

        pred, (srcc, plcc, krcc) = compute_metrics_for_npz(
            merged_path,
            benchmark_name=combo_name,
            lean_kwargs=lean_kwargs,
            verbose=args.verbose_align,
        )
        print(f"{combo_name:12s}  SRCC={srcc:.6f}  PLCC={plcc:.6f}  KRCC={krcc:.6f}")
        y_trues_all.append(gt)
        y_preds_all.append(pred)
        labels_all.append(combo_name.replace("-", "+"))

    overall_all_pred = np.concatenate([np.asarray(p, dtype=float) for p in y_preds_all], axis=0)
    overall_all_gt = np.concatenate([np.asarray(g, dtype=float) for g in y_trues_all], axis=0)
    srcc_all, plcc_all, krcc_all = _corr(overall_all_pred, overall_all_gt)

    print("\n===== Overall (ALL points: benches + combos) =====")
    print(f"overallALL SRCC={srcc_all:.6f}  PLCC={plcc_all:.6f}  KRCC={krcc_all:.6f}  (N={len(overall_all_gt)})")

    if args.plot:
        out_png = os.path.join(args.out_dir, args.plot_name)
        plot_fit_vs_gt_multi_from_bench(
            y_trues=y_trues_all,
            y_preds=y_preds_all,
            labels=labels_all,
            out_png=out_png,
            title=f"Learning Ease (Ours)",
            fontsize=18,
            # fit_degree=args.fit_degree,
            # y_scale_range=(args.y_lo, args.y_hi),
        )
        print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import re
# import argparse
# import numpy as np
# from scipy.stats import spearmanr, pearsonr, kendalltau
# import matplotlib.pyplot as plt

# from dataeval.metric.leanability import compute_leanability_from_npzdata


# # =========================
# # 0) Correlation helper
# # =========================
# def _corr(pred, gt):
#     pred = np.asarray(pred, dtype=float)
#     gt = np.asarray(gt, dtype=float)
#     if pred.shape[0] != gt.shape[0]:
#         raise ValueError(f"Length mismatch: pred={len(pred)} gt={len(gt)}")
#     srcc, _ = spearmanr(pred, gt)
#     plcc, _ = pearsonr(pred, gt)
#     krcc, _ = kendalltau(pred, gt)
#     return float(srcc), float(plcc), float(krcc)


# # =========================
# # 0.5) Plot helper
# # =========================
# def plot_fit_vs_gt_multi_from_bench(
#     y_trues, y_preds, labels,
#     out_png,
#     title="Learning Ease vs GT (ALL Points)",
#     fontsize=18,
#     fit_degree=3,
#     y_scale_range=(20, 100),
# ):
#     markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'H', '8', '<', '>', '|', '_', '.', ','] * 6
#     base_colors = plt.get_cmap('tab10').colors
#     colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

#     all_gt = np.concatenate([np.asarray(a, dtype=float) for a in y_trues], axis=0)
#     all_pred = np.concatenate([np.asarray(a, dtype=float) for a in y_preds], axis=0)

#     y_lo, y_hi = y_scale_range
#     pmin, pmax = float(all_pred.min()), float(all_pred.max())
#     if abs(pmax - pmin) < 1e-12:
#         all_pred_scaled = np.full_like(all_pred, (y_lo + y_hi) / 2.0, dtype=float)
#     else:
#         all_pred_scaled = y_lo + (all_pred - pmin) / (pmax - pmin) * (y_hi - y_lo)

#     deg = int(fit_degree)
#     coefs = np.polyfit(all_gt, all_pred_scaled, deg)
#     poly = np.poly1d(coefs)
#     x_fit = np.linspace(float(all_gt.min()), float(all_gt.max()), 200)
#     y_fit = poly(x_fit)

#     plt.figure(figsize=(11, 5))
#     start = 0
#     for i, gt_i in enumerate(y_trues):
#         gt_i = np.asarray(gt_i, dtype=float)
#         n = len(gt_i)
#         pred_scaled_i = all_pred_scaled[start:start + n]

#         plt.scatter(
#             gt_i, pred_scaled_i,
#             label=labels[i],
#             marker=markers[i],
#             color=colors[i],
#             s=55,
#             alpha=0.75,
#             edgecolor='None',
#         )
#         start += n

#     plt.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label=f'Poly Fit (deg={deg})')

#     srcc, plcc, krcc = _corr(all_pred, all_gt)
#     plt.text(
#         0.02, 0.98,
#         f"SRCC={srcc:.3f}  KRCC={krcc:.3f}  PLCC={plcc:.3f}\nN={len(all_gt)}",
#         transform=plt.gca().transAxes,
#         va='top',
#         fontsize=fontsize - 2,
#         bbox=dict(facecolor='white', alpha=0.65, edgecolor='gray'),
#     )

#     plt.xlabel("Ground Truth", fontsize=fontsize)
#     plt.ylabel("Predicted (scaled)", fontsize=fontsize)
#     plt.title(title, fontsize=fontsize + 2)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=fontsize - 4, loc='lower right', bbox_to_anchor=(1.45, 0.0))

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#     plt.savefig(out_png, dpi=300)
#     plt.close()
#     return dict(srcc=srcc, plcc=plcc, krcc=krcc, fit_coefs=coefs)


# # =========================
# # 1) GT + Task order (use yours)
# # =========================
# GTS = {
#     # openvla
#     "goal":    np.array([40, 52, 90, 82, 96, 96, 78, 40, 88, 94], dtype=float),
#     "object":  np.array([88, 70, 90, 84, 72, 86, 88, 94, 90, 86], dtype=float),
#     "spatial": np.array([94, 96, 76, 96, 82, 84, 88, 90, 80, 72], dtype=float),
#     "ten":     np.array([52, 34, 48, 20, 62, 58, 66, 44, 38, 68], dtype=float),

#     # 20D combos
#     "ten-object":     np.array([54, 50, 24, 14, 48, 48, 74, 36, 30, 64, 84, 66, 94, 88, 76, 76, 92, 88, 76, 92], dtype=float),
#     "goal-object":    np.array([64, 48, 88, 86, 94, 94, 74, 50, 70, 96, 90, 80, 90, 86, 82, 92, 96, 92, 92, 92], dtype=float),
#     "spatial-10":     np.array([94, 92, 82, 92, 84, 90, 90, 100, 72, 76, 50, 42, 38, 32, 54, 44, 66, 44, 60, 72], dtype=float),
#     "spatial-goal":   np.array([90, 90, 72, 96, 84, 86, 86, 94, 84, 66, 60, 44, 88, 90, 90, 94, 72, 42, 90, 94], dtype=float),
#     "spatial-object": np.array([94, 76, 68, 94, 80, 86, 94, 72, 88, 68, 88, 72, 96, 88, 78, 84, 92, 96, 94, 92], dtype=float),
#     "goal+ten":       np.array([58, 52, 82, 84, 94, 98, 64, 38, 92, 98, 52, 52, 58, 28, 70, 54, 78, 44, 56, 64], dtype=float),
# }

# BENCH_ALIAS = {"10": "ten", "Goal": "goal", "Spatial": "spatial"}

# TASK_ORDER_TEXT = {
#     "goal": [
#         "open the middle drawer of the cabinet demo",
#         "open the top drawer and put the bowl inside demo",
#         "push the plate to the front of the stove demo",
#         "put the bowl on the plate demo",
#         "put the bowl on the stove demo",
#         "put the bowl on top of the cabinet demo",
#         "put the cream cheese in the bowl demo",
#         "put the wine bottle on the rack demo",
#         "put the wine bottle on top of the cabinet demo",
#         "turn on the stove demo",
#     ],
#     "object": [
#         "pick up the alphabet soup and place it in the basket demo",
#         "pick up the bbq sauce and place it in the basket demo",
#         "pick up the butter and place it in the basket demo",
#         "pick up the chocolate pudding and place it in the basket demo",
#         "pick up the cream cheese and place it in the basket demo",
#         "pick up the ketchup and place it in the basket demo",
#         "pick up the milk and place it in the basket demo",
#         "pick up the orange juice and place it in the basket demo",
#         "pick up the salad dressing and place it in the basket demo",
#         "pick up the tomato sauce and place it in the basket demo",
#     ],
#     "spatial": [
#         "pick up the black bowl between the plate and the ramekin and place it on the plate demo",
#         "pick up the black bowl from table center and place it on the plate demo",
#         "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate demo",
#         "pick up the black bowl next to the cookie box and place it on the plate demo",
#         "pick up the black bowl next to the plate and place it on the plate demo",
#         "pick up the black bowl next to the ramekin and place it on the plate demo",
#         "pick up the black bowl on the cookie box and place it on the plate demo",
#         "pick up the black bowl on the ramekin and place it on the plate demo",
#         "pick up the black bowl on the stove and place it on the plate demo",
#         "pick up the black bowl on the wooden cabinet and place it on the plate demo",
#     ],
#     "ten": [
#         "KITCHEN SCENE3 turn on the stove and put the moka pot on it demo",
#         "KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it demo",
#         "KITCHEN SCENE6 put the yellow and white mug in the microwave and close it demo",
#         "KITCHEN SCENE8 put both moka pots on the stove demo",
#         "LIVING ROOM SCENE1 put both the alphabet soup and the cream cheese box in the basket demo",
#         "LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket demo",
#         "LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket demo",
#         "LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate demo",
#         "LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate demo",
#         "STUDY SCENE1 pick up the book and place it in the back compartment of the caddy demo",
#     ],
# }


# # =========================
# # 2) Text normalize + fuzzy scoring
# # =========================
# def _norm_text(s: str) -> str:
#     s = str(s).strip().lower()
#     s = s.replace(" demo", "")
#     s = re.sub(r"\s+", " ", s)
#     s = re.sub(r"[^\w\s]", "", s)
#     return s


# def _fuzzy_score(gold_norm: str, cand_norm: str) -> float:
#     gtok = set(gold_norm.split())
#     ctok = set(cand_norm.split())
#     if not gtok and not ctok:
#         return 0.0
#     inter = len(gtok & ctok)
#     union = len(gtok | ctok) + 1e-9
#     jacc = inter / union
#     sub = 1.0 if gold_norm in cand_norm else 0.0
#     len_bonus = min(len(cand_norm), 200) / 200.0
#     return 2.0 * jacc + 0.8 * sub + 0.2 * len_bonus


# # =========================
# # 3) Align 10 tasks with bench-restricted candidate set
# # =========================
# def _load_obj_dict(x):
#     if x is None:
#         return None
#     if isinstance(x, dict):
#         return x
#     if isinstance(x, np.ndarray) and x.shape == ():
#         try:
#             v = x.item()
#             return v if isinstance(v, dict) else None
#         except Exception:
#             return None
#     try:
#         v = x.item()
#         return v if isinstance(v, dict) else None
#     except Exception:
#         return None


# def _align_1x10(per_task: dict, bench: str, *, task_desc_by_id=None, allowed_task_ids=None, verbose=False) -> np.ndarray:
#     bench = bench.lower()
#     gold_order = TASK_ORDER_TEXT[bench]

#     if task_desc_by_id is None:
#         raise ValueError(f"[{bench}] task_descriptions_by_id is required for fuzzy align.")
#     if isinstance(task_desc_by_id, np.ndarray) and task_desc_by_id.shape == ():
#         task_desc_by_id = task_desc_by_id.item()

#     if allowed_task_ids is None:
#         allowed = set(int(k) for k in per_task.keys())
#     else:
#         allowed = set(int(x) for x in allowed_task_ids)
#         allowed &= set(int(k) for k in per_task.keys())

#     tid2nd = {}
#     for tid, desc in task_desc_by_id.items():
#         tid = int(tid)
#         if tid in allowed:
#             tid2nd[tid] = _norm_text(desc)

#     if len(tid2nd) == 0:
#         raise ValueError(f"[{bench}] allowed task id set empty after intersection; check combo meta.")

#     nd2tids = {}
#     for tid, nd in tid2nd.items():
#         nd2tids.setdefault(nd, []).append(tid)

#     chosen_ids = []
#     for gold in gold_order:
#         g = _norm_text(gold)

#         exact = nd2tids.get(g, [])
#         if len(exact) == 1:
#             chosen_ids.append(exact[0])
#             continue

#         best_tid = None
#         best_score = -1e9
#         for tid, nd in tid2nd.items():
#             sc = _fuzzy_score(g, nd)
#             if sc > best_score + 1e-12:
#                 best_score = sc
#                 best_tid = tid
#             elif abs(sc - best_score) <= 1e-12:
#                 if best_tid is None or tid < best_tid:
#                     best_tid = tid

#         if best_tid is None:
#             raise ValueError(f"[{bench}] cannot match gold='{gold}' in allowed pool.")
#         chosen_ids.append(best_tid)

#     if verbose:
#         print(f"[ALIGN {bench}] chosen task_ids (gold order): {chosen_ids}")

#     pred = []
#     for tid in chosen_ids:
#         if tid in per_task:
#             pred.append(per_task[tid])
#         elif str(tid) in per_task:
#             pred.append(per_task[str(tid)])
#         else:
#             raise KeyError(f"[{bench}] chosen tid {tid} missing in per_task.")
#     return np.asarray(pred, dtype=float)


# def compute_correlation(result, benchmark_name, *, combo_meta=None, verbose=False):
#     bench = BENCH_ALIAS.get(benchmark_name, benchmark_name).lower()
#     gt = GTS.get(bench)
#     if gt is None:
#         raise ValueError(f"Benchmark '{benchmark_name}' mapped to '{bench}' not found in GTS.")

#     per_task = result["leanability_per_task"]
#     tdesc = result.get("task_descriptions_by_id", None)

#     if bench in ("goal", "object", "spatial", "ten"):
#         pred = _align_1x10(per_task, bench, task_desc_by_id=tdesc, allowed_task_ids=None, verbose=verbose)
#         return pred, _corr(pred, gt)

#     if "-" in bench or "+" in bench:
#         if combo_meta is None:
#             raise ValueError(f"[{bench}] combo_meta is required for combo alignment.")

#         if "-" in bench:
#             left, right = bench.split("-", 1)
#         else:
#             left, right = bench.split("+", 1)

#         left = BENCH_ALIAS.get(left, left).lower()
#         right = BENCH_ALIAS.get(right, right).lower()

#         left_ids = combo_meta["left_task_ids"] if combo_meta["left_name"] == left else combo_meta["right_task_ids"]
#         right_ids = combo_meta["right_task_ids"] if combo_meta["right_name"] == right else combo_meta["left_task_ids"]

#         pred_left = _align_1x10(per_task, left, task_desc_by_id=tdesc, allowed_task_ids=left_ids, verbose=verbose)
#         pred_right = _align_1x10(per_task, right, task_desc_by_id=tdesc, allowed_task_ids=right_ids, verbose=verbose)
#         pred = np.concatenate([pred_left, pred_right], axis=0)
#         return pred, _corr(pred, gt)

#     raise ValueError(f"Unsupported benchmark name: {benchmark_name} (mapped to {bench})")


# # =========================
# # 4) NPZ merge with COMBO META
# # =========================
# def _npz_to_dict(npzfile) -> dict:
#     return {k: npzfile[k] for k in npzfile.files}


# def _get_task_id_key(d: dict) -> str:
#     for k in ["task_id", "task_ids", "taskid", "task_index", "task_indices"]:
#         if k in d:
#             return k
#     raise KeyError(f"Cannot find task id key in npz. keys={list(d.keys())}")


# def remap_task_ids_in_npz_dict(npz_dict: dict, offset: int):
#     tid_key = _get_task_id_key(npz_dict)
#     npz_dict[tid_key] = npz_dict[tid_key].astype(np.int64) + int(offset)

#     if "task_descriptions_by_id" in npz_dict:
#         desc_map = _load_obj_dict(npz_dict["task_descriptions_by_id"])
#         if isinstance(desc_map, dict):
#             new_map = {int(k) + int(offset): v for k, v in desc_map.items()}
#             npz_dict["task_descriptions_by_id"] = np.array(new_map, dtype=object)

#     return tid_key


# def _should_concat_by_sample_dim(va, vb, Na, Nb):
#     if not (isinstance(va, np.ndarray) and isinstance(vb, np.ndarray)):
#         return False
#     if va.ndim < 1 or vb.ndim < 1:
#         return False
#     if va.shape[0] != Na or vb.shape[0] != Nb:
#         return False
#     if va.ndim != vb.ndim:
#         return False
#     if va.shape[1:] != vb.shape[1:]:
#         return False
#     return True


# def merge_two_npz_for_combo(npz_path_a: str, npz_path_b: str, *, save_path: str, left_name: str, right_name: str):
#     a = _npz_to_dict(np.load(npz_path_a, allow_pickle=True))
#     b = _npz_to_dict(np.load(npz_path_b, allow_pickle=True))

#     tid_key_a = _get_task_id_key(a)
#     tid_key_b = _get_task_id_key(b)
#     Na = int(a[tid_key_a].shape[0])
#     Nb = int(b[tid_key_b].shape[0])

#     left_task_ids = sorted(set(a[tid_key_a].astype(np.int64).tolist()))
#     right_task_ids_raw = sorted(set(b[tid_key_b].astype(np.int64).tolist()))

#     max_a = int(np.max(a[tid_key_a].astype(np.int64)))
#     offset = max_a + 1
#     remap_task_ids_in_npz_dict(b, offset=offset)
#     right_task_ids = [int(x) + int(offset) for x in right_task_ids_raw]

#     merged = {}
#     all_keys = sorted(set(a.keys()) | set(b.keys()))
#     for k in all_keys:
#         in_a, in_b = (k in a), (k in b)
#         if in_a and in_b:
#             va, vb = a[k], b[k]

#             if k == "task_descriptions_by_id":
#                 da = _load_obj_dict(va)
#                 db = _load_obj_dict(vb)
#                 if isinstance(da, dict) and isinstance(db, dict):
#                     out = dict(da)
#                     out.update(db)
#                     merged[k] = np.array(out, dtype=object)
#                 else:
#                     merged[k] = va
#                 continue

#             if _should_concat_by_sample_dim(va, vb, Na, Nb):
#                 merged[k] = np.concatenate([va, vb], axis=0)
#             else:
#                 merged[k] = va
#         elif in_a:
#             merged[k] = a[k]
#         else:
#             merged[k] = b[k]

#     combo_meta = {
#         "left_name": left_name.lower(),
#         "right_name": right_name.lower(),
#         "left_task_ids": left_task_ids,
#         "right_task_ids": right_task_ids,
#         "offset": int(offset),
#     }
#     merged["_combo_meta"] = np.array(combo_meta, dtype=object)

#     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#     np.savez_compressed(save_path, **merged)
#     return save_path, combo_meta


# # =========================
# # 5) Leanability wrapper (takes **lean_kwargs)
# # =========================
# def compute_metrics_for_npz(npz_path, benchmark_name, *, lean_kwargs=None, verbose=False):
#     npzdata = np.load(npz_path, allow_pickle=True)

#     combo_meta = None
#     if "_combo_meta" in npzdata.files:
#         combo_meta = _load_obj_dict(npzdata["_combo_meta"])

#     lean_kwargs = dict(lean_kwargs or {})

#     # ---- Backward-compat: drop unsupported kwargs if needed ----
#     try:
#         result = compute_leanability_from_npzdata(npzdata=npzdata, **lean_kwargs)
#     except TypeError as e:
#         # likely "unexpected keyword argument"
#         msg = str(e)
#         bad_keys = []
#         for k in list(lean_kwargs.keys()):
#             if k in msg:
#                 bad_keys.append(k)

#         # if cannot parse, try removing known "new" keys
#         if not bad_keys:
#             for k in ["task_knn", "task_temp"]:
#                 if k in lean_kwargs:
#                     bad_keys.append(k)

#         if bad_keys:
#             for k in bad_keys:
#                 lean_kwargs.pop(k, None)
#             print(f"[WARN] compute_leanability_from_npzdata does not accept {bad_keys}, dropped for compat.")
#             result = compute_leanability_from_npzdata(npzdata=npzdata, **lean_kwargs)
#         else:
#             raise

#     pred, (srcc, plcc, krcc) = compute_correlation(
#         result, benchmark_name,
#         combo_meta=combo_meta,
#         verbose=verbose
#     )
#     return pred, (srcc, plcc, krcc)


# # =========================
# # 6) CLI
# # =========================
# def build_argparser():
#     p = argparse.ArgumentParser(
#         description="Evaluate Leanability against GT for LIBERO benchmarks (+ combos).",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # paths
#     p.add_argument("--npz_goal", type=str, required=True)
#     p.add_argument("--npz_object", type=str, required=True)
#     p.add_argument("--npz_spatial", type=str, required=True)
#     p.add_argument("--npz_ten", type=str, required=True)

#     # output
#     p.add_argument("--out_dir", type=str, default="./plots")
#     p.add_argument("--tmp_dir", type=str, default="./tmp")
#     p.add_argument("--plot", action="store_true", help="Save one ALL-points plot.")
#     p.add_argument("--plot_name", type=str, default="learning_ease_fit_ALL_points.png")
#     p.add_argument("--fit_degree", type=int, default=3)
#     p.add_argument("--y_lo", type=float, default=20.0)
#     p.add_argument("--y_hi", type=float, default=100.0)

#     # debug
#     p.add_argument("--dry_run", action="store_true")
#     p.add_argument("--verbose_align", action="store_true")

#     # ---- leanability params ----
#     p.add_argument("--beta", type=float, default=0.5)
#     p.add_argument("--sigma_task", type=float, default=1e-7)
#     p.add_argument("--sigma_center", type=float, default=1e-6)
#     p.add_argument("--pi_scale", type=float, default=0.02)

#     p.add_argument("--transfer_mode", type=str, default="harmonic",
#                    choices=["semantic", "visual_center", "harmonic"])

#     p.add_argument("--intra_sim", type=str, default="exp", choices=["exp", "linear"])
#     p.add_argument("--tau_intra", type=float, default=0.05)
#     p.add_argument("--adaptive_tau", action="store_true")
#     p.add_argument("--no_adaptive_tau", dest="adaptive_tau", action="store_false")
#     p.set_defaults(adaptive_tau=True)

#     p.add_argument("--tau_floor", type=float, default=0.03)
#     p.add_argument("--tau_ceiling", type=float, default=0.5)

#     p.add_argument("--length_penalty", action="store_true")
#     p.add_argument("--no_length_penalty", dest="length_penalty", action="store_false")
#     p.set_defaults(length_penalty=True)

#     p.add_argument("--use_self_loop", action="store_true")
#     p.add_argument("--no_use_self_loop", dest="use_self_loop", action="store_false")
#     p.set_defaults(use_self_loop=True)

#     p.add_argument("--alpha", type=float, default=0.35)

#     # ---- NEW: task transfer graph stabilization ----
#     p.add_argument("--task_knn", type=int, default=5,
#                    help="Keep top-k neighbors per task row for task transfer graph.")
#     p.add_argument("--task_temp", type=float, default=0.07,
#                    help="Softmax temperature for task transfer graph (smaller -> sharper).")

#     p.add_argument("--eps", type=float, default=1e-12)
#     p.add_argument("--debug", action="store_true")

#     return p


# def main():
#     args = build_argparser().parse_args()

#     paths = {
#         "goal": args.npz_goal,
#         "object": args.npz_object,
#         "spatial": args.npz_spatial,
#         "ten": args.npz_ten,
#     }

#     lean_kwargs = dict(
#         beta=args.beta,
#         sigma_task=args.sigma_task,
#         sigma_center=args.sigma_center,
#         pi_scale=args.pi_scale,

#         transfer_mode=args.transfer_mode,

#         intra_sim=args.intra_sim,
#         tau_intra=args.tau_intra,
#         adaptive_tau=args.adaptive_tau,
#         tau_floor=args.tau_floor,
#         tau_ceiling=args.tau_ceiling,
#         length_penalty=args.length_penalty,

#         use_self_loop=args.use_self_loop,
#         alpha=args.alpha,

#         # NEW
#         task_knn=args.task_knn,
#         task_temp=args.task_temp,

#         eps=args.eps,
#         debug=args.debug,
#     )

#     print("==== Leanability kwargs ====")
#     for k in sorted(lean_kwargs.keys()):
#         print(f"{k:16s}: {lean_kwargs[k]}")

#     if args.dry_run:
#         return

#     combos = [
#         ("goal", "object", "goal-object"),
#         ("spatial", "goal", "spatial-goal"),
#         ("spatial", "ten", "spatial-10"),
#         ("ten", "object", "ten-object"),
#         ("spatial", "object", "spatial-object"),
#         ("goal", "ten", "goal+ten"),
#     ]

#     # collect ALL points
#     y_trues_all, y_preds_all, labels_all = [], [], []

#     print("\n===== Per-benchmark Metrics (10D each) =====")
#     for bench in ["goal", "object", "spatial", "ten"]:
#         pred, (srcc, plcc, krcc) = compute_metrics_for_npz(
#             paths[bench],
#             benchmark_name=bench,
#             lean_kwargs=lean_kwargs,
#             verbose=args.verbose_align,
#         )
#         gt = GTS[bench]
#         print(f"{bench:8s}  SRCC={srcc:.6f}  PLCC={plcc:.6f}  KRCC={krcc:.6f}")
#         y_trues_all.append(gt)
#         y_preds_all.append(pred)
#         labels_all.append(bench)

#     print("\n===== 20D Combos (merge npz then compute once) =====")
#     os.makedirs(args.tmp_dir, exist_ok=True)

#     for left, right, combo_name in combos:
#         combo_key = combo_name.lower()
#         gt = GTS[combo_key]

#         merged_path = os.path.join(args.tmp_dir, f"merged_{left}_{right}.npz")
#         merge_two_npz_for_combo(
#             paths[left], paths[right],
#             save_path=merged_path,
#             left_name=left, right_name=right
#         )

#         pred, (srcc, plcc, krcc) = compute_metrics_for_npz(
#             merged_path,
#             benchmark_name=combo_name,
#             lean_kwargs=lean_kwargs,
#             verbose=args.verbose_align,
#         )
#         print(f"{combo_name:12s}  SRCC={srcc:.6f}  PLCC={plcc:.6f}  KRCC={krcc:.6f}")
#         y_trues_all.append(gt)
#         y_preds_all.append(pred)
#         labels_all.append(combo_name)

#     overall_all_pred = np.concatenate([np.asarray(p, dtype=float) for p in y_preds_all], axis=0)
#     overall_all_gt = np.concatenate([np.asarray(g, dtype=float) for g in y_trues_all], axis=0)
#     srcc_all, plcc_all, krcc_all = _corr(overall_all_pred, overall_all_gt)

#     print("\n===== Overall (ALL points: benches + combos) =====")
#     print(f"overallALL SRCC={srcc_all:.6f}  PLCC={plcc_all:.6f}  KRCC={krcc_all:.6f}  (N={len(overall_all_gt)})")

#     if args.plot:
#         out_png = os.path.join(args.out_dir, args.plot_name)
#         plot_fit_vs_gt_multi_from_bench(
#             y_trues=y_trues_all,
#             y_preds=y_preds_all,
#             labels=labels_all,
#             out_png=out_png,
#             title=f"Learning Ease vs GT (ALL points, N={len(overall_all_gt)})",
#             fontsize=18,
#             fit_degree=args.fit_degree,
#             y_scale_range=(args.y_lo, args.y_hi),
#         )
#         print(f"Saved plot: {out_png}")


# if __name__ == "__main__":
#     main()