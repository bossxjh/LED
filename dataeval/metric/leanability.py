# dataeval/metric/leanability.py

import numpy as np
from .utils.task_grouping import group_by_task
from .learning_ease_v5 import compute_learning_ease_with_task_transfer

def compute_leanability_from_npzdata(
    npzdata,
    beta: float = 0.5,
    sigma_task=0.001,               # 任务内样本相似度带宽（None 则用 0.001 保底）
    sigma_center=0.001,              # 任务中心相似度带宽（None 则用 median heuristic）
    pi_scale=0.01698373,

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
    alpha=1.0,               # residual mix: L_adj = (1-a)L_raw + a*(W@L_raw)
    task_knn=5,              # NEW: keep top-k neighbors per task row
    task_temp=0.07,          # NEW: softmax temperature for task graph
    eps=1e-12,
    debug=False,


):
    # -------- load & group --------
    task_groups = group_by_task(npzdata)

    # ---- build mapping: task_id -> task_description ----
    task_desc_by_id = {}
    for g in task_groups:
        tid = int(g["task_id"])
        desc = str(g.get("task_description", ""))
        task_desc_by_id[tid] = desc

    # -------- DEBUG: inspect task_groups --------
    # print(f"num_tasks = {len(task_groups)}")
    # for i, g in enumerate(task_groups[:10]):
    #     print(f"\n[Task {i}]")
    #     print("  task_id:", g["task_id"])
    #     print("  task_length:", g["task_length"])
    #     print("  task_description:", g["task_description"])
    #     print("  features shape:", g["features"].shape)
    #     print("  demo_lengths shape:", g["demo_lengths"].shape)

    # input("Press Enter to continue...")

    # -------- core leanability computation --------
    dataset_score, task_scores = compute_learning_ease_with_task_transfer(
        task_groups=task_groups,
        beta=beta,
        sigma_task=sigma_task,               # 任务内样本相似度带宽（None 则用 0.001 保底）
        sigma_center=sigma_center,  
        pi_scale=pi_scale,

        transfer_mode=transfer_mode,  # "semantic" or "visual_center"
        # ---- intra-task cosine knobs ----
        intra_sim=intra_sim,         # "exp" or "linear"
        tau_intra=tau_intra,          # used if intra_sim="exp"
        adaptive_tau=adaptive_tau,       # per-task tau based on median(1-cos)
        tau_floor=tau_floor,
        tau_ceiling=tau_ceiling,
        length_penalty=length_penalty,
        # ---- inter-task transfer knobs ----
        use_self_loop=use_self_loop,
        alpha=alpha,               # residual mix: L_adj = (1-a)L_raw + a*(W@L_raw). set <1 to avoid oversmooth
        eps=eps,
        debug=debug,
        task_knn=task_knn,              # NEW: keep top-k neighbors per task row
        task_temp=task_temp,          # NEW: softmax temperature for task graph
    )
    print("leanability_dataset",float(dataset_score))
    return {
        "leanability_dataset": float(dataset_score),
        "leanability_per_task": {
            int(k): float(v) for k, v in task_scores.items()
        },
        "task_descriptions_by_id": task_desc_by_id,   # ✅ 新增
        "num_tasks": len(task_groups),
        "beta": beta,
        "sigma_task": sigma_task,
        "sigma_center":sigma_center,
    }






# def compute_leanability_from_npzdata(
#     npzdata,
#     beta: float = 0.8,
#     sigma=None,
# ):
#     """
#     Compute dataset leanability from loaded npz data.

#     Args:
#         npzdata: result of np.load(..., allow_pickle=True)
#                  or a dict-like object with required fields
#         beta: trade-off coefficient
#         sigma: kernel bandwidth
#     """

#     # -------- basic fields --------
#     X = npzdata["features"]              # (N, D)
#     y = npzdata["task_ids"].astype(int)  # (N,)
#     task_lengths_demo = npzdata["task_lengths"]
#     dataset_name = str(npzdata["dataset_name"])

#     # -------- aggregate task-level lengths --------
#     task_length_dict = {}
#     for t in np.unique(y):
#         task_length_dict[int(t)] = float(
#             task_lengths_demo[y == t].mean()
#         )

#     # -------- call original leanability core --------
#     L_dataset, L_task = compute_learning_ease_with_task_transfer(
#         X=X,
#         y=y,
#         task_lengths=task_length_dict,
#         dataset_name=dataset_name,
#         beta=beta,
#         sigma=sigma,
#     )

#     # -------- package outputs --------
#     result = {
#         "dataset_score": float(L_dataset),
#         "task_scores": {int(k): float(v) for k, v in L_task.items()},
#         "beta": beta,
#         "sigma": sigma,
#         "num_demos": int(X.shape[0]),
#         "feature_dim": int(X.shape[1]),
#         "dataset_name": dataset_name,
#     }

#     return result
