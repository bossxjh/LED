import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# compute_learning_ease_with_task_transfer
# │
# ├─ Step 0: 基本准备（task_ids, 比例 pi）
# │
# ├─ Step 1: 对每个 task 单独算一个 L_t（task 内部可学习性）
# │   ├─ task 内样本相似度 S_t
# │   ├─ 局部 entropy（h_local）
# │   ├─ 表征复杂度 R_t
# │   ├─ 密度 / 有效覆盖度 E_t
# │   └─ 得到 raw L_t
# │
# ├─ Step 2: 任务之间的相似度（task → task 迁移）
# │   ├─ task center
# │   ├─ task similarity matrix
# │   └─ 跨任务加权 L_t_adj
# │
# └─ 输出：
#     ├─ L_dataset
#     └─ 每个 task 的 L_t_adj


def covariance_entropy(X):
    """
    X: (N_t, D)
    返回归一化协方差熵
    """
    if X.shape[0] <= 1:
        return 0.0
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)  # (D, D)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / eigvals.sum()
    H = -np.sum(p * np.log(p))
    # 归一化到 [0,1]
    # H_norm = H / np.log(len(p))
    H_norm = H
    return H_norm

def compute_learning_ease_with_task_transfer(
        task_groups,
        # -------- 新增：所有可调超参 --------
        beta=0.5,                       # 局部复杂度 vs 有效覆盖的权衡
        sigma_task=0.001,               # 任务内样本相似度带宽（None 则用 0.001 保底）
        sigma_center=0.001,              # 任务中心相似度带宽（None 则用 median heuristic）
        pi_scale=0.01698373,            # pi 的 tanh 缩放（这个参数是模型相关的）
        # -------- 新增：模块选择消融 --------
        R_t_scale='tanh',               # R_t 里 d_avg 的缩放方式：'tanh' 或 'linear'
        E_t_log_base=10,                # E_t 里 log 的底（原文 log10）
        entropy_normalize=True,         # 是否把 covariance entropy 归一化到 [0,1]
        # -----------------------------------
):
    """
    Task-centric version of compute_learning_ease_with_task_transfer.
    
    Args:
        task_groups: list of dicts, each dict has
            {
                "task_id": int,
                "features": (Nt, D) np.array,
                "demo_lengths": (Nt,) np.array,
                "task_length": float,
                "task_description": str
            }
        beta: trade-off coefficient
        sigma: kernel bandwidth (float or None)
    
    Returns:
        dataset_score: float
        task_scores: dict, task_id -> float
    """

    task_ids = [g["task_id"] for g in task_groups]
    n_tasks = len(task_groups)

    # Step 1: 计算每个 task 的 L_t 和 task center
    L_t_raw = {}
    task_centers = {}

    for i, g in enumerate(task_groups):
        X_t = g["features"]
        N_t = X_t.shape[0]
        if N_t <= 1:
            L_t_raw[g["task_id"]] = 0.0
            task_centers[g["task_id"]] = X_t.mean(axis=0) if N_t>0 else np.zeros(X_t.shape[1])
            continue

        # pairwise squared distances
        dists_t = pairwise_distances(X_t, X_t, metric="euclidean") ** 2 #两两样本点之间的距离 
        sigma_t = sigma_task   # 任务内样本相似度带宽（None 则用 0.001 保底）
        # print("dists_t:",dists_t)
        S_t = np.exp(-dists_t / (2 * sigma_t**2))   #两两样本点之间相似度矩阵
        P_t = S_t / S_t.sum(axis=1, keepdims=True)  #每个点的概率密度分布

        # 计算E_t 
        rho_t = S_t.mean(axis=1)
        E_t = rho_t.mean()
        E_t = E_t / np.log10(1 + g["task_length"])

        # 计算R_t 可以按原来的 covariance_entropy
        d_avg = np.mean(np.sqrt(dists_t[np.triu_indices(N_t, 1)]))
        R_t = covariance_entropy(X_t) * np.tanh(d_avg / sigma_t) #形状*放缩长度

        # 计算L_t = (R_t**beta) * (E_t**(1-beta))
        L_t_raw[g["task_id"]] = (R_t**beta) * (E_t**(1-beta))
        print(L_t_raw)

        # task center
        task_centers[g["task_id"]] = X_t.mean(axis=0)

    # Step 2: 计算 task → task 相似度
    centers_array = np.stack([task_centers[t] for t in task_ids])
    center_dists = pairwise_distances(centers_array, centers_array, metric="euclidean") ** 2
    sigma_center = np.median(np.sqrt(center_dists)) if sigma_center is None else sigma_center
    S_task = np.exp(-center_dists / (2 * sigma_center**2))

    # Step 3: 跨任务加权 L_t_adj
    # 这里可以用 pi 权重或均匀权重
    task_scores = {}
    total_demo_count = sum(len(task["demo_lengths"]) for task in task_groups)

    for i, t in enumerate(task_ids):
        task = task_groups[i]
        # Nt = len(task["demo_lengths"])  # Nt 是当前任务的示范数
        pi_t = 1/10  # 任务的比例
        pi_t = np.tanh(pi_t / pi_scale)  # 应用 tanh 函数进行缩放


        L_t_adj = sum(S_task[i, j] * L_t_raw[task_ids[j]] for j in range(n_tasks))
        task_scores[t] = L_t_adj * pi_t

    # Step 4: dataset-level leanability
    dataset_score = np.mean(list(task_scores.values()))

    return dataset_score, task_scores