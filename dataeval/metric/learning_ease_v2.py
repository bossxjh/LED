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
        beta=0.5,
        sigma_task=0.001,
        sigma_center=0.001,  #这个参数其实已经没有用了，可以删掉
        pi_scale=0.01698373,
        # ---- minimal new knobs ----
        transfer_mode="semantic",   # "semantic" or "visual_center"
):
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
            # ---- minimal change: still define something ----
            task_centers[g["task_id"]] = X_t.mean(axis=0) if N_t>0 else np.zeros(X_t.shape[1])
            continue

        dists_t = pairwise_distances(X_t, X_t, metric="euclidean") ** 2
        sigma_t = sigma_task
        S_t = np.exp(-dists_t / (2 * sigma_t**2))

        rho_t = S_t.mean(axis=1)
        E_t = rho_t.mean()
        E_t = E_t / np.log10(1 + g["task_length"])

        d_avg = np.mean(np.sqrt(dists_t[np.triu_indices(N_t, 1)]))
        R_t = covariance_entropy(X_t) * np.tanh(d_avg / sigma_t)

        L_t_raw[g["task_id"]] = (R_t**beta) * (E_t**(1-beta))
        print(L_t_raw)

        # ------------------------------
        # ✅ minimal change: task representation for transfer
        # ------------------------------
        if transfer_mode == "semantic":
            te = np.asarray(g["task_embeddings"])
            if te.ndim == 2:      # (Nt,E) -> mean pool
                te = te.mean(axis=0)
            task_centers[g["task_id"]] = te
        else:
            # original behavior
            task_centers[g["task_id"]] = X_t.mean(axis=0)

    # Step 2: 计算 task → task 相似度（semantic 用 cosine）

    centers_array = np.stack([task_centers[t] for t in task_ids])
    # 先归一化
    # norm = np.linalg.norm(centers_array, axis=1, keepdims=True) + 1e-12
    # centers_array = centers_array / norm
    # centers_array = centers_array 

    # cosine similarity
    S_task = centers_array @ centers_array.T   # [-1,1]
    # # 映射到 [0,1]
    S_task = np.clip(S_task, 0.0, 1.0)         # 负的直接当 0（很多情况下不会出现）
    print(S_task)

    # Step 3: 跨任务加权 L_t_adj（不改结构）
    task_scores = {}
    total_demo_count = sum(len(task["demo_lengths"]) for task in task_groups)
    for i, t in enumerate(task_ids):
        task = task_groups[i]
        Nt = len(task["demo_lengths"])  # Nt 是当前任务的示范数
        pi_t = Nt/total_demo_count
        pi_t = np.tanh(pi_t / pi_scale)

        L_t_adj = sum(S_task[i, j] * L_t_raw[task_ids[j]] for j in range(n_tasks))
        task_scores[t] = L_t_adj * pi_t

    dataset_score = np.mean(list(task_scores.values()))
    return dataset_score, task_scores