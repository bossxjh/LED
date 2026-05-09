# dataeval/metric/utils/task_grouping.py
import numpy as np
import hashlib


def _fallback_text_embedding(text, dim=256):
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().replace("_", " ").split():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def group_by_task(npzdata):
    features = npzdata["features"]                  # (N, D) or (N, T, D)
    task_ids = npzdata["task_ids"].astype(int)      # (N,)
    task_lengths = npzdata["task_lengths"]          # (N,)
    demo_lengths = npzdata["demo_lengths"]          # (N,)
    task_descriptions = npzdata["task_descriptions"]# (N,)

    # ---- NEW: optional task embeddings ----
    task_emb = None
    if "task_embeddings" in npzdata.files:
        task_emb = npzdata["task_embeddings"]       # (N, E) or (N_task, E) depending on how you saved
    elif "task_embedding" in npzdata.files:
        task_emb = npzdata["task_embedding"]

    # flatten feature if needed
    if len(features.shape) == 3:
        N, T, D = features.shape
        features = features.reshape(N, T * D)

    task_groups = []
    print("features shape", features.shape)

    unique_task_ids = np.unique(task_ids)

    for task_id in unique_task_ids:
        mask = task_ids == task_id

        g = {
            "task_id": int(task_id),
            "task_length": float(task_lengths[mask][0]),
            "task_description": task_descriptions[mask][0],
            "features": features[mask],          # (Nt, D)
            "demo_lengths": demo_lengths[mask],  # (Nt,)
        }

        # ---- NEW: attach task embedding if available ----
        if task_emb is not None:
            # Most common case: you saved per-demo embeddings -> shape (N, E)
            # Same task: embeddings repeated, so take the first one.
            if task_emb.shape[0] == task_ids.shape[0]:
                g["task_embeddings"] = task_emb[mask][0]   # (E,)
            else:
                # If you saved per-task embeddings, you need a mapping.
                # Minimal fallback: try index by sorted unique_task_ids order.
                # (Only correct if you saved in exactly this order.)
                idx = np.where(unique_task_ids == task_id)[0][0]
                g["task_embeddings"] = task_emb[idx]
        else:
            g["task_embeddings"] = _fallback_text_embedding(g["task_description"])

        task_groups.append(g)

    return task_groups
# # dataeval/utils/task_grouping.py
# import numpy as np


# def group_by_task(npzdata):
#     features = npzdata["features"]                  # (N, D)
#     task_ids = npzdata["task_ids"].astype(int)      # (N,)
#     task_lengths = npzdata["task_lengths"]          # (N,)
#     demo_lengths = npzdata["demo_lengths"]          # (N,)
#     task_descriptions = npzdata["task_descriptions"]# (N,)

#     if len(features.shape) == 3:
#         N, T, D = features.shape
#         features = features.reshape(N, T*D)

#     task_groups = []
#     print("features shape", features.shape)

#     for task_id in np.unique(task_ids):
#         mask = task_ids == task_id

#         task_groups.append({
#             "task_id": int(task_id),
#             # 取第一个就行，因为所有 demo 的值都是一样的
#             "task_length": float(task_lengths[mask][0]),
#             "task_description": task_descriptions[mask][0],
#             "features": features[mask],          # (Nt, D)
#             "demo_lengths": demo_lengths[mask],  # (Nt,)
#         })

#     return task_groups
