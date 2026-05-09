import numpy as np
from tqdm import tqdm

from dataeval.datasets import DATASET_PARSERS
from dataeval.datasets_meta import DATASET_PARSERS_META
from dataeval.embedding.text_embedding import TextEmbedder
from dataeval.models import MODEL_ADAPTERS
from dataeval.paths import BGE_MODEL_PATH


_ADAPTER_CACHE = {}


def _get_adapter(model_name):
    if model_name not in _ADAPTER_CACHE:
        _ADAPTER_CACHE[model_name] = MODEL_ADAPTERS[model_name]()
    return _ADAPTER_CACHE[model_name]


def extract_features_with_metadata(
    model_name,
    dataset_name,
    dataset_path,
    num_frames=3,
    batch_size=16,
    add_task_embedding=True,
    task_embed_model=None,
    task_embed_device=None,
    task_embed_batch_size=8,
    add_sample_features=True,
):
    if task_embed_model is None:
        task_embed_model = str(BGE_MODEL_PATH if BGE_MODEL_PATH.exists() else "BAAI/bge-large-en-v1.5")

    parser = DATASET_PARSERS_META[dataset_name]
    adapter = _get_adapter(model_name)

    demo_features_list = parser(dataset_path, num_frames=num_frames)
    total_demos = len(demo_features_list)

    task_emb_by_desc = {}
    if add_task_embedding:
        unique_desc = sorted(set(d["task_description"] for d in demo_features_list))
        embedder = TextEmbedder(model_name=task_embed_model, device=task_embed_device, normalize=True)
        embs = embedder.encode(unique_desc, batch_size=task_embed_batch_size, show_progress_bar=True)
        task_emb_by_desc = {desc: embs[i] for i, desc in enumerate(unique_desc)}

    all_features = []
    batch_frames = []
    batch_meta = []

    with tqdm(total=total_demos, desc="Extracting features") as pbar:
        for demo in demo_features_list:
            batch_frames.append(demo["frames"])

            meta = {
                "task_id": demo["task_id"],
                "task_length": demo["task_length"],
                "demo_length": demo["demo_length"],
                "task_description": demo["task_description"],
            }
            if add_task_embedding:
                meta["task_embedding"] = task_emb_by_desc[demo["task_description"]]

            for key in ["episode_index", "demo_uid", "native_episode_id_key", "native_episode_id"]:
                if key in demo:
                    meta[key] = demo[key]

            for key in ["actions", "action_energy", "action_smoothness", "action_jerk", "action_small_ratio"]:
                if key in demo:
                    value = demo[key]
                    if isinstance(value, (int, float, np.number)):
                        meta[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        meta[key] = value.astype(np.float32)
                    else:
                        meta[key] = value

            if "sampled_actions" in demo:
                meta["sampled_actions"] = np.asarray(demo["sampled_actions"], dtype=np.float32)

            batch_meta.append(meta)

            if len(batch_frames) == batch_size:
                feats = adapter.extract_batch(batch_frames)
                for feat, meta_i in zip(feats, batch_meta):
                    out = {**meta_i, "features": np.asarray(feat, dtype=np.float32)}
                    if add_sample_features:
                        out["sample_features"] = np.asarray(feat, dtype=np.float32)
                    all_features.append(out)

                batch_frames.clear()
                batch_meta.clear()
                pbar.update(batch_size)

        if batch_frames:
            feats = adapter.extract_batch(batch_frames)
            for feat, meta_i in zip(feats, batch_meta):
                out = {**meta_i, "features": np.asarray(feat, dtype=np.float32)}
                if add_sample_features:
                    out["sample_features"] = np.asarray(feat, dtype=np.float32)
                all_features.append(out)
            pbar.update(len(batch_frames))

    return all_features


def extract_features(model_name, dataset_name, dataset_path, num_frames=3, batch_size=16):
    parser = DATASET_PARSERS[dataset_name]
    adapter = _get_adapter(model_name)

    batch = []
    for frames in parser(dataset_path, num_frames=num_frames):
        batch.append(frames)
        if len(batch) == batch_size:
            feats = adapter.extract_batch(batch)
            for feat in feats:
                yield feat
            batch.clear()

    if batch:
        feats = adapter.extract_batch(batch)
        for feat in feats:
            yield feat
