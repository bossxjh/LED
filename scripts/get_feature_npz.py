import argparse
import os
import time

import numpy as np
from tqdm import tqdm


def run_one(dataset_path, dataset_name, save_dir, model_name="openvla", num_frames=3, batch_size=8):
    from dataeval.api import extract_features_with_metadata

    print(f"\n===== Processing {dataset_name} =====")
    start_time = time.time()

    demo_features = []
    extracted = extract_features_with_metadata(
        model_name,
        dataset_name,
        dataset_path,
        num_frames=num_frames,
        batch_size=batch_size,
        add_sample_features=True,
    )

    for feat_dict in tqdm(extracted, desc=f"Collecting {dataset_name}"):
        demo_features.append(feat_dict)

    if not demo_features:
        raise RuntimeError(f"No demos were extracted from dataset_path={dataset_path}")

    total_time = time.time() - start_time
    print(f"num demos: {len(demo_features)}")
    print(f"first demo keys: {demo_features[0].keys()}")
    print(f"first feature shape: {demo_features[0]['features'].shape}")
    print(f"elapsed: {total_time:.2f}s")

    features_array = np.stack([d["features"] for d in demo_features], axis=0)
    task_ids = np.array([d["task_id"] for d in demo_features], dtype=np.int64)
    task_lengths = np.array([d["task_length"] for d in demo_features], dtype=np.int64)
    demo_lengths = np.array([d["demo_length"] for d in demo_features], dtype=np.int64)
    task_descriptions = np.array([d["task_description"] for d in demo_features], dtype=object)
    task_embeddings = np.stack([d["task_embedding"] for d in demo_features], axis=0)

    sample_features = np.stack([d["sample_features"] for d in demo_features], axis=0)
    sample_actions = np.stack([d["sampled_actions"] for d in demo_features], axis=0)

    episode_indices = np.array([d.get("episode_index", -1) for d in demo_features], dtype=np.int64)
    demo_uids = np.array([d.get("demo_uid", "") for d in demo_features], dtype=object)
    native_id_key = np.array([d.get("native_episode_id_key", None) for d in demo_features], dtype=object)
    native_id_val = np.array([d.get("native_episode_id", None) for d in demo_features], dtype=object)

    actions = np.array([d.get("actions", None) for d in demo_features], dtype=object)
    action_jerk = np.array([d.get("action_jerk", 0.0) for d in demo_features], dtype=np.float32)
    action_smoothness = np.array([d.get("action_smoothness", 0.0) for d in demo_features], dtype=np.float32)
    action_energy = np.array([d.get("action_energy", 0.0) for d in demo_features], dtype=np.float32)
    action_small_ratio = np.array([d.get("action_small_ratio", 0.0) for d in demo_features], dtype=np.float32)

    save_name = f"{dataset_name}_{os.path.basename(dataset_path)}_{model_name}_nf{num_frames}_bs{batch_size}.npz"
    save_path = os.path.join(save_dir, save_name)

    np.savez_compressed(
        save_path,
        features=features_array,
        task_ids=task_ids,
        task_lengths=task_lengths,
        demo_lengths=demo_lengths,
        task_descriptions=task_descriptions,
        task_embeddings=task_embeddings,
        sample_features=sample_features,
        sample_actions=sample_actions,
        episode_indices=episode_indices,
        demo_uids=demo_uids,
        native_episode_id_key=native_id_key,
        native_episode_id=native_id_val,
        actions=actions,
        action_jerk=action_jerk,
        action_smoothness=action_smoothness,
        action_energy=action_energy,
        action_small_ratio=action_small_ratio,
    )

    print(f"saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Extract per-demo features from an RLDS/TFDS LIBERO dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the prepared TFDS dataset, e.g. /path/to/libero_10_no_noops",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rlds-libero-10",
        help="Dataset adapter name: rlds-libero-10, rlds-libero-goal, rlds-libero-object, or rlds-libero-spatial",
    )
    parser.add_argument("--save_dir", type=str, default="./feature/openvla")
    parser.add_argument("--model_name", type=str, default="openvla")
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_one(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        save_dir=args.save_dir,
        model_name=args.model_name,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
    )
