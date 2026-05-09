#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import shutil
import argparse
from typing import List, Set, Tuple, Optional

import numpy as np
import tensorflow as tf


# ----------------------------
# IO helpers
# ----------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_version_dir(name: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+$", name))


def find_tfds_version_dir(dataset_root: str) -> str:
    dataset_root = os.path.abspath(dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"src_dataset_path not found: {dataset_root}")

    subs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    vers = [d for d in subs if _is_version_dir(d)]
    if not vers:
        raise RuntimeError(f"Cannot find TFDS version directory under: {dataset_root}")

    vers.sort(key=lambda s: tuple(map(int, s.split("."))))
    return os.path.join(dataset_root, vers[-1])


def load_selected_episode_indices(filtered_npz: str) -> Set[int]:
    z = np.load(filtered_npz, allow_pickle=True)

    if "episode_indices" in z.files:
        ep = np.asarray(z["episode_indices"]).astype(np.int64)
        ep = ep[ep >= 0]
        if ep.size == 0:
            raise ValueError("episode_indices is empty after filtering negatives.")
        return set(int(x) for x in ep.tolist())

    if "demo_uids" in z.files:
        uids = [str(x) for x in z["demo_uids"].tolist()]
        out = set()
        for u in uids:
            m = re.search(r"ep=(\d+)", u)
            if m:
                out.add(int(m.group(1)))
        if out:
            return out

    raise KeyError("filtered npz must contain episode_indices or demo_uids with 'ep=xxxxxx'.")


# ----------------------------
# TFDS naming + shards
# ----------------------------
def infer_tfds_prepared_name(src_ver_dir: str) -> str:
    info_path = os.path.join(src_ver_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json not found in {src_ver_dir}")
    info = load_json(info_path)
    name = info.get("name", None)
    if not name:
        raise KeyError("dataset_info.json missing 'name'")
    return str(name)


def list_split_shards(src_ver_dir: str, tfds_name: str, split: str) -> List[str]:
    # matches: liber_o10-train.tfrecord-00000-of-00032
    pat = re.compile(
        rf"^{re.escape(tfds_name)}-{re.escape(split)}\.(tfrecord|tfrecords|array_record)-\d+-of-\d+$"
    )
    files = []
    for fn in os.listdir(src_ver_dir):
        if pat.match(fn):
            files.append(os.path.join(src_ver_dir, fn))
    files.sort()
    return files


def parse_shard_suffix(base: str) -> Tuple[int, int]:
    """
    base ends with '-00000-of-00032'
    returns (idx_width=5, num_shards=32)
    """
    m = re.search(r"-(\d+)-of-(\d+)$", base)
    if not m:
        raise ValueError(f"Cannot parse shard suffix: {base}")
    return len(m.group(1)), int(m.group(2))


def copy_metadata_files(src_ver_dir: str, dst_ver_dir: str) -> None:
    os.makedirs(dst_ver_dir, exist_ok=True)
    for fn in os.listdir(src_ver_dir):
        src = os.path.join(src_ver_dir, fn)
        dst = os.path.join(dst_ver_dir, fn)

        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            continue

        # skip shards; will rewrite
        if re.search(r"\.(tfrecord|tfrecords|array_record)-\d+-of-\d+$", fn):
            continue

        shutil.copy2(src, dst)


def rewrite_split(
    src_ver_dir: str,
    dst_ver_dir: str,
    tfds_name: str,
    split: str,
    keep_episode_indices: Set[int],
    max_examples_per_shard: int,
) -> Tuple[int, List[int], int, int]:
    """
    Returns:
      kept_count, shard_lengths(list[int]), num_shards_new, num_bytes_new
    """
    src_shards = list_split_shards(src_ver_dir, tfds_name=tfds_name, split=split)
    if not src_shards:
        examples = [x for x in os.listdir(src_ver_dir) if "tfrecord" in x or "array_record" in x]
        raise RuntimeError(
            f"No TFRecord shards found for split='{split}' under {src_ver_dir}\n"
            f"Expected pattern: {tfds_name}-{split}.tfrecord-00000-of-00032\n"
            f"Files containing 'tfrecord': {examples[:30]}"
        )

    idx_width, _ = parse_shard_suffix(os.path.basename(src_shards[0]))

    os.makedirs(dst_ver_dir, exist_ok=True)

    # keep file format consistent with your source (tfrecord)
    prefix = f"{tfds_name}-{split}.tfrecord"
    tmp_tail = "of-?????"

    kept = 0
    shard_lengths: List[int] = []
    shard_id = 0
    cur_in_shard = 0
    writer: Optional[tf.io.TFRecordWriter] = None

    def open_writer(sid: int) -> tf.io.TFRecordWriter:
        fn = f"{prefix}-{sid:0{idx_width}d}-{tmp_tail}"
        return tf.io.TFRecordWriter(os.path.join(dst_ver_dir, fn))

    episode_idx = 0  # enumeration within split

    for shard in src_shards:
        ds = tf.data.TFRecordDataset(shard, compression_type="")
        for raw in ds:
            if episode_idx in keep_episode_indices:
                if writer is None:
                    writer = open_writer(shard_id)
                    cur_in_shard = 0

                writer.write(raw.numpy())
                kept += 1
                cur_in_shard += 1

                if cur_in_shard >= max_examples_per_shard:
                    writer.close()
                    writer = None
                    shard_lengths.append(cur_in_shard)
                    shard_id += 1
                    cur_in_shard = 0
            episode_idx += 1

    if writer is not None:
        writer.close()
        shard_lengths.append(cur_in_shard)
        shard_id += 1

    num_shards_new = shard_id

    # rename temp shards to "-of-{num_shards_new}"
    tmp_files = sorted([
        os.path.join(dst_ver_dir, fn)
        for fn in os.listdir(dst_ver_dir)
        if fn.startswith(f"{prefix}-") and fn.endswith(tmp_tail)
    ])
    for i, p in enumerate(tmp_files):
        new_name = f"{prefix}-{i:0{idx_width}d}-of-{num_shards_new:0{idx_width}d}"
        os.replace(p, os.path.join(dst_ver_dir, new_name))

    # compute bytes
    num_bytes = 0
    for fn in os.listdir(dst_ver_dir):
        if fn.startswith(f"{prefix}-") and re.search(r"-of-\d+$", fn):
            num_bytes += os.path.getsize(os.path.join(dst_ver_dir, fn))

    return kept, shard_lengths, num_shards_new, num_bytes


def update_dataset_info_json(
    dst_ver_dir: str,
    split: str,
    shard_lengths: List[int],
    num_shards: int,
    num_bytes: int,
) -> None:
    """
    Your dataset_info.json uses:
      "splits": [ { "name": "train", "numBytes": "...", "shardLengths": ["12", ...] } ]
    TFDS proto allows fields:
      name, numShards, shardLengths, numBytes, statistics, filepathTemplate
    IMPORTANT: do NOT add unknown fields like numExamples.
    """
    info_path = os.path.join(dst_ver_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json not found at {info_path}")

    info = load_json(info_path)
    splits = info.get("splits", None)
    if not isinstance(splits, list):
        raise RuntimeError("dataset_info.json 'splits' is not a list; cannot safely patch.")

    found = False
    for s in splits:
        if isinstance(s, dict) and s.get("name") == split:
            s["numBytes"] = str(int(num_bytes))
            s["shardLengths"] = [str(int(x)) for x in shard_lengths]
            s["numShards"] = str(int(num_shards))  # proto supports it; safe
            found = True
            break

    if not found:
        # keep filepathTemplate if any existing split has it
        template = None
        for s in splits:
            if isinstance(s, dict) and "filepathTemplate" in s:
                template = s["filepathTemplate"]
                break
        if template is None:
            template = "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"

        splits.append({
            "name": split,
            "filepathTemplate": template,
            "numBytes": str(int(num_bytes)),
            "numShards": str(int(num_shards)),
            "shardLengths": [str(int(x)) for x in shard_lengths],
        })

    info["splits"] = splits
    save_json(info_path, info)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Filter OpenVLA TFDS RLDS dataset by episode indices and write new TFDS directory.")
    ap.add_argument("--src_dataset_path", type=str, required=True,
                    help=".../modified_libero_rlds/libero_10_no_noops")
    ap.add_argument("--dst_dataset_path", type=str, required=True,
                    help=".../modified_libero_rlds/libero_10_no_noops_r0_4_best")
    ap.add_argument("--filtered_npz", type=str, required=True,
                    help="filtered_r{ratio}.npz containing episode_indices or demo_uids")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_examples_per_shard", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()

    keep_eps = load_selected_episode_indices(args.filtered_npz)
    print(f"[INFO] keep episodes: {len(keep_eps)}")

    src_ver = find_tfds_version_dir(args.src_dataset_path)
    ver_name = os.path.basename(src_ver.rstrip("/"))
    dst_ver = os.path.join(os.path.abspath(args.dst_dataset_path), ver_name)

    tfds_name = infer_tfds_prepared_name(src_ver)
    print(f"[INFO] src_ver={src_ver}")
    print(f"[INFO] tfds_name={tfds_name}")
    print(f"[INFO] dst_ver={dst_ver}")

    # copy metadata excluding shards
    copy_metadata_files(src_ver, dst_ver)

    kept, shard_lengths, num_shards_new, num_bytes_new = rewrite_split(
        src_ver_dir=src_ver,
        dst_ver_dir=dst_ver,
        tfds_name=tfds_name,
        split=args.split,
        keep_episode_indices=keep_eps,
        max_examples_per_shard=int(args.max_examples_per_shard),
    )

    print(f"[OK] kept={kept}  new_shards={num_shards_new}  bytes={num_bytes_new}")
    print(f"[OK] shard_lengths(head)={shard_lengths[:10]}")

    update_dataset_info_json(
        dst_ver_dir=dst_ver,
        split=args.split,
        shard_lengths=shard_lengths,
        num_shards=num_shards_new,
        num_bytes=num_bytes_new,
    )

    print("\n[DONE] New dataset ready:")
    print(f"  {os.path.abspath(args.dst_dataset_path)}")
    print("TFDS load example:")
    print(f"  tfds.load('{os.path.basename(os.path.abspath(args.dst_dataset_path))}', data_dir='{os.path.dirname(os.path.abspath(args.dst_dataset_path))}', split='{args.split}', shuffle_files=False)")


if __name__ == "__main__":
    main()