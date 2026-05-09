import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict

def decode_image_bytes(image_bytes):
    """把 tfrecord 里的 bytes 解码成 np.array RGB 图像"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Failed to decode image: {e}")
        return None

def parse_bcz(dataset_path, num_frames=3):
    """
    BC-Z 数据集解析器（边读边均匀抽帧）
    dataset_path: TFRecord 文件夹
    输出：生成器，每次 yield np.array [num_frames,H,W,3] 
    """
    # 收集 TFRecord 文件
    tfrecord_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if not f.startswith("._"):
                tfrecord_files.append(os.path.join(root, f))
    tfrecord_files = sorted(tfrecord_files)
    print(f"Found {len(tfrecord_files)} TFRecord files in {dataset_path}")

    # 遍历 TFRecord 文件
    for tf_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tf_file)
        episodes = defaultdict(list)  # episode_id -> list of image bytes

        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            episode_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            image_bytes_list = example.features.feature['present/image/encoded'].bytes_list.value
            if len(image_bytes_list) == 0:
                continue
            episodes[episode_id].extend(image_bytes_list)

        # 对每个 episode 直接均匀采样 num_frames
        for episode_id, image_bytes_list in episodes.items():
            T = len(image_bytes_list)
            if T < num_frames:
                continue

            # 均匀采样索引
            if num_frames == 1:
                idx = [0]
            elif num_frames == 2:
                idx = [0, T-1]
            else:
                idx = np.linspace(0, T-1, num_frames, dtype=int)

            frames = []
            for i in idx:
                img_np = decode_image_bytes(image_bytes_list[i])
                if img_np is not None:
                    frames.append(img_np)
            if len(frames) == num_frames:
                yield np.stack(frames, axis=0)  # [num_frames,H,W,3]
