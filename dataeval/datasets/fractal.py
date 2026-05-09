import os
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

def parse_fractal(tfrecord_folder, num_frames=3):
    """
    生成器，每次 yield np.array [num_frames, H, W, 3]
    适用于 fractal 数据集（TFRecord 格式）。
    
    Args:
        tfrecord_folder (str): 存放 TFRecord 文件的目录
        num_frames (int): 每个 sample 抽取的帧数，默认 3 (首/中/尾)
    """
    def decode_image_bytes(image_bytes):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            return np.array(img)
        except Exception as e:
            print(f"[WARN] Failed to decode image: {e}")
            return None

    # 找到所有 shard 文件
    tfrecord_files = sorted([
        os.path.join(tfrecord_folder, f)
        for f in os.listdir(tfrecord_folder)
        if ".tfrecord-" in f and not f.startswith("._")
    ])

    for tf_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tf_file)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            if 'steps/observation/image' not in example.features.feature:
                continue

            images_bytes_list = example.features.feature['steps/observation/image'].bytes_list.value
            total_frames = len(images_bytes_list)
            if total_frames == 0:
                continue

            indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            frames = []
            for idx in indices:
                img_bytes = images_bytes_list[idx]
                img = decode_image_bytes(img_bytes)
                if img is not None:
                    frames.append(img)

            if frames:
                yield np.stack(frames, axis=0)  # shape [num_frames, H, W, 3]
