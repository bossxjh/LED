from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from dataeval.paths import OPENVLA_CACHE_DIR

# ---------------- 配置 ----------------
# 权重缓存目录
cache_path = str(OPENVLA_CACHE_DIR)
# 图片路径
image_path = "example.jpg"

# ---------------- 加载 processor ----------------
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
    cache_dir=cache_path
)

# ---------------- 加载模型 ----------------
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="eager",
    torch_dtype=torch.float32,       # CPU 必须用 float32
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=cache_path
).to("cpu")  # Mac CPU

# ---------------- 加载图片 ----------------
image = Image.open(image_path).convert("RGB")

# ---------------- 处理图片（必须传 text 占位） ----------------
inputs = processor(
    images=image,
    text="",                # 占位，不生成文本
    return_tensors="pt"
).to("cpu")

pixel_values = inputs["pixel_values"]

# ---------------- 提取视觉特征 ----------------
with torch.no_grad():
    features = vla.get_image_features(pixel_values)

# ---------------- 输出 ----------------
print("特征 shape:", features.shape)
print("特征内容:", features)
