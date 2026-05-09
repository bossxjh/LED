# dataeval/models.py
import torch
from transformers import AutoProcessor, AutoModel


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "clip":
        import clip
        model, processor = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, processor, device

    elif model_name == "dinov2":
        processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model.eval()
        return model, processor, device

    elif model_name == "siglip":
        processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(device)
        model.eval()
        return model, processor, device

    elif model_name == "x_clip":
        processor = AutoProcessor.from_pretrained(
            "microsoft/xclip-base-patch16"
        )
        model = AutoModel.from_pretrained(
            "microsoft/xclip-base-patch16"
        ).to(device)
        model.eval()
        return model, processor, device

    else:
        raise ValueError(f"Unsupported model: {model_name}")
