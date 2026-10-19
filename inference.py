from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny, maxvit_t

ROOT = Path(__file__).resolve().parent
CLASSES = ["healthy", "mosaic", "rust", "pest_attack"]
CLASS_MAPPING = {
    label: index for index, label in enumerate(CLASSES)
}
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
MODELS = {}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(mode):
    if mode not in ("leaf", "drone"):
        raise ValueError("Mode must be 'leaf' or 'drone'.")

    if mode not in MODELS:
        checkpoint_path = (
            ROOT / "models" / f"{mode}_best_model.pth"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["class_to_index"] != CLASS_MAPPING:
            raise ValueError("Checkpoint class mapping does not match.")

        if mode == "leaf":
            model = convnext_tiny(weights=None)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, 4
            )
        else:
            model = maxvit_t(weights=None)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features,
                4,
                bias=False
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        MODELS[mode] = model.eval()

    return MODELS[mode]


def predict_crop(image_path, mode="leaf"):
    mode = mode.strip().lower()
    if mode not in ("leaf", "drone"):
        raise ValueError("Choose mode='leaf' or mode='drone'.")

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    size = 384 if mode == "leaf" else 224

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.pad(
            image,
            (size, size),
            method=Image.Resampling.BICUBIC,
            color=(0, 0, 0),
            centering=(0.5, 0.5)
        )

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    model = load_model(mode).to(DEVICE)

    try:
        with torch.inference_mode():
            if DEVICE.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    logits = model(tensor)
            else:
                logits = model(tensor)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite prediction.")

        scores = (
            torch.softmax(logits.float(), dim=1)[0]
            .cpu()
            .tolist()
        )
    finally:
        model.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    index = max(
        range(len(scores)),
        key=scores.__getitem__
    )

    return {
        "image": str(image_path),
        "mode": mode,
        "model": (
            "ConvNeXt-Tiny" if mode == "leaf"
            else "MaxViT-Tiny"
        ),
        "predicted_label": CLASSES[index],
        "softmax_score": scores[index],
        "class_scores": dict(zip(CLASSES, scores))
    }
