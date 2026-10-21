"""Individual soybean predictions and experimental equal-weight late fusion."""

import math
import threading
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

MODEL_NAMES = {
    "leaf": "ConvNeXt-Tiny",
    "drone": "MaxViT-Tiny"
}

IMAGE_SIZES = {
    "leaf": 384,
    "drone": 224
}

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

MODELS = {}
MODEL_LOCK = threading.RLock()

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(mode):
    """Load and cache a saved model on CPU."""
    if mode not in MODEL_NAMES:
        raise ValueError("Mode must be 'leaf' or 'drone'.")

    with MODEL_LOCK:
        if mode in MODELS:
            return MODELS[mode]

        checkpoint_path = (
            ROOT / "models" / f"{mode}_best_model.pth"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint missing: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint.get("class_to_index") != CLASS_MAPPING:
            raise ValueError(
                f"Checkpoint class mapping does not match for {mode}."
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Model weights missing from checkpoint: {checkpoint_path}"
            )

        if mode == "leaf":
            model = convnext_tiny(weights=None)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features,
                len(CLASSES)
            )
        else:
            model = maxvit_t(weights=None)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features,
                len(CLASSES),
                bias=False
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        model.requires_grad_(False)
        MODELS[mode] = model.eval()

        return MODELS[mode]


def prepare_image(image_path, mode):
    """Apply the preprocessing used for the selected branch."""
    if mode not in IMAGE_SIZES:
        raise ValueError("Mode must be 'leaf' or 'drone'.")

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_size = IMAGE_SIZES[mode]

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.pad(
            image,
            (image_size, image_size),
            method=Image.Resampling.BICUBIC,
            color=(0, 0, 0),
            centering=(0.5, 0.5)
        )

    return TRANSFORM(image).unsqueeze(0)


def predict_crop(image_path, mode="leaf"):
    """Predict one image with the selected leaf or drone model."""
    if not isinstance(mode, str):
        raise ValueError("Mode must be 'leaf' or 'drone'.")

    mode = mode.strip().lower()

    if mode not in MODEL_NAMES:
        raise ValueError("Choose mode='leaf' or mode='drone'.")

    image_path = Path(image_path)
    image_tensor = prepare_image(image_path, mode)

    with MODEL_LOCK:
        model = load_model(mode)

        try:
            model.to(DEVICE)
            image_tensor = image_tensor.to(DEVICE)

            with torch.inference_mode():
                if DEVICE.type == "cuda":
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    ):
                        logits = model(image_tensor)
                else:
                    logits = model(image_tensor)

            if logits.shape != (1, len(CLASSES)):
                raise RuntimeError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite model prediction.")

            scores = (
                torch.softmax(logits.float(), dim=1)[0]
                .cpu()
                .tolist()
            )

        finally:
            model.cpu()

            if DEVICE.type == "cuda":
                del image_tensor
                torch.cuda.empty_cache()

    predicted_index = max(
        range(len(scores)),
        key=scores.__getitem__
    )

    return {
        "image": str(image_path),
        "mode": mode,
        "model": MODEL_NAMES[mode],
        "predicted_label": CLASSES[predicted_index],
        "softmax_score": float(scores[predicted_index]),
        "class_scores": dict(zip(CLASSES, scores))
    }


def validate_class_scores(class_scores):
    """Check that scores form a finite four-class distribution."""
    if set(class_scores) != set(CLASSES):
        raise ValueError("Class scores do not match the expected labels.")

    values = [float(class_scores[label]) for label in CLASSES]

    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("Class scores must be finite and between 0 and 1.")

    if not math.isclose(
        sum(values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5
    ):
        raise ValueError("Class scores do not sum to approximately one.")


def predict_fusion(leaf_path, drone_path):
    """Combine branch scores; input correspondence is not verified here."""
    with MODEL_LOCK:
        leaf_result = predict_crop(leaf_path, mode="leaf")
        drone_result = predict_crop(drone_path, mode="drone")

    leaf_scores = leaf_result["class_scores"]
    drone_scores = drone_result["class_scores"]

    validate_class_scores(leaf_scores)
    validate_class_scores(drone_scores)

    combined_scores = {
        label: float(
            0.5 * leaf_scores[label]
            + 0.5 * drone_scores[label]
        )
        for label in CLASSES
    }

    validate_class_scores(combined_scores)

    highest_score = max(combined_scores.values())

    top_labels = [
        label
        for label in CLASSES
        if math.isclose(
            combined_scores[label],
            highest_score,
            rel_tol=0.0,
            abs_tol=1e-6
        )
    ]

    tied = len(top_labels) > 1

    return {
        "method": "Equal-weight late fusion",
        "leaf_weight": 0.5,
        "drone_weight": 0.5,
        "leaf_model": MODEL_NAMES["leaf"],
        "drone_model": MODEL_NAMES["drone"],
        "leaf_prediction": leaf_result["predicted_label"],
        "drone_prediction": drone_result["predicted_label"],
        "branches_agree": (
            leaf_result["predicted_label"]
            == drone_result["predicted_label"]
        ),
        "fused_prediction": None if tied else top_labels[0],
        "top_score_tied": tied,
        "top_labels": top_labels,
        "highest_combined_score": float(highest_score),
        "leaf_scores": leaf_scores,
        "drone_scores": drone_scores,
        "combined_scores": combined_scores,
        "input_relationship": "Not verified by inference code",
        "evaluation_status": "Experimental; fusion accuracy not evaluated"
    }