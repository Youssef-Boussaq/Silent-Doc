"""
Silent Doctor — Skin Disease Detection Model
==============================================
MobileNetV2-based classifier for skin condition analysis.

Classes: acne, eczema, psoriasis, melanoma, normal_skin

Usage:
    model = SkinModel()
    result = model.predict("path/to/skin_image.jpg")
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Optional

from config.settings import IMAGE_SIZE, SKIN_CLASSES, SKIN_MODEL_PATH
from utils.helpers import load_image, resize_image, setup_logger

logger = setup_logger(__name__)


class SkinModel:
    """
    Skin disease classifier built on MobileNetV2.

    Architecture:
        MobileNetV2 (pretrained feature extractor)
        → Dropout(0.3)
        → Linear(1280 → num_classes)

    The model can be loaded from a .pth weights file or initialized
    with random weights for development/testing.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        classes: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.classes = classes or SKIN_CLASSES
        self.num_classes = len(self.classes)
        self.device = torch.device(device)
        self.model_path = Path(model_path) if model_path else SKIN_MODEL_PATH

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = self._build_model()
        self._load_weights()

    def _build_model(self) -> nn.Module:
        """Build MobileNetV2 with a custom classification head."""
        base_model = models.mobilenet_v2(weights=None)

        # Replace the classifier head
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, self.num_classes),
        )

        return base_model.to(self.device)

    def _load_weights(self):
        """Load pretrained weights if available."""
        if self.model_path.exists():
            logger.info(f"Loading skin model weights from {self.model_path}")
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            logger.info("✅ Skin model weights loaded successfully.")
        else:
            logger.warning(
                f"⚠️  No skin model weights found at {self.model_path}. "
                "Using uninitialized model (for development only)."
            )

        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL Image for inference."""
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Run skin disease prediction on an image.

        Args:
            image_path: Path to the skin image.

        Returns:
            dict with keys:
                - prediction (str): Predicted class name
                - confidence (float): Confidence score (0–1)
                - all_scores (dict): Scores for all classes
        """
        # Load and preprocess
        img = load_image(image_path)
        tensor = self.preprocess(img)

        # Forward pass
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        # Extract results
        top_idx = probabilities.argmax().item()
        top_class = self.classes[top_idx]
        top_confidence = probabilities[top_idx].item()

        all_scores = {
            cls: round(probabilities[i].item(), 4)
            for i, cls in enumerate(self.classes)
        }

        logger.info(
            f"🔬 Skin prediction: {top_class} "
            f"(confidence: {top_confidence:.2%})"
        )

        return {
            "prediction": top_class,
            "confidence": round(top_confidence, 4),
            "all_scores": all_scores,
        }


# ── Convenience function ────────────────────────────────────────────────

_cached_model: Optional[SkinModel] = None


def get_skin_model(**kwargs) -> SkinModel:
    """Get or create a cached SkinModel instance (singleton)."""
    global _cached_model
    if _cached_model is None:
        _cached_model = SkinModel(**kwargs)
    return _cached_model
