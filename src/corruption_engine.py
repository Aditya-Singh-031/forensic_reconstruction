"""
Corruption Engine for Forensic Face Reconstruction Training.

Generates corrupted faces by compositing randomly sampled features onto
face_contour base images. Supports multiple corruption levels.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CorruptionEngine:
    """Engine for generating corrupted faces by compositing random features."""

    # Feature types that can be corrupted (exclude face_contour and jawline)
    CORRUPTIBLE_FEATURES = [
        "eyes_left",
        "eyes_right",
        "eyebrows_left",
        "eyebrows_right",
        "nose",
        "mouth_outer",
        "mouth_inner",
        "ears_left",
        "ears_right",
    ]

    def __init__(
        self,
        feature_index_path: str | Path,
        dataset_root_path: str | Path,
    ) -> None:
        """
        Initialize the CorruptionEngine.

        Args:
            feature_index_path: Path to feature_index.json
            dataset_root_path: Root path of the dataset
        """
        self.feature_index_path = Path(feature_index_path)
        self.dataset_root = Path(dataset_root_path)
        self.features_dir = self.dataset_root / "features"

        # Load feature index
        logger.info("Loading feature index from %s", self.feature_index_path)
        with self.feature_index_path.open("r", encoding="utf-8") as f:
            self.feature_index: Dict[str, Dict] = json.load(f)

        # Build list of available image IDs
        self.available_ids = list(self.feature_index.keys())
        logger.info("Loaded %d images from feature index", len(self.available_ids))

        # Initialize random generator
        self.random_gen = random.Random(42)

    def load_feature_with_mask(
        self, feature_id: str, feature_type: str
    ) -> Optional[Image.Image]:
        """
        Load feature image and mask, composite into RGBA image.

        Args:
            feature_id: Image ID (e.g., "00001")
            feature_type: Feature type (e.g., "eyes_left")

        Returns:
            PIL Image in RGBA mode, or None if loading fails
        """
        if feature_id not in self.feature_index:
            logger.warning("Feature ID %s not found in index", feature_id)
            return None

        entry = self.feature_index[feature_id]

        # Get feature and mask paths
        features = entry.get("features", {})
        masks = entry.get("masks", {})

        feature_path = features.get(feature_type)
        mask_path = masks.get(feature_type)

        if not feature_path or not mask_path:
            logger.warning(
                "Missing feature or mask for %s/%s: feature=%s, mask=%s",
                feature_id,
                feature_type,
                feature_path is not None,
                mask_path is not None,
            )
            return None

        feature_path = Path(feature_path)
        mask_path = Path(mask_path)

        if not feature_path.exists() or not mask_path.exists():
            logger.warning(
                "Feature or mask file missing for %s/%s", feature_id, feature_type
            )
            return None

        try:
            # Load feature image (RGB)
            feature_img = Image.open(feature_path).convert("RGB")

            # Load mask (L)
            mask_img = Image.open(mask_path).convert("L")

            # Ensure same size
            if feature_img.size != mask_img.size:
                logger.warning(
                    "Size mismatch for %s/%s: feature=%s, mask=%s. Resizing mask.",
                    feature_id,
                    feature_type,
                    feature_img.size,
                    mask_img.size,
                )
                mask_img = mask_img.resize(feature_img.size, Image.Resampling.LANCZOS)

            # Create RGBA image
            rgba_img = Image.new("RGBA", feature_img.size)
            rgba_img.paste(feature_img, (0, 0))
            rgba_img.putalpha(mask_img)

            return rgba_img

        except Exception as e:
            logger.warning(
                "Failed to load feature %s/%s: %s", feature_id, feature_type, e
            )
            return None

    def sample_random_features(
        self, num_features: int
    ) -> List[Tuple[str, str]]:
        """
        Randomly sample feature types and IDs.

        Args:
            num_features: Number of features to sample

        Returns:
            List of (feature_type, feature_id) tuples
        """
        # Sample feature types (without replacement)
        sampled_types = self.random_gen.sample(
            self.CORRUPTIBLE_FEATURES, min(num_features, len(self.CORRUPTIBLE_FEATURES))
        )

        # For each type, sample a random feature_id
        results = []
        for feature_type in sampled_types:
            # Filter IDs that have this feature
            available_for_type = [
                img_id
                for img_id in self.available_ids
                if feature_type in self.feature_index[img_id].get("features", {})
                and feature_type in self.feature_index[img_id].get("masks", {})
            ]

            if not available_for_type:
                logger.warning("No available images for feature type %s", feature_type)
                continue

            random_id = self.random_gen.choice(available_for_type)
            results.append((feature_type, random_id))

        return results

    def composite_feature_on_base(
        self,
        base_image: Image.Image,
        feature_rgba: Image.Image,
        feature_id: str,
        feature_type: str,
    ) -> Image.Image:
        """
        Composite feature onto base image at correct position.

        Args:
            base_image: Base face_contour image (RGB)
            feature_rgba: Feature image with alpha channel (RGBA)
            feature_id: ID of the feature (for bbox lookup)
            feature_type: Type of feature

        Returns:
            Composite image (RGB)
        """
        # Convert base to RGBA for compositing
        base_rgba = base_image.convert("RGBA")

        # Get bbox from feature_index
        entry = self.feature_index.get(feature_id, {})
        bboxes = entry.get("bboxes", {})

        # Try to get bbox for this feature
        bbox = bboxes.get(feature_type)

        if bbox and isinstance(bbox, dict):
            # Use bbox position
            x_min = bbox.get("x_min", 0)
            y_min = bbox.get("y_min", 0)
            x_max = bbox.get("x_max", base_image.width)
            y_max = bbox.get("y_max", base_image.height)

            # Resize feature to bbox size
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            if bbox_width > 0 and bbox_height > 0:
                feature_resized = feature_rgba.resize(
                    (bbox_width, bbox_height), Image.Resampling.LANCZOS
                )
                # Paste at bbox position
                base_rgba.paste(feature_resized, (x_min, y_min), feature_resized)
            else:
                logger.warning(
                    "Invalid bbox for %s/%s: %s", feature_id, feature_type, bbox
                )
                # Fallback: center placement
                self._paste_centered(base_rgba, feature_rgba)
        else:
            # No bbox available - use center placement as fallback
            logger.debug(
                "No bbox for %s/%s, using center placement", feature_id, feature_type
            )
            self._paste_centered(base_rgba, feature_rgba)

        # Convert back to RGB
        return base_rgba.convert("RGB")

    def _paste_centered(
        self, base_rgba: Image.Image, feature_rgba: Image.Image
    ) -> None:
        """Paste feature centered on base image."""
        base_w, base_h = base_rgba.size
        feat_w, feat_h = feature_rgba.size

        # Center position
        x = (base_w - feat_w) // 2
        y = (base_h - feat_h) // 2

        base_rgba.paste(feature_rgba, (x, y), feature_rgba)

    def generate_corrupted_face(
        self, face_contour_id: str, corruption_level: int
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Generate corrupted face by compositing random features.

        Args:
            face_contour_id: ID of the face_contour image to use as base
            corruption_level: Corruption level (1, 2, or 3)

        Returns:
            Tuple of (corrupted_image, corruption_mask) or (None, None) on failure
        """
        if face_contour_id not in self.feature_index:
            logger.warning("Face contour ID %s not found", face_contour_id)
            return None, None

        entry = self.feature_index[face_contour_id]
        features = entry.get("features", {})

        # Load face_contour base image
        face_contour_path = features.get("face_contour")
        if not face_contour_path:
            logger.warning("No face_contour for ID %s", face_contour_id)
            return None, None

        face_contour_path = Path(face_contour_path)
        if not face_contour_path.exists():
            logger.warning("Face contour file missing: %s", face_contour_path)
            return None, None

        try:
            base_image = Image.open(face_contour_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load face_contour %s: %s", face_contour_id, e)
            return None, None

        # Determine number of features to corrupt
        if corruption_level == 1:
            num_features = self.random_gen.randint(2, 3)
        elif corruption_level == 2:
            num_features = self.random_gen.randint(4, 5)
        elif corruption_level == 3:
            num_features = self.random_gen.randint(6, 8)
        else:
            logger.warning("Invalid corruption level %d, using level 2", corruption_level)
            num_features = self.random_gen.randint(4, 5)

        # Sample random features
        sampled_features = self.sample_random_features(num_features)

        if not sampled_features:
            logger.warning("No features sampled for %s", face_contour_id)
            return None, None

        # Create corruption mask (same size as base, black initially)
        corruption_mask = Image.new("L", base_image.size, 0)
        corruption_mask_array = np.array(corruption_mask)

        # Start with base image
        corrupted_image = base_image.copy()

        # Composite each sampled feature
        for feature_type, sampled_id in sampled_features:
            # Load feature with mask
            feature_rgba = self.load_feature_with_mask(sampled_id, feature_type)

            if feature_rgba is None:
                logger.warning(
                    "Skipping feature %s/%s (failed to load)", sampled_id, feature_type
                )
                continue

            # Composite onto base
            corrupted_image = self.composite_feature_on_base(
                corrupted_image, feature_rgba, sampled_id, feature_type
            )

            # Update corruption mask
            # Get the mask from the feature
            masks = self.feature_index[sampled_id].get("masks", {})
            mask_path = masks.get(feature_type)

            if mask_path and Path(mask_path).exists():
                try:
                    feature_mask = Image.open(mask_path).convert("L")

                    # Get bbox for positioning
                    entry_sampled = self.feature_index.get(sampled_id, {})
                    bboxes = entry_sampled.get("bboxes", {})

                    bbox = bboxes.get(feature_type)
                    if bbox and isinstance(bbox, dict):
                        x_min = bbox.get("x_min", 0)
                        y_min = bbox.get("y_min", 0)
                        x_max = bbox.get("x_max", base_image.width)
                        y_max = bbox.get("y_max", base_image.height)

                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        if bbox_width > 0 and bbox_height > 0:
                            # Resize mask to bbox size
                            feature_mask_resized = feature_mask.resize(
                                (bbox_width, bbox_height), Image.Resampling.LANCZOS
                            )
                            mask_array = np.array(feature_mask_resized)

                            # Paste into corruption mask at bbox position
                            y_end = min(y_min + bbox_height, corruption_mask_array.shape[0])
                            x_end = min(x_min + bbox_width, corruption_mask_array.shape[1])

                            if y_min < corruption_mask_array.shape[0] and x_min < corruption_mask_array.shape[1]:
                                mask_crop = mask_array[
                                    : y_end - y_min, : x_end - x_min
                                ]
                                corruption_mask_array[y_min:y_end, x_min:x_end] = np.maximum(
                                    corruption_mask_array[y_min:y_end, x_min:x_end],
                                    mask_crop,
                                )
                    else:
                        # Fallback: center placement
                        base_w, base_h = base_image.size
                        feat_w, feat_h = feature_mask.size
                        x = (base_w - feat_w) // 2
                        y = (base_h - feat_h) // 2

                        if x >= 0 and y >= 0:
                            mask_array = np.array(feature_mask)
                            y_end = min(y + feat_h, corruption_mask_array.shape[0])
                            x_end = min(x + feat_w, corruption_mask_array.shape[1])

                            if y < corruption_mask_array.shape[0] and x < corruption_mask_array.shape[1]:
                                mask_crop = mask_array[
                                    : y_end - y, : x_end - x
                                ]
                                corruption_mask_array[y:y_end, x:x_end] = np.maximum(
                                    corruption_mask_array[y:y_end, x:x_end],
                                    mask_crop,
                                )

                except Exception as e:
                    logger.warning(
                        "Failed to update corruption mask for %s/%s: %s",
                        sampled_id,
                        feature_type,
                        e,
                    )

        # Convert corruption mask array back to Image
        corruption_mask = Image.fromarray(corruption_mask_array, mode="L")

        return corrupted_image, corruption_mask


def main() -> None:
    """Main execution for testing corruption engine."""
    parser = argparse.ArgumentParser(
        description="Test corruption engine with sample images"
    )
    parser.add_argument(
        "--feature-index",
        type=str,
        default="/DATA/facial_features_dataset/metadata/feature_index.json",
        help="Path to feature_index.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/DATA/facial_features_dataset",
        help="Root path of dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/corruption_test",
        help="Output directory for test samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of face_contour IDs to test",
    )

    args = parser.parse_args()

    # Initialize engine
    engine = CorruptionEngine(args.feature_index, args.dataset_root)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get random sample of face_contour IDs
    available_ids = engine.available_ids
    sample_ids = random.sample(available_ids, min(args.num_samples, len(available_ids)))

    logger.info("Testing corruption on %d face_contour IDs", len(sample_ids))

    generated_count = 0

    # Generate corrupted faces for each ID and corruption level
    for face_id in tqdm(sample_ids, desc="Generating corrupted faces"):
        for level in [1, 2, 3]:
            corrupted_img, corruption_mask = engine.generate_corrupted_face(
                face_id, level
            )

            if corrupted_img is None or corruption_mask is None:
                logger.warning(
                    "Failed to generate corrupted face for %s level %d", face_id, level
                )
                continue

            # Save corrupted image
            corrupted_path = output_dir / f"{face_id}_level{level}_corrupted.png"
            corrupted_img.save(corrupted_path, quality=95)

            # Save corruption mask
            mask_path = output_dir / f"{face_id}_level{level}_corruption_mask.png"
            corruption_mask.save(mask_path)

            generated_count += 1

    logger.info("Generated %d sample corrupted faces", generated_count)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()

