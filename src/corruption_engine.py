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
from typing import Any, Dict, List, Optional, Tuple

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
    
    # Mapping from landmark detector names to our feature names
    LANDMARK_TO_FEATURE_MAP = {
        "left_eye": "eyes_left",
        "right_eye": "eyes_right",
        "left_eyebrow": "eyebrows_left",
        "right_eyebrow": "eyebrows_right",
        "nose_tip": "nose",
        "nose_base": "nose",
        "mouth_outer": "mouth_outer",
        "mouth_inner": "mouth_inner",
        "left_ear": "ears_left",
        "right_ear": "ears_right",
    }

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
            self.feature_index: Dict[str, Dict[str, Any]] = json.load(f)

        # Build list of available image IDs
        self.available_ids = list(self.feature_index.keys())
        logger.info("Loaded %d images from feature index", len(self.available_ids))

        # Initialize random generator
        self.random_gen = random.Random(42)
        
        # Cache for landmark detections (bboxes per image)
        self.bbox_cache: Dict[str, Dict[str, Dict[str, int]]] = {}

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
        results: List[Tuple[str, str]] = []
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
        target_bbox: Dict[str, int],
    ) -> Image.Image:
        """
        Composite feature onto base image at target bbox position.

        Args:
            base_image: Base face_contour image (RGB)
            feature_rgba: Feature image with alpha channel (RGBA) - will be resized to target bbox
            target_bbox: Target bbox dict with keys: x_min, y_min, x_max, y_max

        Returns:
            Composite image (RGB)
        """
        # Convert base to RGBA for compositing
        base_rgba = base_image.convert("RGBA")

        # Extract bbox coordinates
        x_min = target_bbox.get("x_min", 0)
        y_min = target_bbox.get("y_min", 0)
        x_max = target_bbox.get("x_max", base_image.width)
        y_max = target_bbox.get("y_max", base_image.height)

        # Calculate target size
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width > 0 and bbox_height > 0:
            # Resize sampled feature to match target bbox size
            feature_resized = feature_rgba.resize(
                (bbox_width, bbox_height), Image.Resampling.LANCZOS
            )
            # Paste at target bbox position
            base_rgba.paste(feature_resized, (x_min, y_min), feature_resized)
        else:
            logger.warning(
                "Invalid target bbox: x_min=%d, y_min=%d, x_max=%d, y_max=%d",
                x_min, y_min, x_max, y_max
            )
            # Fallback: center placement
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

        # Get TARGET bboxes from face_contour_id's feature_index
        # First try to get from stored bboxes (if available)
        raw_bboxes = entry.get("bboxes", {})
        target_bboxes = {}
        
        # Map landmark detector names to our feature names
        for landmark_name, bbox in raw_bboxes.items():
            bbox_dict = {}
            if isinstance(bbox, list) and len(bbox) >= 4:
                bbox_dict = {
                    "x_min": bbox[0],
                    "y_min": bbox[1],
                    "x_max": bbox[2],
                    "y_max": bbox[3],
                }
            elif isinstance(bbox, dict):
                bbox_dict = bbox
            else:
                continue

            mapped_name = self.LANDMARK_TO_FEATURE_MAP.get(landmark_name)
            if mapped_name:
                target_bboxes[mapped_name] = bbox_dict
            # Also handle direct matches (in case some are already mapped)
            elif landmark_name in self.CORRUPTIBLE_FEATURES:
                target_bboxes[landmark_name] = bbox_dict
        
        # If bboxes not available, detect from jawline (ground truth) image
        if not target_bboxes:
            # Check cache first
            if face_contour_id in self.bbox_cache:
                target_bboxes = self.bbox_cache[face_contour_id]
            else:
                features = entry.get("features", {})
                jawline_path = features.get("jawline")
                if jawline_path and Path(jawline_path).exists():
                    try:
                        # Use landmark detector to get bboxes from original image
                        # Import directly to avoid package __init__ issues
                        import importlib.util
                        from pathlib import Path as PathLib
                        landmark_detector_path = PathLib(__file__).parent / "landmark_detector.py"
                        spec = importlib.util.spec_from_file_location("landmark_detector", landmark_detector_path)
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Could not load landmark_detector from {landmark_detector_path}")
                        landmark_detector_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(landmark_detector_module)
                        LandmarkDetector = landmark_detector_module.LandmarkDetector
                        detector = LandmarkDetector()
                        result = detector.detect(str(jawline_path), return_groups=True, return_visualization=False)
                        groups = result.get("groups", {})
                        
                        # Map landmark detector groups to our feature names
                        detected_bboxes = {}
                        for landmark_name, group_data in groups.items():
                            mapped_name = self.LANDMARK_TO_FEATURE_MAP.get(landmark_name)
                            if mapped_name and mapped_name in self.CORRUPTIBLE_FEATURES:
                                bbox = group_data.get("bbox")
                                if bbox and isinstance(bbox, dict):
                                    detected_bboxes[mapped_name] = bbox
                        
                        # Cache the results
                        self.bbox_cache[face_contour_id] = detected_bboxes
                        target_bboxes = detected_bboxes
                    except Exception as e:
                        logger.warning("Failed to detect landmarks for %s: %s", face_contour_id, e)
                        # Cache empty dict to avoid repeated attempts
                        self.bbox_cache[face_contour_id] = {}

        # Start with base image
        corrupted_image = base_image.copy()

        for feature_type, sampled_id in sampled_features:
            # Get TARGET bbox from face_contour_id (where feature should be placed)
            target_bbox = target_bboxes.get(feature_type)
            
            if not target_bbox:
                logger.warning(
                    "No target bbox for %s/%s on face %s, skipping",
                    face_contour_id, feature_type, face_contour_id
                )
                continue

            # Load random feature with mask (from sampled_id)
            feature_rgba = self.load_feature_with_mask(sampled_id, feature_type)

            if feature_rgba is None:
                logger.warning(
                    "Skipping feature %s/%s (failed to load)", sampled_id, feature_type
                )
                continue

            # Composite onto base using TARGET bbox (resize sampled feature to target size)
            corrupted_image = self.composite_feature_on_base(
                corrupted_image, feature_rgba, target_bbox
            )

            # Update corruption mask using TARGET bbox
            x_min = target_bbox.get("x_min", 0)
            y_min = target_bbox.get("y_min", 0)
            x_max = target_bbox.get("x_max", base_image.width)
            y_max = target_bbox.get("y_max", base_image.height)

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            if bbox_width > 0 and bbox_height > 0:
                # Create a mask for the target bbox region (white = corrupted)
                # We use the full bbox area as corrupted
                y_end = min(y_min + bbox_height, corruption_mask_array.shape[0])
                x_end = min(x_min + bbox_width, corruption_mask_array.shape[1])

                if (y_min < corruption_mask_array.shape[0] and 
                    x_min < corruption_mask_array.shape[1] and
                    y_end > y_min and x_end > x_min):
                    # Mark entire bbox region as corrupted
                    corruption_mask_array[y_min:y_end, x_min:x_end] = 255

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
        default="dataset/metadata/feature_index.json",
        help="Path to feature_index.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
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
        # Load original face contour
        entry = engine.feature_index[face_id]
        features = entry.get("features", {})
        face_contour_path = features.get("face_contour")
        
        if not face_contour_path:
            continue
            
        try:
            original_img = Image.open(face_contour_path).convert("RGB")
        except Exception:
            logger.warning(f"Could not load original image for {face_id}")
            continue

        images = [original_img]
        
        for level in [1, 2, 3]:
            corrupted_img, corruption_mask = engine.generate_corrupted_face(
                face_id, level
            )

            if corrupted_img is None:
                # Create blank placeholder
                corrupted_img = Image.new("RGB", original_img.size, (0, 0, 0))
            
            images.append(corrupted_img)

        # Create side-by-side comparison
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        combined_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width
            
        # Save combined image
        combined_path = output_dir / f"{face_id}_comparison.png"
        combined_img.save(combined_path, quality=95)
        
        generated_count += 1

    logger.info("Generated %d sample corrupted faces", generated_count)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()

