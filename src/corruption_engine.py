"""
Corruption Engine using bbox metadata - FIXED coordinate handling
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see what's happening
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CORRUPTIBLE_FEATURES = [
    "eyes_left",
    "eyes_right",
    "eyebrows_left",
    "eyebrows_right",
    "nose",
    "mouth_outer",
    #"mouth_inner",
]


class CorruptionEngine:
    def __init__(self, feature_index: Dict[str, Dict], dataset_dir: str):
        self.feature_index = feature_index
        self.image_names = list(feature_index.keys())
        self.dataset_dir = Path(dataset_dir)

        logger.info(f"Initialized CorruptionEngine with {len(self.image_names)} images")
        logger.info(f"Dataset dir: {self.dataset_dir}")

    @staticmethod
    def _load_rgba(path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            return img
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _choose_features_for_level(self, level: int) -> List[str]:
        """
        Level 1: 2 random features
        Level 2: 4 random features
        Level 3: ALL 7 features
        """
        if level == 1:
            k = 2
            return random.sample(CORRUPTIBLE_FEATURES, k)
        elif level == 2:
            k = 4
            return random.sample(CORRUPTIBLE_FEATURES, k)
        elif level == 3:
            return list(CORRUPTIBLE_FEATURES)  # All 7
        else:
            return random.sample(CORRUPTIBLE_FEATURES, 4)

    def create_corrupted_face(
        self, image_name: str, level: int = 2
    ) -> Tuple[Image.Image, Image.Image, List[str]]:
        entry = self.feature_index[image_name]

        base_path = entry["features"]["face_contour"]
        target_path = entry["features"]["jawline"]

        base_img = self._load_rgba(base_path)
        target_img = self._load_rgba(target_path)

        if base_img is None or target_img is None:
            raise RuntimeError(f"Failed to load base/target for {image_name}")

        logger.debug(f"Base (face_contour) size: {base_img.size}")
        logger.debug(f"Target (jawline) size: {target_img.size}")

        corrupted = base_img.copy()
        features_to_corrupt = self._choose_features_for_level(level)
        actually_corrupted: List[str] = []

        for ft in features_to_corrupt:
            logger.debug(f"\n  Processing feature: {ft}")
            
            # Check if feature exists for this image
            if ft not in entry["bboxes"]:
                logger.debug(f"    ❌ No bbox for {ft}")
                continue
            if ft not in entry["features"]:
                logger.debug(f"    ❌ No feature path for {ft}")
                continue

            bbox = entry["bboxes"][ft]
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            x_max = int(bbox["x_max"])
            y_max = int(bbox["y_max"])

            logger.debug(f"    Bbox: ({x_min}, {y_min}, {x_max}, {y_max})")

            # Load current feature (for size/mask reference)
            current_path = entry["features"][ft]
            current_img = self._load_rgba(current_path)
            if current_img is None:
                logger.debug(f"    ❌ Failed to load current feature")
                continue

            w, h = current_img.size
            logger.debug(f"    Current feature size: {w}x{h}")

            # Pick random donor
            donor_image_name = random.choice(self.image_names)
            donor_entry = self.feature_index[donor_image_name]
            
            if ft not in donor_entry["features"]:
                logger.debug(f"    ❌ Donor {donor_image_name} missing {ft}")
                continue

            donor_path = donor_entry["features"][ft]
            donor_img = self._load_rgba(donor_path)
            if donor_img is None:
                logger.debug(f"    ❌ Failed to load donor feature")
                continue

            logger.debug(f"    Donor: {donor_image_name}, size: {donor_img.size}")

            # Resize donor to match current feature dimensions
            donor_resized = donor_img.resize((w, h), Image.Resampling.LANCZOS)

            # Get alpha mask from current feature
            mask = current_img.split()[3]
            
            # Check if mask is valid
            import numpy as np
            mask_array = np.array(mask)
            non_zero = np.count_nonzero(mask_array)
            logger.debug(f"    Mask non-zero pixels: {non_zero}/{mask_array.size}")

            # === KEY FIX: Paste feature crop at bbox position ===
            # The donor_resized is a small crop (w x h)
            # We paste it at (x_min, y_min) using its own alpha mask
            try:
                corrupted.paste(donor_resized, (x_min, y_min), mask)
                actually_corrupted.append(ft)
                logger.debug(f"    ✓ Pasted at ({x_min}, {y_min})")
            except Exception as e:
                logger.error(f"    ❌ Paste failed: {e}")

        logger.debug(f"\nTotal corrupted: {len(actually_corrupted)}")
        return corrupted, target_img, actually_corrupted

    def visualize(
        self,
        image_name: str,
        level: int,
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        entry = self.feature_index[image_name]
        base_img = self._load_rgba(entry["features"]["face_contour"])

        corrupted, target, feats = self.create_corrupted_face(image_name, level)

        # Composite onto white background
        w, h = base_img.size  # type: ignore
        white = Image.new("RGB", (w, h), (255, 255, 255))

        def on_white(img_rgba: Image.Image) -> Image.Image:
            canvas = white.copy()
            canvas.paste(img_rgba, (0, 0), img_rgba)
            return canvas

        left = on_white(base_img)        # face_contour
        mid = on_white(corrupted)       # corrupted
        right = on_white(target)        # jawline

        vis = Image.new("RGB", (w * 3, h), (255, 255, 255))
        vis.paste(left, (0, 0))
        vis.paste(mid, (w, 0))
        vis.paste(right, (2 * w, 0))

        if save_path is not None:
            vis.save(save_path)
            logger.info(f"Saved visualization to {save_path}")
            logger.info(f"  Corrupted features: {feats}")

        return vis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test corruption engine with bboxes")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/teaching/G14/forensic_reconstruction/dataset",
    )
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    index_path = dataset_dir / "metadata" / "feature_index.json"
    if not index_path.exists():
        logger.error(f"feature_index.json not found at {index_path}")
        return

    with open(index_path) as f:
        feature_index = json.load(f)

    engine = CorruptionEngine(feature_index, str(dataset_dir))

    if args.image is not None:
        images = [args.image]
    else:
        images = list(feature_index.keys())[:3]

    vis_dir = dataset_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    for image_name in images:
        logger.info("\n" + "=" * 60)
        logger.info(f"IMAGE: {image_name}")
        logger.info("=" * 60)
        for level in [1, 2, 3]:
            out_path = vis_dir / f"corruption_L{level}_{image_name}.png"
            engine.visualize(image_name, level, out_path)

    logger.info(f"\nDone. Check visualizations in {vis_dir}\n")


if __name__ == "__main__":
    main()
