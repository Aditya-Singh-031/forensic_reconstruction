"""
Corruption Engine using bbox metadata.

Left  : face_contour (base with holes)
Middle: base + random donor features pasted at correct positions
Right : jawline (original full face)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
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
    "mouth_inner",
    "ears_left",
    "ears_right",
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
        if level == 1:
            k = random.randint(2, 3)
        elif level == 2:
            k = random.randint(4, 5)
        elif level == 3:
            k = random.randint(6, 8)
        else:
            k = 4
        k = min(k, len(CORRUPTIBLE_FEATURES))
        return random.sample(CORRUPTIBLE_FEATURES, k)

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

        corrupted = base_img.copy()
        features_to_corrupt = self._choose_features_for_level(level)
        actually_corrupted: List[str] = []

        for ft in features_to_corrupt:
            if ft not in entry["bboxes"]:
                continue
            if ft not in entry["features"]:
                continue

            bbox = entry["bboxes"][ft]
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            x_max = int(bbox["x_max"])
            y_max = int(bbox["y_max"])

            # Current image's feature (for mask/shape)
            current_path = entry["features"][ft]
            current_img = self._load_rgba(current_path)
            if current_img is None:
                continue

            w, h = current_img.size

            # Donor feature from random image
            donor_image_name = random.choice(self.image_names)
            donor_entry = self.feature_index[donor_image_name]
            if ft not in donor_entry["features"]:
                continue
            donor_path = donor_entry["features"][ft]
            donor_img = self._load_rgba(donor_path)
            if donor_img is None:
                continue

            donor_resized = donor_img.resize((w, h), Image.Resampling.LANCZOS)

            # Use current feature alpha as placement mask
            mask = current_img.split()[3]  # alpha channel

            # Paste at correct (x_min, y_min)
            corrupted.paste(donor_resized, (x_min, y_min), mask)
            actually_corrupted.append(ft)

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
        images = list(feature_index.keys())[:3]  # first 3 for sanity

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
