"""
Dataset Indexer with bbox metadata.

Builds feature_index.json of the form:
{
  "00001": {
    "image_name": "00001",
    "image_size": {"width": W, "height": H},
    "features": { "eyes_left": ".../eyes_left/00001_eyes_left.png", ... },
    "bboxes": {
      "eyes_left": {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...},
      ...
    }
  },
  ...
}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_MAPPING = {
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
    "face_contour": "face_contour",
    "jawline": "jawline",
}

VALID_FEATURE_TYPES = [
    "eyes_left",
    "eyes_right",
    "eyebrows_left",
    "eyebrows_right",
    "nose",
    "mouth_outer",
    "mouth_inner",
    "ears_left",
    "ears_right",
    "face_contour",
    "jawline",
]


class DatasetIndexer:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.landmarks_dir = self.features_dir / "landmarks"

        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset dir   : {self.dataset_dir}")
        logger.info(f"Features dir  : {self.features_dir}")
        logger.info(f"Landmarks dir : {self.landmarks_dir}")

    def _load_landmarks_json(self, path: Path) -> Dict:
        with open(path) as f:
            return json.load(f)

    def build_index(self) -> Dict[str, Dict]:
        """
        Build feature index by combining landmarks JSON with feature PNGs.
        """
        logger.info("\n" + "=" * 60)
        logger.info("BUILDING FEATURE INDEX (with bboxes)")
        logger.info("=" * 60)

        if not self.landmarks_dir.exists():
            logger.error(f"Landmarks dir not found: {self.landmarks_dir}")
            return {}

        index: Dict[str, Dict] = {}

        landmark_files = sorted(self.landmarks_dir.glob("*_landmarks.json"))
        logger.info(f"Found {len(landmark_files)} landmark JSON files")

        for lm_path in landmark_files:
            image_name = lm_path.stem.replace("_landmarks", "")
            data = self._load_landmarks_json(lm_path)

            img_size = data.get("image_size", {})
            groups = data.get("groups", {})

            entry = {
                "image_name": image_name,
                "image_size": img_size,
                "features": {},
                "bboxes": {},
            }

            # For each detector group, map to feature type
            for detector_group_name, group_data in groups.items():
                if "bbox" not in group_data or group_data["bbox"] is None:
                    continue
                if "landmarks_pixel" not in group_data or not group_data["landmarks_pixel"]:
                    continue

                feature_type = FEATURE_MAPPING.get(detector_group_name)
                if feature_type not in VALID_FEATURE_TYPES:
                    continue

                # Feature image path (same naming as extractor)
                feature_img_path = (
                    self.features_dir
                    / feature_type
                    / f"{image_name}_{feature_type}.png"
                )
                if not feature_img_path.exists():
                    # It might have failed extraction; skip
                    continue

                bbox = group_data["bbox"]
                # Normalize to ints
                bbox_clean = {
                    "x_min": int(bbox["x_min"]),
                    "y_min": int(bbox["y_min"]),
                    "x_max": int(bbox["x_max"]),
                    "y_max": int(bbox["y_max"]),
                }

                entry["features"][feature_type] = str(feature_img_path)
                entry["bboxes"][feature_type] = bbox_clean

            # Require at least face_contour and jawline
            if "face_contour" in entry["features"] and "jawline" in entry["features"]:
                index[image_name] = entry
            else:
                logger.debug(
                    f"Skipping {image_name}: missing face_contour or jawline"
                )

        logger.info(f"\nIndexed images: {len(index)}")
        return index

    def save_index(self, index: Dict[str, Dict], filename: str = "feature_index.json"):
        out_path = self.metadata_dir / filename
        with open(out_path, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved feature index to {out_path} ({len(index)} images)")

    def compute_stats(self, index: Dict[str, Dict]) -> Dict:
        stats = {
            "total_images": len(index),
            "feature_counts": defaultdict(int),
        }
        for _, entry in index.items():
            for ft in entry["features"].keys():
                stats["feature_counts"][ft] += 1
        return stats

    def save_stats(self, stats: Dict, filename: str = "feature_stats.json"):
        out_path = self.metadata_dir / filename
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved stats to {out_path}")

    def create_splits(
        self,
        index: Dict[str, Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        image_names = list(index.keys())
        random.Random(seed).shuffle(image_names)

        n = len(image_names)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": image_names[:n_train],
            "val": image_names[n_train : n_train + n_val],
            "test": image_names[n_train + n_val :],
        }
        return splits

    def save_splits(self, splits: Dict[str, List[str]], filename: str = "splits.json"):
        out_path = self.metadata_dir / filename
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Saved splits to {out_path}")

    def run(self):
        index = self.build_index()
        if not index:
            logger.error("Index is empty; aborting.")
            return
        self.save_index(index)
        stats = self.compute_stats(index)
        self.save_stats(stats)
        splits = self.create_splits(index)
        self.save_splits(splits)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build feature index with bboxes")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/teaching/G14/forensic_reconstruction/dataset",
    )
    args = parser.parse_args()

    indexer = DatasetIndexer(args.dataset_dir)
    indexer.run()


if __name__ == "__main__":
    main()
