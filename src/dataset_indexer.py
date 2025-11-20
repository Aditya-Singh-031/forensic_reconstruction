"""
Dataset Indexer for Forensic Face Reconstruction.

Scans the extracted facial feature dataset, validates feature availability,
loads landmark metadata, and produces:
    - feature_index.json
    - train/val/test splits (80/10/10)
    - dataset statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for a single facial feature."""

    name: str
    has_mask: bool = True
    description: Optional[str] = None


@dataclass
class DatasetPaths:
    """Container for dataset paths."""

    dataset_root: Path
    features_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    landmarks_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.features_dir = self.dataset_root / "features"
        self.metadata_dir = self.dataset_root / "metadata"
        self.landmarks_dir = self.features_dir / "landmarks"


class DatasetIndexer:
    """Dataset Indexer implementation."""

    FEATURE_CONFIGS: Tuple[FeatureConfig, ...] = (
        FeatureConfig("eyes_left"),
        FeatureConfig("eyes_right"),
        FeatureConfig("eyebrows_left"),
        FeatureConfig("eyebrows_right"),
        FeatureConfig("nose"),
        FeatureConfig("mouth_outer"),
        FeatureConfig("mouth_inner"),
        FeatureConfig("ears_left"),
        FeatureConfig("ears_right"),
        FeatureConfig("face_contour"),
        FeatureConfig("jawline", has_mask=False),
    )

    def __init__(
        self,
        dataset_root: Path,
        seed: int = 42,
    ) -> None:
        self.paths = DatasetPaths(dataset_root=dataset_root)
        self.seed = seed
        self.random_gen = random.Random(seed)
        self.sources_map: Dict[str, str] = {}
        self.feature_stats: Dict[str, Dict[str, int]] = {}
        self.missing_records: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Execute the dataset indexing pipeline."""
        self._validate_directories()
        self.sources_map = self._load_sources()
        image_names = self._discover_image_names()
        logger.info("Found %d candidate images", len(image_names))

        feature_index = self._build_feature_index(image_names)
        if not feature_index:
            raise RuntimeError("No valid images found with complete feature set.")

        splits = self._create_splits(list(feature_index.keys()))
        statistics = self._compute_statistics(feature_index, splits)

        self._save_outputs(feature_index, splits, statistics)
        self._print_summary(statistics)

    # ------------------------------------------------------------------ #
    # Validation & Discovery
    # ------------------------------------------------------------------ #
    def _validate_directories(self) -> None:
        """Ensure required directories exist."""
        if not self.paths.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.paths.dataset_root}")
        if not self.paths.features_dir.exists():
            raise FileNotFoundError(f"Features directory missing: {self.paths.features_dir}")
        if not self.paths.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory missing: {self.paths.metadata_dir}")
        if not self.paths.landmarks_dir.exists():
            raise FileNotFoundError(f"Landmarks directory missing: {self.paths.landmarks_dir}")

    def _discover_image_names(self) -> List[str]:
        """Discover unique image names using jawline feature filenames."""
        jawline_dir = self.paths.features_dir / "jawline"
        if not jawline_dir.exists():
            raise FileNotFoundError(f"Jawline directory missing: {jawline_dir}")

        image_names: List[str] = []
        for file_path in jawline_dir.glob("*_jawline.png"):
            base = file_path.name.replace("_jawline.png", "")
            if base:
                image_names.append(base)
        image_names.sort()
        return image_names

    # ------------------------------------------------------------------ #
    # Metadata Loading
    # ------------------------------------------------------------------ #
    def _load_sources(self) -> Dict[str, str]:
        """Load image source mapping from processing log."""
        processing_log = self.paths.metadata_dir / "processing_log.csv"
        sources: Dict[str, str] = {}
        if not processing_log.exists():
            logger.warning("Processing log not found: %s", processing_log)
            return sources

        try:
            with processing_log.open("r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    image_name = row.get("image_name")
                    source = row.get("source", "unknown")
                    if image_name:
                        sources[image_name] = source
        except Exception as exc:
            logger.warning("Failed to read processing log: %s", exc)
        return sources

    def _load_landmark_data(self, image_name: str) -> Tuple[Dict, Dict[str, Optional[Dict[str, int]]]]:
        """Load landmark JSON and extract bounding boxes."""
        landmarks_path = self.paths.landmarks_dir / f"{image_name}_landmarks.json"
        if not landmarks_path.exists():
            logger.warning("Missing landmarks JSON for %s", image_name)
            return {}, {}

        try:
            with landmarks_path.open("r", encoding="utf-8") as json_file:
                data = json.load(json_file)
        except Exception as exc:
            logger.warning("Failed to parse landmarks for %s: %s", image_name, exc)
            return {}, {}

        groups = data.get("groups", {})
        bbox_map: Dict[str, Optional[Dict[str, int]]] = {}
        if isinstance(groups, dict):
            for feature_name, group_data in groups.items():
                bbox_map[feature_name] = group_data.get("bbox") if isinstance(group_data, dict) else None
        return data, bbox_map

    # ------------------------------------------------------------------ #
    # Feature Index Construction
    # ------------------------------------------------------------------ #
    def _build_feature_index(self, image_names: List[str]) -> Dict[str, Dict]:
        """Build the feature index for all valid images."""
        feature_index: Dict[str, Dict] = {}
        self._init_feature_stats()

        for image_name in tqdm(image_names, desc="Indexing images"):
            entry = self._build_single_entry(image_name)
            if entry.get("status", {}).get("is_complete"):
                feature_index[image_name] = entry
            else:
                logger.warning(
                    "Skipping %s due to missing components: %s",
                    image_name,
                    ", ".join(entry.get("status", {}).get("missing", [])),
                )
        return feature_index

    def _build_single_entry(self, image_name: str) -> Dict:
        """Build index entry for a single image."""
        entry: Dict[str, object] = {
            "image_name": image_name,
            "source": self.sources_map.get(image_name, "unknown"),
            "features": {},
            "masks": {},
        }

        landmarks_data, bbox_map = self._load_landmark_data(image_name)
        entry["landmarks_json"] = str(
            self.paths.landmarks_dir / f"{image_name}_landmarks.json"
        )
        entry["bboxes"] = bbox_map

        missing_components: List[str] = []

        for config in self.FEATURE_CONFIGS:
            feature_path = self._feature_file(config.name, image_name)
            feature_exists = feature_path.exists()
            if feature_exists:
                entry["features"][config.name] = str(feature_path)
                self.feature_stats[config.name]["available"] += 1
            else:
                missing_components.append(config.name)
                self.feature_stats[config.name]["missing"] += 1

            if config.has_mask:
                mask_path = self._feature_mask_file(config.name, image_name)
                if mask_path.exists():
                    entry["masks"][config.name] = str(mask_path)
                    self.feature_stats[f"{config.name}_mask"]["available"] += 1
                else:
                    missing_components.append(f"{config.name}_mask")
                    self.feature_stats[f"{config.name}_mask"]["missing"] += 1

        entry["status"] = {
            "is_complete": len(missing_components) == 0,
            "missing": missing_components,
        }

        # Attach raw landmarks if available
        if landmarks_data:
            entry["landmarks_metadata"] = landmarks_data

        if missing_components:
            self.missing_records[image_name] = missing_components

        return entry

    def _init_feature_stats(self) -> None:
        """Initialize statistics tracking."""
        self.feature_stats = {}
        for config in self.FEATURE_CONFIGS:
            self.feature_stats[config.name] = {"available": 0, "missing": 0}
            if config.has_mask:
                self.feature_stats[f"{config.name}_mask"] = {"available": 0, "missing": 0}

    def _feature_file(self, feature_name: str, image_name: str) -> Path:
        """Get feature image path."""
        return self.paths.features_dir / feature_name / f"{image_name}_{feature_name}.png"

    def _feature_mask_file(self, feature_name: str, image_name: str) -> Path:
        """Get feature mask path."""
        return self.paths.features_dir / feature_name / f"{image_name}_{feature_name}_mask.png"

    # ------------------------------------------------------------------ #
    # Splits & Statistics
    # ------------------------------------------------------------------ #
    def _create_splits(self, image_names: List[str]) -> Dict[str, List[str]]:
        """Create train/val/test split with fixed seed."""
        shuffled = image_names[:]
        self.random_gen.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        # Ensure no empty splits if total < 10
        if not splits["val"] and splits["train"]:
            splits["val"].append(splits["train"].pop())
        if not splits["test"] and splits["train"]:
            splits["test"].append(splits["train"].pop())

        return splits

    def _compute_statistics(
        self,
        feature_index: Dict[str, Dict],
        splits: Dict[str, List[str]],
    ) -> Dict[str, object]:
        """Compute dataset statistics."""
        stats = {
            "total_images": len(feature_index),
            "split_counts": {split: len(names) for split, names in splits.items()},
            "feature_availability": self.feature_stats,
            "missing_records": self.missing_records,
        }
        return stats

    # ------------------------------------------------------------------ #
    # Output Handling
    # ------------------------------------------------------------------ #
    def _save_outputs(
        self,
        feature_index: Dict[str, Dict],
        splits: Dict[str, List[str]],
        statistics: Dict[str, object],
    ) -> None:
        """Persist all outputs to metadata directory."""
        outputs = {
            "feature_index.json": feature_index,
            "train_val_test_split.json": splits,
            "statistics.json": statistics,
        }

        for filename, content in outputs.items():
            path = self.paths.metadata_dir / filename
            try:
                with path.open("w", encoding="utf-8") as json_file:
                    json.dump(content, json_file, indent=2)
                logger.info("Saved %s", path)
            except Exception as exc:
                logger.error("Failed to save %s: %s", path, exc)

    def _print_summary(self, statistics: Dict[str, object]) -> None:
        """Print summary statistics."""
        logger.info("=" * 60)
        logger.info("DATASET INDEX SUMMARY")
        logger.info("=" * 60)
        logger.info("Total images indexed: %s", statistics.get("total_images", 0))
        for split, count in statistics.get("split_counts", {}).items():
            logger.info("  %s: %s", split.capitalize(), count)
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Dataset Indexer")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/DATA/facial_features_dataset"),
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    indexer = DatasetIndexer(dataset_root=args.dataset, seed=args.seed)
    indexer.run()


if __name__ == "__main__":
    main()

