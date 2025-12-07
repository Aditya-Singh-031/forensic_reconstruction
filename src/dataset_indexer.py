"""
FAST Dataset Indexer using MULTIPROCESSING.
Creates comprehensive feature index with bbox metadata for training.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_MAPPING = {
    "left_eye": "eyes_left", "right_eye": "eyes_right",
    "left_eyebrow": "eyebrows_left", "right_eyebrow": "eyebrows_right",
    "nose_tip": "nose", "nose_base": "nose",
    "mouth_outer": "mouth_outer", "mouth_inner": "mouth_inner",
    "left_ear": "ears_left", "right_ear": "ears_right",
    "face_contour": "face_contour", "jawline": "jawline",
}

VALID_FEATURE_TYPES = [
    "eyes_left", "eyes_right", "eyebrows_left", "eyebrows_right",
    "nose", "mouth_outer", "mouth_inner", "ears_left", "ears_right",
    "face_contour", "jawline",
]

CORRUPTIBLE_FEATURES = [
    "eyes_left", "eyes_right", "eyebrows_left", "eyebrows_right",
    "nose", "mouth_outer", "mouth_inner",
]

# ---------------------------------------------------------
# WORKER FUNCTION (Your Logic ported to Parallel)
# ---------------------------------------------------------
def compute_nose_bbox(nose_img_path: Path, groups: Dict) -> Dict:
    """Compute correct bbox for nose from expanded crop."""
    try:
        with Image.open(nose_img_path) as nose_img:
            crop_width, crop_height = nose_img.size
        
        if "nose_tip" not in groups or "nose_base" not in groups: return None
        
        nt = groups["nose_tip"]
        nb = groups["nose_base"]
        
        if "landmarks_pixel" not in nt or "landmarks_pixel" not in nb: return None
        
        nt_lms = np.array(nt["landmarks_pixel"])
        nb_lms = np.array(nb["landmarks_pixel"])
        
        # Triangle bounds (matching your logic)
        top = nt_lms[np.argmin(nt_lms[:, 1])]
        left = nb_lms[np.argmin(nb_lms[:, 0])]
        right = nb_lms[np.argmax(nb_lms[:, 0])]
        
        tri = np.array([top, left, right])
        x_min_tri = int(np.min(tri[:, 0]))
        y_min_tri = int(np.min(tri[:, 1]))
        
        # Expanded crop margins
        x_min = max(0, x_min_tri - 180)
        y_min = max(0, y_min_tri - 140)
        
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + crop_width,
            "y_max": y_min + crop_height,
        }
    except Exception: return None

def process_single_image(args):
    lm_path, features_dir = args
    
    try:
        image_name = lm_path.stem.replace("_landmarks", "")
        with open(lm_path) as f: data = json.load(f)
        
        img_size = data.get("image_size", {})
        groups = data.get("groups", {})
        
        entry = {
            "image_name": image_name,
            "image_size": img_size,
            "features": {},
            "bboxes": {},
        }
        
        processed_nose = False
        
        for group_name, group_data in groups.items():
            ft = FEATURE_MAPPING.get(group_name)
            if ft not in VALID_FEATURE_TYPES: continue
            
            # De-duplicate nose
            if ft == "nose":
                if processed_nose: continue
                processed_nose = True
            
            f_path = features_dir / ft / f"{image_name}_{ft}.png"
            if not f_path.exists(): continue
            
            # Compute BBox
            bbox = None
            if ft == "nose":
                bbox = compute_nose_bbox(f_path, groups)
            else:
                if "bbox" in group_data and group_data["bbox"]:
                    b = group_data["bbox"]
                    bbox = {
                        "x_min": int(b["x_min"]), "y_min": int(b["y_min"]),
                        "x_max": int(b["x_max"]), "y_max": int(b["y_max"])
                    }
            
            entry["features"][ft] = str(f_path)
            if bbox: entry["bboxes"][ft] = bbox

        # Validate (Must have Contour + Jawline)
        if "face_contour" in entry["features"] and "jawline" in entry["features"]:
            return (image_name, entry)
            
    except Exception: pass
    return None

# ---------------------------------------------------------
# MAIN INDEXER CLASS
# ---------------------------------------------------------
class DatasetIndexer:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.landmarks_dir = self.features_dir / "landmarks"
        
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info("BUILDING FEATURE INDEX (Multiprocessing)")
        logger.info("=" * 60)
        
        if not self.landmarks_dir.exists():
            logger.error(f"Landmarks dir not found: {self.landmarks_dir}")
            return

        landmark_files = sorted(self.landmarks_dir.glob("*_landmarks.json"))
        logger.info(f"Found {len(landmark_files)} landmark files")
        
        # Parallel Execution
        tasks = [(f, self.features_dir) for f in landmark_files]
        num_cores = max(1, int(cpu_count() * 0.9))
        
        index = {}
        logger.info(f"Starting pool with {num_cores} cores...")
        
        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(process_single_image, tasks), total=len(tasks)))
            
        for res in results:
            if res: index[res[0]] = res[1]
            
        logger.info(f"\nIndexed {len(index)} complete training samples")
        
        self.save_index(index)
        self.save_statistics(self.compute_statistics(index))
        self.save_splits(self.create_splits(index))
        
        logger.info("=" * 60)
        logger.info("✓ INDEXING COMPLETE")
        logger.info("=" * 60)

    def save_index(self, index, filename="feature_index.json"):
        with open(self.metadata_dir / filename, "w") as f: json.dump(index, f, indent=2)
        logger.info(f"✓ Saved index: {self.metadata_dir / filename}")

    def compute_statistics(self, index):
        stats = {
            "total_images": len(index),
            "feature_counts": defaultdict(int),
            "corruptible_feature_counts": defaultdict(int)
        }
        for entry in index.values():
            for ft in entry["features"]:
                stats["feature_counts"][ft] += 1
                if ft in CORRUPTIBLE_FEATURES:
                    stats["corruptible_feature_counts"][ft] += 1
        return dict(stats)

    def save_statistics(self, stats, filename="dataset_statistics.json"):
        with open(self.metadata_dir / filename, "w") as f: json.dump(stats, f, indent=2)
        logger.info(f"✓ Saved statistics: {self.metadata_dir / filename}")

    def create_splits(self, index, train=0.8, val=0.1, seed=42):
        names = list(index.keys())
        random.Random(seed).shuffle(names)
        n = len(names)
        nt, nv = int(n*train), int(n*val)
        return {
            "train": names[:nt],
            "val": names[nt:nt+nv],
            "test": names[nt+nv:]
        }

    def save_splits(self, splits, filename="splits.json"):
        with open(self.metadata_dir / filename, "w") as f: json.dump(splits, f, indent=2)
        logger.info(f"✓ Saved splits: {self.metadata_dir / filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/teaching/G14/forensic_reconstruction/dataset")
    args = parser.parse_args()
    DatasetIndexer(args.dataset_dir).run()

if __name__ == "__main__":
    main()