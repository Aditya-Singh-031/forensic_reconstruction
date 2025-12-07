"""
FAST Dataset Indexer with CORRECTED bbox for expanded nose crops.
Uses MULTIPROCESSING to speed up file I/O.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
    "eyes_left", "eyes_right", "eyebrows_left", "eyebrows_right",
    "nose", "mouth_outer", "mouth_inner", "ears_left", "ears_right",
    "face_contour", "jawline",
]

def compute_nose_bbox(nose_img_path: Path, groups: Dict) -> Dict:
    """
    Compute correct bbox for nose by recovering the crop origin.
    Logic ported for worker process.
    """
    try:
        if not nose_img_path.exists():
            return None

        # Load the extracted nose image to get actual dimensions
        with Image.open(nose_img_path) as nose_img:
            crop_width, crop_height = nose_img.size
        
        if "nose_tip" not in groups or "nose_base" not in groups:
            return None
        
        nose_tip_data = groups["nose_tip"]
        nose_base_data = groups["nose_base"]
        
        if ("landmarks_pixel" not in nose_tip_data or 
            "landmarks_pixel" not in nose_base_data):
            return None
        
        nose_tip_lms = np.array(nose_tip_data["landmarks_pixel"])
        nose_base_lms = np.array(nose_base_data["landmarks_pixel"])
        
        # Triangle key points (matching extraction logic)
        top_point = nose_tip_lms[np.argmin(nose_tip_lms[:, 1])]
        left_point = nose_base_lms[np.argmin(nose_base_lms[:, 0])]
        right_point = nose_base_lms[np.argmax(nose_base_lms[:, 0])]
        
        triangle_points = np.array([top_point, left_point, right_point])
        
        # Triangle bounds
        x_min_tri = int(np.min(triangle_points[:, 0]))
        y_min_tri = int(np.min(triangle_points[:, 1]))
        
        # Recover top-left using fixed margins from extraction
        x_min_crop = max(0, x_min_tri - 180)
        y_min_crop = max(0, y_min_tri - 140)
        
        # Bottom-right is crop top-left + crop dimensions
        x_max_crop = x_min_crop + crop_width
        y_max_crop = y_min_crop + crop_height
        
        return {
            "x_min": x_min_crop,
            "y_min": y_min_crop,
            "x_max": x_max_crop,
            "y_max": y_max_crop,
        }
    except Exception:
        return None

def process_single_image(args):
    """Worker function to process a single image entry."""
    lm_path, features_dir = args
    
    try:
        image_name = lm_path.stem.replace("_landmarks", "")
        
        with open(lm_path) as f:
            data = json.load(f)
        
        img_size = data.get("image_size", {})
        groups = data.get("groups", {})
        
        entry = {
            "image_name": image_name,
            "image_size": img_size,
            "features": {},
            "bboxes": {},
        }
        
        processed_nose = False
        
        for detector_group_name, group_data in groups.items():
            feature_type = FEATURE_MAPPING.get(detector_group_name)
            if feature_type not in VALID_FEATURE_TYPES:
                continue
            
            # Handle nose specially
            if feature_type == "nose":
                if processed_nose: continue
                processed_nose = True
            
            feature_img_path = features_dir / feature_type / f"{image_name}_{feature_type}.png"
            
            if not feature_img_path.exists():
                continue
            
            # Calculate BBox
            bbox_clean = None
            
            if feature_type == "nose":
                # Use special nose logic
                bbox_clean = compute_nose_bbox(feature_img_path, groups)
            else:
                # Use standard bbox from metadata
                if "bbox" in group_data and group_data["bbox"]:
                    bbox = group_data["bbox"]
                    bbox_clean = {
                        "x_min": int(bbox["x_min"]),
                        "y_min": int(bbox["y_min"]),
                        "x_max": int(bbox["x_max"]),
                        "y_max": int(bbox["y_max"]),
                    }
            
            if bbox_clean:
                entry["features"][feature_type] = str(feature_img_path)
                entry["bboxes"][feature_type] = bbox_clean

        # Validate completeness
        if "face_contour" in entry["features"] and "jawline" in entry["features"]:
            return (image_name, entry)
            
    except Exception:
        pass
    
    return None

class DatasetIndexerFast:
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
            logger.error("Landmarks dir not found")
            return

        landmark_files = sorted(self.landmarks_dir.glob("*_landmarks.json"))
        logger.info(f"Found {len(landmark_files)} landmark files")
        
        # Prepare arguments for workers
        tasks = [(f, self.features_dir) for f in landmark_files]
        
        # Run Pool
        num_cores = max(1, int(cpu_count() * 0.8))
        logger.info(f"Starting pool with {num_cores} cores...")
        
        index = {}
        with Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, tasks), 
                total=len(tasks),
                desc="Indexing"
            ))
            
            for res in results:
                if res:
                    index[res[0]] = res[1]

        self.save_index(index)
        self.save_stats(self.compute_stats(index))
        self.save_splits(self.create_splits(index))
        
        logger.info(f"\nIndex Complete: {len(index)} valid images.")

    def save_index(self, index, filename="feature_index.json"):
        with open(self.metadata_dir / filename, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved index to {filename}")

    def compute_stats(self, index):
        stats = {"total_images": len(index), "feature_counts": defaultdict(int)}
        for _, entry in index.items():
            for ft in entry["features"]: stats["feature_counts"][ft] += 1
        return dict(stats)

    def save_stats(self, stats, filename="feature_stats.json"):
        with open(self.metadata_dir / filename, "w") as f:
            json.dump(stats, f, indent=2)

    def create_splits(self, index, train=0.8, val=0.1, seed=42):
        names = list(index.keys())
        random.Random(seed).shuffle(names)
        n = len(names)
        nt = int(n * train)
        nv = int(n * val)
        return {
            "train": names[:nt],
            "val": names[nt:nt+nv],
            "test": names[nt+nv:]
        }

    def save_splits(self, splits, filename="splits.json"):
        with open(self.metadata_dir / filename, "w") as f:
            json.dump(splits, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/teaching/G14/forensic_reconstruction/dataset")
    args = parser.parse_args()
    DatasetIndexerFast(args.dataset_dir).run()

if __name__ == "__main__":
    main()