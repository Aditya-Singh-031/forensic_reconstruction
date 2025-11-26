"""
Dataset Indexer - UPDATED with Bounding Box Calculation
Creates fast lookup database for all extracted features
Calculates and stores BBOXES to fix corruption positioning
"""

import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
from collections import defaultdict

# Add src to path to import LandmarkDetector
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.landmark_detector import LandmarkDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetIndexer:
    """Index all extracted features and calculate their bounding boxes."""
    
    def __init__(self, dataset_dir: str = "/home/teaching/G14/forensic_reconstruction/dataset"):
        """Initialize indexer."""
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.metadata_dir = self.dataset_dir / "metadata"
        
        # Try to locate raw images for bbox recalculation
        # Check standard locations
        self.raw_images_dir = self.dataset_dir / "raw_images"
        if not self.raw_images_dir.exists():
            # Try parent directory structure
            self.raw_images_dir = self.dataset_dir.parent / "dataset" / "raw_images"
        
        self.has_raw_images = self.raw_images_dir.exists()
        
        if self.has_raw_images:
            logger.info(f"Raw images found at: {self.raw_images_dir}")
            logger.info("Initializing LandmarkDetector for bbox calculation...")
            self.detector = LandmarkDetector()
        else:
            logger.warning("Raw images NOT found. Bounding boxes cannot be calculated!")
        
        # Feature types to index
        self.feature_types = [
            "eyes_left", "eyes_right", 
            "eyebrows_left", "eyebrows_right",
            "nose",
            "mouth_outer", "mouth_inner",
            "ears_left", "ears_right",
            "face_contour", "jawline"
        ]
        
        logger.info(f"Features directory: {self.features_dir}")
    
    def _calculate_feature_bbox(self, groups: Dict, feature_type: str, img_w: int, img_h: int) -> List[int]:
        """
        Replicate exact logic from create_features_dataset.py to find crop coordinates.
        Returns [x_min, y_min, x_max, y_max]
        """
        # 1. Handle NOSE (Triangle Logic)
        if feature_type == "nose":
            if 'nose_tip' in groups and 'nose_base' in groups:
                nose_tip = groups['nose_tip']
                nose_base = groups['nose_base']
                
                if nose_tip['count'] > 0 and nose_base['count'] > 0:
                    nt_lms = np.array(nose_tip['landmarks_pixel'])
                    nb_lms = np.array(nose_base['landmarks_pixel'])
                    
                    # Logic from create_features_dataset.py lines 215-235
                    top_point = nt_lms[np.argmin(nt_lms[:, 1])]
                    left_point = nb_lms[np.argmin(nb_lms[:, 0])]
                    right_point = nb_lms[np.argmax(nb_lms[:, 0])]
                    
                    triangle_points = np.array([top_point, left_point, right_point])
                    xs, ys = triangle_points[:, 0], triangle_points[:, 1]
                    
                    x_min_tri, x_max_tri = int(np.min(xs)), int(np.max(xs))
                    y_min_tri, y_max_tri = int(np.min(ys)), int(np.max(ys))
                    
                    # Expand with margins (must match extractor!)
                    margin_x = 180
                    margin_y = 140
                    
                    x_min = max(0, x_min_tri - margin_x)
                    y_min = max(0, y_min_tri - margin_y)
                    x_max = min(img_w, x_max_tri + margin_x)
                    y_max = min(img_h, y_max_tri + margin_y)
                    
                    return [x_min, y_min, x_max, y_max]
            return None

        # 2. Map feature names
        detector_map = {
            "eyes_left": "left_eye", "eyes_right": "right_eye",
            "eyebrows_left": "left_eyebrow", "eyebrows_right": "right_eyebrow",
            "mouth_outer": "mouth_outer", "mouth_inner": "mouth_inner",
            "ears_left": "left_ear", "ears_right": "right_ear",
            "face_contour": "face_contour", "jawline": "jawline"
        }
        
        det_key = detector_map.get(feature_type)
        if det_key and det_key in groups:
            data = groups[det_key]
            if data['bbox']:
                # Standard bbox logic
                bbox = data['bbox']
                x_min = max(0, int(bbox['x_min']))
                y_min = max(0, int(bbox['y_min']))
                x_max = min(img_w, int(bbox['x_max']))
                y_max = min(img_h, int(bbox['y_max']))
                return [x_min, y_min, x_max, y_max]
        
        return None

    def build_index(self) -> Dict:
        """Build complete feature index with bboxes."""
        logger.info("\n" + "="*60)
        logger.info("BUILDING FEATURE INDEX (WITH BBOXES)")
        logger.info("="*60)
        
        index = {}
        
        # Get all face_contour images as base
        face_contour_dir = self.features_dir / "face_contour"
        if not face_contour_dir.exists():
            logger.error(f"Face contour directory not found: {face_contour_dir}")
            return {}
        
        face_contour_images = list(face_contour_dir.glob("*_face_contour.png"))
        logger.info(f"Found {len(face_contour_images)} face contour images")
        
        for face_contour_path in tqdm(face_contour_images, desc="Indexing features"):
            image_name = face_contour_path.stem.replace("_face_contour", "")
            
            index[image_name] = {
                "image_name": image_name,
                "features": {},
                "masks": {},
                "bboxes": {},  # NEW: Store coordinates here
                "complete": True
            }
            
            # --- BBOX CALCULATION START ---
            if self.has_raw_images:
                # Try to find raw image
                raw_path = None
                # Check ffhq png
                if (self.raw_images_dir / "ffhq" / f"{image_name}.png").exists():
                    raw_path = self.raw_images_dir / "ffhq" / f"{image_name}.png"
                # Check celeba jpg/png
                elif (self.raw_images_dir / "celeba_hq" / f"{image_name}.jpg").exists():
                    raw_path = self.raw_images_dir / "celeba_hq" / f"{image_name}.jpg"
                elif (self.raw_images_dir / "celeba_hq" / f"{image_name}.png").exists():
                    raw_path = self.raw_images_dir / "celeba_hq" / f"{image_name}.png"
                
                groups = {}
                img_w, img_h = 0, 0
                
                if raw_path:
                    try:
                        # Run detection to get landmarks
                        res = self.detector.detect(str(raw_path), return_visualization=False)
                        groups = res['groups']
                        img_w, img_h = res['image_size']
                    except Exception as e:
                        logger.warning(f"Detection failed for {image_name}: {e}")
            # --- BBOX CALCULATION END ---

            # Index all feature types
            for feature_type in self.feature_types:
                feature_dir = self.features_dir / feature_type
                feature_path = feature_dir / f"{image_name}_{feature_type}.png"
                mask_path = feature_dir / f"{image_name}_{feature_type}_mask.png"
                
                if feature_path.exists():
                    index[image_name]["features"][feature_type] = str(feature_path)
                    if mask_path.exists():
                        index[image_name]["masks"][feature_type] = str(mask_path)
                    
                    # Calculate and save bbox
                    if self.has_raw_images and groups:
                        bbox = self._calculate_feature_bbox(groups, feature_type, img_w, img_h)
                        if bbox:
                            index[image_name]["bboxes"][feature_type] = bbox
                else:
                    if feature_type in ["face_contour", "jawline"]:
                        index[image_name]["complete"] = False
        
        complete_index = {k: v for k, v in index.items() if v["complete"]}
        
        logger.info(f"\nIndexing complete:")
        logger.info(f"  Total images: {len(index)}")
        logger.info(f"  Complete entries: {len(complete_index)}")
        
        return complete_index
    
    # ... [Existing compute_statistics, create_splits, save_ methods remain unchanged] ...
    
    def compute_statistics(self, index: Dict) -> Dict:
        """Compute dataset statistics."""
        logger.info("\nComputing statistics...")
        stats = {
            "total_images": len(index),
            "feature_counts": defaultdict(int),
            "complete_triplets": 0
        }
        for image_name, data in index.items():
            for feature_type in data["features"].keys():
                stats["feature_counts"][feature_type] += 1
            if "face_contour" in data["features"] and "jawline" in data["features"]:
                stats["complete_triplets"] += 1
        return dict(stats)
    
    def create_splits(self, index: Dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        image_names = list(index.keys())
        random.seed(seed)
        random.shuffle(image_names)
        total = len(image_names)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        return {
            "train": image_names[:train_size],
            "val": image_names[train_size:train_size + val_size],
            "test": image_names[train_size + val_size:]
        }
    
    def save_index(self, index: Dict, filename: str = "feature_index.json"):
        output_path = self.metadata_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(index, f, indent=2)
        logger.info(f"✓ Saved index to {output_path}")

    def save_statistics(self, stats, filename="statistics.json"):
        with open(self.metadata_dir / filename, 'w') as f:
            json.dump(stats, f, indent=2)

    def save_splits(self, splits, filename="train_val_test_split.json"):
        with open(self.metadata_dir / filename, 'w') as f:
            json.dump(splits, f, indent=2)

    def run(self):
        index = self.build_index()
        if len(index) == 0: return
        self.save_index(index)
        self.save_statistics(self.compute_statistics(index))
        self.save_splits(self.create_splits(index))
        logger.info("Indexing Pipeline Complete.")

def main():
    indexer = DatasetIndexer()
    indexer.run()

if __name__ == "__main__":
    main()