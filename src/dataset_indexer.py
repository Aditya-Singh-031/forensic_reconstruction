"""
Dataset Indexer - Updated for your dataset path
Creates fast lookup database for all extracted features
Generates train/val/test splits
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetIndexer:
    """Index all extracted features for fast training data access."""
    
    def __init__(self, dataset_dir: str = "/home/teaching/G14/forensic_reconstruction/dataset"):
        """Initialize indexer.
        
        Args:
            dataset_dir: Root directory containing features/ and metadata/
        """
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.metadata_dir = self.dataset_dir / "metadata"
        
        # Feature types to index
        self.feature_types = [
            "eyes_left",
            "eyes_right", 
            "eyebrows_left",
            "eyebrows_right",
            "nose",
            "mouth_outer",
            "mouth_inner",
            "ears_left",
            "ears_right",
            "face_contour",  # Base canvas
            "jawline"        # Ground truth target
        ]
        
        logger.info(f"Dataset directory: {self.dataset_dir}")
        logger.info(f"Features directory: {self.features_dir}")
    
    def build_index(self) -> Dict:
        """Build complete feature index.
        
        Returns:
            index: {image_name: {feature_type: path, ...}, ...}
        """
        logger.info("\n" + "="*60)
        logger.info("BUILDING FEATURE INDEX")
        logger.info("="*60)
        
        index = {}
        
        # Get all face_contour images as base
        face_contour_dir = self.features_dir / "face_contour"
        if not face_contour_dir.exists():
            logger.error(f"Face contour directory not found: {face_contour_dir}")
            return {}
        
        # Find all images with face_contour
        face_contour_images = list(face_contour_dir.glob("*_face_contour.png"))
        logger.info(f"Found {len(face_contour_images)} face contour images")
        
        for face_contour_path in tqdm(face_contour_images, desc="Indexing features"):
            # Extract image name (e.g., "00001" from "00001_face_contour.png")
            image_name = face_contour_path.stem.replace("_face_contour", "")
            
            # Initialize entry
            index[image_name] = {
                "image_name": image_name,
                "features": {},
                "masks": {},
                "complete": True  # Will set to False if any feature missing
            }
            
            # Index all feature types
            for feature_type in self.feature_types:
                feature_dir = self.features_dir / feature_type
                
                # Look for feature image
                feature_pattern = f"{image_name}_{feature_type}.png"
                feature_path = feature_dir / feature_pattern
                
                # Look for mask
                mask_pattern = f"{image_name}_{feature_type}_mask.png"
                mask_path = feature_dir / mask_pattern
                
                if feature_path.exists():
                    index[image_name]["features"][feature_type] = str(feature_path)
                    
                    if mask_path.exists():
                        index[image_name]["masks"][feature_type] = str(mask_path)
                else:
                    # Mark as incomplete if required feature missing
                    if feature_type in ["face_contour", "jawline"]:
                        index[image_name]["complete"] = False
                        logger.warning(f"Missing required {feature_type} for {image_name}")
        
        # Filter to only complete entries
        complete_index = {k: v for k, v in index.items() if v["complete"]}
        incomplete_count = len(index) - len(complete_index)
        
        logger.info(f"\nIndexing complete:")
        logger.info(f"  Total images: {len(index)}")
        logger.info(f"  Complete: {len(complete_index)}")
        logger.info(f"  Incomplete: {incomplete_count}")
        
        return complete_index
    
    def compute_statistics(self, index: Dict) -> Dict:
        """Compute dataset statistics.
        
        Args:
            index: Feature index
            
        Returns:
            stats: Dataset statistics
        """
        logger.info("\nComputing statistics...")
        
        stats = {
            "total_images": len(index),
            "feature_counts": defaultdict(int),
            "complete_triplets": 0  # Images with face_contour + jawline
        }
        
        for image_name, data in index.items():
            for feature_type in data["features"].keys():
                stats["feature_counts"][feature_type] += 1
            
            # Check if has required pair
            if "face_contour" in data["features"] and "jawline" in data["features"]:
                stats["complete_triplets"] += 1
        
        logger.info(f"Statistics:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Complete triplets (contour+jawline): {stats['complete_triplets']}")
        logger.info(f"  Feature counts:")
        for feature_type, count in sorted(stats["feature_counts"].items()):
            logger.info(f"    {feature_type}: {count}")
        
        return dict(stats)
    
    def create_splits(self, index: Dict, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, List[str]]:
        """Create train/val/test splits.
        
        Args:
            index: Feature index
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            seed: Random seed for reproducibility
            
        Returns:
            splits: {"train": [...], "val": [...], "test": [...]}
        """
        logger.info("\nCreating splits...")
        
        # Get all image names
        image_names = list(index.keys())
        random.seed(seed)
        random.shuffle(image_names)
        
        # Calculate split sizes
        total = len(image_names)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Create splits
        splits = {
            "train": image_names[:train_size],
            "val": image_names[train_size:train_size + val_size],
            "test": image_names[train_size + val_size:]
        }
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(splits['train'])} ({len(splits['train'])/total*100:.1f}%)")
        logger.info(f"  Val:   {len(splits['val'])} ({len(splits['val'])/total*100:.1f}%)")
        logger.info(f"  Test:  {len(splits['test'])} ({len(splits['test'])/total*100:.1f}%)")
        
        return splits
    
    def save_index(self, index: Dict, filename: str = "feature_index.json"):
        """Save index to file.
        
        Args:
            index: Feature index
            filename: Output filename
        """
        output_path = self.metadata_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving index to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"✓ Saved index ({len(index)} images)")
    
    def save_statistics(self, stats: Dict, filename: str = "statistics.json"):
        """Save statistics to file.
        
        Args:
            stats: Statistics dictionary
            filename: Output filename
        """
        output_path = self.metadata_dir / filename
        
        logger.info(f"Saving statistics to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Saved statistics")
    
    def save_splits(self, splits: Dict, filename: str = "train_val_test_split.json"):
        """Save splits to file.
        
        Args:
            splits: Splits dictionary
            filename: Output filename
        """
        output_path = self.metadata_dir / filename
        
        logger.info(f"Saving splits to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"✓ Saved splits")
    
    def run(self):
        """Run complete indexing pipeline."""
        logger.info("\n" + "="*60)
        logger.info("DATASET INDEXING PIPELINE")
        logger.info("="*60)
        
        # Build index
        index = self.build_index()
        
        if len(index) == 0:
            logger.error("No complete samples found. Exiting.")
            return
        
        # Compute statistics
        stats = self.compute_statistics(index)
        
        # Create splits
        splits = self.create_splits(index)
        
        # Save all outputs
        self.save_index(index)
        self.save_statistics(stats)
        self.save_splits(splits)
        
        logger.info("\n" + "="*60)
        logger.info("INDEXING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Output files:")
        logger.info(f"  {self.metadata_dir / 'feature_index.json'}")
        logger.info(f"  {self.metadata_dir / 'statistics.json'}")
        logger.info(f"  {self.metadata_dir / 'train_val_test_split.json'}")
        logger.info("="*60 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index facial features dataset")
    parser.add_argument("--dataset-dir", type=str, 
                       default="/home/teaching/G14/forensic_reconstruction/dataset",
                       help="Dataset root directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    indexer = DatasetIndexer(dataset_dir=args.dataset_dir)
    indexer.run()


if __name__ == "__main__":
    main()