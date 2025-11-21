"""
Dataset Verification Script
Comprehensive checks for dataset integrity before training
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetVerifier:
    """Verify dataset integrity before training."""
    
    def __init__(self, dataset_dir: str = "/DATA/facial_features_dataset"):
        """Initialize verifier.
        
        Args:
            dataset_dir: Dataset root directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.metadata_dir = self.dataset_dir / "metadata"
        
        self.errors = []
        self.warnings = []
    
    def check_directory_structure(self) -> bool:
        """Check if required directories exist.
        
        Returns:
            success: True if all directories exist
        """
        logger.info("\n" + "="*60)
        logger.info("CHECKING DIRECTORY STRUCTURE")
        logger.info("="*60)
        
        required_dirs = [
            self.features_dir,
            self.metadata_dir,
            self.features_dir / "face_contour",
            self.features_dir / "jawline",
            self.features_dir / "nose",
            self.features_dir / "eyes_left",
            self.features_dir / "eyes_right"
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            if dir_path.exists():
                logger.info(f"  ✓ {dir_path.relative_to(self.dataset_dir)}")
            else:
                logger.error(f"  ✗ {dir_path.relative_to(self.dataset_dir)} MISSING")
                self.errors.append(f"Missing directory: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def check_metadata_files(self) -> bool:
        """Check if required metadata files exist.
        
        Returns:
            success: True if all files exist
        """
        logger.info("\n" + "="*60)
        logger.info("CHECKING METADATA FILES")
        logger.info("="*60)
        
        required_files = [
            "feature_index.json",
            "statistics.json",
            "train_val_test_split.json"
        ]
        
        all_exist = True
        for filename in required_files:
            file_path = self.metadata_dir / filename
            if file_path.exists():
                # Try to load JSON
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    logger.info(f"  ✓ {filename} ({len(data)} entries)")
                except Exception as e:
                    logger.error(f"  ✗ {filename} CORRUPTED: {e}")
                    self.errors.append(f"Corrupted file: {filename}")
                    all_exist = False
            else:
                logger.error(f"  ✗ {filename} MISSING")
                self.errors.append(f"Missing file: {filename}")
                all_exist = False
        
        return all_exist
    
    def check_image_integrity(self, max_check: int = 100) -> bool:
        """Check if images can be loaded and have valid formats.
        
        Args:
            max_check: Maximum number of images to check
            
        Returns:
            success: True if all checked images valid
        """
        logger.info("\n" + "="*60)
        logger.info(f"CHECKING IMAGE INTEGRITY (sampling {max_check} images)")
        logger.info("="*60)
        
        # Load feature index
        index_path = self.metadata_dir / "feature_index.json"
        if not index_path.exists():
            logger.error("Cannot check images: feature_index.json missing")
            return False
        
        with open(index_path) as f:
            feature_index = json.load(f)
        
        # Sample images to check
        image_names = list(feature_index.keys())[:max_check]
        
        corrupt_count = 0
        size_mismatches = 0
        
        for image_name in tqdm(image_names, desc="Checking images"):
            data = feature_index[image_name]
            
            # Check face_contour and jawline (required)
            for feature_type in ["face_contour", "jawline"]:
                if feature_type not in data["features"]:
                    continue
                
                path = data["features"][feature_type]
                
                try:
                    img = Image.open(path)
                    
                    # Check format
                    if img.mode not in ['RGB', 'RGBA']:
                        self.warnings.append(
                            f"{image_name}/{feature_type}: Unusual format {img.mode}"
                        )
                    
                    # Check size
                    if img.size[0] < 128 or img.size[1] < 128:
                        self.warnings.append(
                            f"{image_name}/{feature_type}: Small size {img.size}"
                        )
                        size_mismatches += 1
                
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {path}: {e}")
                    self.errors.append(f"Corrupt image: {path}")
                    corrupt_count += 1
        
        logger.info(f"\nImage integrity check:")
        logger.info(f"  Checked: {len(image_names)} images")
        logger.info(f"  Corrupt: {corrupt_count}")
        logger.info(f"  Size issues: {size_mismatches}")
        
        return corrupt_count == 0
    
    def check_feature_completeness(self) -> bool:
        """Check if all images have required features.
        
        Returns:
            success: True if all images complete
        """
        logger.info("\n" + "="*60)
        logger.info("CHECKING FEATURE COMPLETENESS")
        logger.info("="*60)
        
        # Load feature index
        index_path = self.metadata_dir / "feature_index.json"
        with open(index_path) as f:
            feature_index = json.load(f)
        
        required_features = ["face_contour", "jawline"]
        optional_features = [
            "eyes_left", "eyes_right", "eyebrows_left", "eyebrows_right",
            "nose", "mouth_outer", "mouth_inner"
        ]
        
        missing_required = 0
        missing_optional = defaultdict(int)
        
        for image_name, data in feature_index.items():
            features = data["features"]
            
            # Check required
            for feature_type in required_features:
                if feature_type not in features:
                    logger.error(f"  ✗ {image_name}: missing {feature_type}")
                    self.errors.append(f"{image_name}: missing {feature_type}")
                    missing_required += 1
            
            # Check optional
            for feature_type in optional_features:
                if feature_type not in features:
                    missing_optional[feature_type] += 1
        
        logger.info(f"\nCompleteness check:")
        logger.info(f"  Total images: {len(feature_index)}")
        logger.info(f"  Missing required features: {missing_required}")
        
        if missing_optional:
            logger.info(f"\n  Optional features missing:")
            for feature_type, count in sorted(missing_optional.items()):
                pct = count / len(feature_index) * 100
                logger.info(f"    {feature_type}: {count} ({pct:.1f}%)")
        
        return missing_required == 0
    
    def check_train_val_test_splits(self) -> bool:
        """Check if splits are valid and non-overlapping.
        
        Returns:
            success: True if splits valid
        """
        logger.info("\n" + "="*60)
        logger.info("CHECKING TRAIN/VAL/TEST SPLITS")
        logger.info("="*60)
        
        splits_path = self.metadata_dir / "train_val_test_split.json"
        with open(splits_path) as f:
            splits = json.load(f)
        
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])
        
        # Check for overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        has_overlap = False
        
        if train_val_overlap:
            logger.error(f"  ✗ Train/val overlap: {len(train_val_overlap)} images")
            self.errors.append(f"Train/val overlap: {len(train_val_overlap)}")
            has_overlap = True
        
        if train_test_overlap:
            logger.error(f"  ✗ Train/test overlap: {len(train_test_overlap)} images")
            self.errors.append(f"Train/test overlap: {len(train_test_overlap)}")
            has_overlap = True
        
        if val_test_overlap:
            logger.error(f"  ✗ Val/test overlap: {len(val_test_overlap)} images")
            self.errors.append(f"Val/test overlap: {len(val_test_overlap)}")
            has_overlap = True
        
        if not has_overlap:
            logger.info(f"  ✓ No overlaps between splits")
            logger.info(f"  Train: {len(train_set)} images")
            logger.info(f"  Val:   {len(val_set)} images")
            logger.info(f"  Test:  {len(test_set)} images")
        
        return not has_overlap
    
    def run_all_checks(self) -> bool:
        """Run all verification checks.
        
        Returns:
            success: True if all checks pass
        """
        logger.info("\n" + "="*60)
        logger.info("DATASET VERIFICATION")
        logger.info("="*60)
        
        # Run checks
        checks = [
            ("Directory Structure", self.check_directory_structure),
            ("Metadata Files", self.check_metadata_files),
            ("Image Integrity", self.check_image_integrity),
            ("Feature Completeness", self.check_feature_completeness),
            ("Train/Val/Test Splits", self.check_train_val_test_splits)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                logger.error(f"Check '{check_name}' failed: {e}")
                results.append(False)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*60)
        
        for (check_name, _), result in zip(checks, results):
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {status}: {check_name}")
        
        # Report errors and warnings
        if self.errors:
            logger.info(f"\n{len(self.errors)} ERRORS:")
            for error in self.errors[:10]:  # Show first 10
                logger.error(f"  - {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more")
        
        if self.warnings:
            logger.info(f"\n{len(self.warnings)} WARNINGS:")
            for warning in self.warnings[:10]:
                logger.warning(f"  - {warning}")
            if len(self.warnings) > 10:
                logger.warning(f"  ... and {len(self.warnings) - 10} more")
        
        all_passed = all(results)
        
        if all_passed:
            logger.info("\n" + "="*60)
            logger.info("✓ ALL CHECKS PASSED - READY FOR TRAINING!")
            logger.info("="*60 + "\n")
        else:
            logger.info("\n" + "="*60)
            logger.info("✗ VERIFICATION FAILED - FIX ERRORS BEFORE TRAINING")
            logger.info("="*60 + "\n")
        
        return all_passed


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify dataset integrity")
    parser.add_argument("--dataset-dir", type=str,
                       default="/DATA/facial_features_dataset",
                       help="Dataset root directory")
    parser.add_argument("--max-check", type=int, default=100,
                       help="Maximum images to check for integrity")
    
    args = parser.parse_args()
    
    verifier = DatasetVerifier(dataset_dir=args.dataset_dir)
    success = verifier.run_all_checks()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
