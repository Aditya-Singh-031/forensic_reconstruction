"""
Dataset Validation Script
Validates extracted features and generates quality reports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate extracted dataset."""
    
    def __init__(self, dataset_root: str = "/DATA/facial_features_dataset"):
        """Initialize validator."""
        self.dataset_root = Path(dataset_root)
        self.features_dir = self.dataset_root / "features"
        self.metadata_dir = self.dataset_root / "metadata"
        self.annotations_dir = self.dataset_root / "annotations"
        
        logger.info(f"Validating dataset at: {self.dataset_root}")
    
    def validate_directory_structure(self) -> Dict:
        """Validate directory structure."""
        logger.info("\n[1/5] Validating directory structure...")
        
        issues = []
        required_dirs = [
            "raw_images",
            "features/segmentation",
            "features/landmarks",
            "features/eyes_left",
            "features/eyes_right",
            "features/eyebrows_left",
            "features/eyebrows_right",
            "features/nose",
            "features/mouth_outer",
            "features/mouth_inner",
            "features/ears_left",
            "features/ears_right",
            "features/face_contour",
            "features/jawline",
            "metadata",
            "annotations"
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                issues.append(f"Missing directory: {dir_path}")
            else:
                logger.info(f"  ✓ {dir_path}")
        
        if issues:
            logger.warning(f"Found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Directory structure is valid")
        
        return {'issues': issues, 'valid': len(issues) == 0}
    
    def validate_feature_files(self) -> Dict:
        """Validate extracted feature files."""
        logger.info("\n[2/5] Validating feature files...")
        
        feature_types = [
            "segmentation", "landmarks", "eyes_left", "eyes_right",
            "eyebrows_left", "eyebrows_right", "nose", "mouth_outer",
            "mouth_inner", "ears_left", "ears_right", "face_contour", "jawline"
        ]
        
        stats = {
            'total_images': 0,
            'features_by_type': {},
            'corrupted_files': [],
            'issues': []
        }
        
        # Count files
        for feature_type in feature_types:
            feature_dir = self.features_dir / feature_type
            if feature_dir.exists():
                files = list(feature_dir.glob("*"))
                stats['features_by_type'][feature_type] = len(files)
                logger.info(f"  {feature_type}: {len(files)} files")
        
        # Check for corrupted images
        logger.info("\nChecking image integrity...")
        image_dirs = [d for d in (self.features_dir).glob("*") if d.is_dir()]
        
        for feature_dir in image_dirs:
            for img_file in feature_dir.glob("*.png"):
                try:
                    img = Image.open(img_file)
                    img.verify()
                except Exception as e:
                    stats['corrupted_files'].append(str(img_file))
                    logger.warning(f"  Corrupted: {img_file.name}")
        
        if stats['corrupted_files']:
            logger.warning(f"Found {len(stats['corrupted_files'])} corrupted files")
        else:
            logger.info("✓ All images are valid")
        
        return stats
    
    def validate_metadata_files(self) -> Dict:
        """Validate metadata files."""
        logger.info("\n[3/5] Validating metadata files...")
        
        stats = {
            'files_found': [],
            'issues': [],
            'valid': True
        }
        
        # Check dataset_info.json
        info_file = self.metadata_dir / "dataset_metadata.json"
        if info_file.exists():
            try:
                with open(info_file) as f:
                    data = json.load(f)
                logger.info(f"  ✓ dataset_metadata.json")
                stats['files_found'].append('dataset_metadata.json')
            except Exception as e:
                stats['issues'].append(f"Invalid JSON: dataset_metadata.json - {e}")
                stats['valid'] = False
        else:
            stats['issues'].append("Missing: dataset_metadata.json")
        
        # Check processing_log.csv
        log_file = self.metadata_dir / "processing_log.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                logger.info(f"  ✓ processing_log.csv ({len(df)} entries)")
                stats['files_found'].append('processing_log.csv')
                stats['log_entries'] = len(df)
                stats['successful'] = df['segmentation_success'].sum()
                stats['failed'] = len(df) - stats['successful']
            except Exception as e:
                stats['issues'].append(f"Invalid CSV: processing_log.csv - {e}")
                stats['valid'] = False
        else:
            stats['issues'].append("Missing: processing_log.csv")
        
        if stats['valid']:
            logger.info("✓ All metadata files valid")
        else:
            logger.warning(f"Found {len(stats['issues'])} issues")
        
        return stats
    
    def validate_landmarks(self) -> Dict:
        """Validate landmark annotations."""
        logger.info("\n[4/5] Validating landmarks...")
        
        stats = {
            'total_landmarks': 0,
            'files_checked': 0,
            'invalid_coordinates': 0,
            'issues': []
        }
        
        landmarks_dir = self.features_dir / "landmarks"
        if not landmarks_dir.exists():
            stats['issues'].append("Landmarks directory not found")
            return stats
        
        for json_file in tqdm(landmarks_dir.glob("*_landmarks.json"), desc="Checking landmarks"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                stats['files_checked'] += 1
                
                if 'landmarks' in data:
                    for lm in data['landmarks']:
                        if 'x' in lm and 'y' in lm:
                            if not (0 <= lm['x'] <= 10000 and 0 <= lm['y'] <= 10000):
                                stats['invalid_coordinates'] += 1
                        stats['total_landmarks'] += 1
            
            except Exception as e:
                stats['issues'].append(f"Invalid landmark file: {json_file.name}")
        
        logger.info(f"  Files checked: {stats['files_checked']}")
        logger.info(f"  Total landmarks: {stats['total_landmarks']}")
        logger.info(f"  Invalid coords: {stats['invalid_coordinates']}")
        
        if stats['invalid_coordinates'] > 0:
            logger.warning(f"Found {stats['invalid_coordinates']} invalid coordinates")
        else:
            logger.info("✓ All landmarks valid")
        
        return stats
    
    def generate_report(self) -> Dict:
        """Generate validation report."""
        logger.info("\n[5/5] Generating validation report...")
        
        # Run all validations
        dir_validation = self.validate_directory_structure()
        file_validation = self.validate_feature_files()
        metadata_validation = self.validate_metadata_files()
        landmark_validation = self.validate_landmarks()
        
        # Compile report
        report = {
            "validation_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_root),
            "validations": {
                "directory_structure": dir_validation,
                "feature_files": file_validation,
                "metadata": metadata_validation,
                "landmarks": landmark_validation
            },
            "summary": {
                "total_issues": (
                    len(dir_validation.get('issues', [])) +
                    len(file_validation.get('issues', [])) +
                    len(metadata_validation.get('issues', [])) +
                    len(landmark_validation.get('issues', []))
                ),
                "is_valid": all([
                    dir_validation.get('valid', False),
                    metadata_validation.get('valid', True),
                    len(file_validation.get('corrupted_files', [])) == 0,
                    len(landmark_validation.get('issues', [])) == 0
                ])
            }
        }
        
        # Save report
        report_path = self.metadata_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to {report_path}")
        
        return report
    
    def print_summary(self, report: Dict):
        """Print validation summary."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        summary = report['summary']
        logger.info(f"Total issues: {summary['total_issues']}")
        logger.info(f"Status: {'✓ VALID' if summary['is_valid'] else '✗ INVALID'}")
        
        if report['validations']['feature_files'].get('features_by_type'):
            logger.info("\nFeature counts:")
            for feat_type, count in report['validations']['feature_files']['features_by_type'].items():
                logger.info(f"  {feat_type}: {count}")
        
        if 'log_entries' in report['validations']['metadata']:
            meta = report['validations']['metadata']
            logger.info(f"\nProcessing statistics:")
            logger.info(f"  Total: {meta['log_entries']}")
            logger.info(f"  Successful: {meta['successful']}")
            logger.info(f"  Failed: {meta['failed']}")
        
        logger.info("="*60 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate facial features dataset")
    parser.add_argument("--path", type=str, default="/DATA/facial_features_dataset",
                       help="Dataset root path")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.path)
    report = validator.generate_report()
    validator.print_summary(report)


if __name__ == "__main__":
    main()
