"""
Complete Feature Extraction Pipeline - FIXED
Extracts facial features from images using segmentation and landmarks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from datetime import datetime
import csv
from typing import Dict, List, Optional, Tuple
import traceback

from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FacialFeaturesExtractor:
    """Extract facial features from images."""
    
    # Landmark group definitions (MediaPipe 468 landmarks)
    LANDMARK_GROUPS = {
        'face_contour': list(range(0, 17)),
        'left_eyebrow': list(range(17, 27)),
        'right_eyebrow': list(range(27, 37)),
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
        'nose': list(range(27, 36)) + list(range(48, 68)),
        'mouth_outer': list(range(48, 60)),
        'mouth_inner': list(range(60, 68)),
        'left_ear': list(range(454, 464)),
        'right_ear': list(range(464, 474)),
        'jawline': list(range(0, 17))
    }
    
    def __init__(self, 
                 input_dir: str = "/DATA/facial_features_dataset/raw_images",
                 output_dir: str = "/DATA/facial_features_dataset",
                 device: str = "cuda"):
        """Initialize feature extractor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output structure
        self.features_dir = self.output_dir / "features"
        self.metadata_dir = self.output_dir / "metadata"
        self.annotations_dir = self.output_dir / "annotations"
        
        # Feature subdirectories
        self.feature_types = [
            "segmentation",
            "landmarks",
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
            "jawline"
        ]
        
        # Create directories
        for feature_type in self.feature_types:
            (self.features_dir / feature_type).mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info("Initializing models...")
        self.segmenter = FaceSegmenter(device=device)
        self.landmark_detector = LandmarkDetector()
        
        logger.info("✓ Feature extractor initialized")
        
        # Initialize log file
        self.log_file = self.metadata_dir / "processing_log.csv"
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize processing log CSV."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_name', 'source', 'processed_date',
                    'segmentation_success', 'landmarks_success',
                    'features_extracted', 'processing_time_sec', 'errors', 'notes'
                ])
    
    def process_image(self, image_path: Path, source: str) -> Dict:
        """
        Process single image and extract features.
        
        Args:
            image_path: Path to input image
            source: Dataset source (ffhq/celeba_hq)
            
        Returns:
            Processing results dict
        """
        start_time = datetime.now()
        image_name = image_path.stem
        
        result = {
            'image_name': image_name,
            'source': source,
            'success': False,
            'errors': [],
            'notes': []
        }
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 1. Segmentation
            logger.debug(f"Processing: {image_name} [Segmentation]")
            try:
                seg_result = self.segmenter.segment(str(image_path))
                
                # Save segmentation mask (numpy)
                mask_path = self.features_dir / "segmentation" / f"{image_name}_segmentation_mask.npy"
                np.save(str(mask_path), seg_result['segmentation'])
                
                # Save segmentation visualization
                vis_path = self.features_dir / "segmentation" / f"{image_name}_segmentation.png"
                seg_vis = self._visualize_segmentation(seg_result['segmentation'])
                seg_vis.save(str(vis_path))
                
                result['segmentation_success'] = True
                logger.debug(f"  ✓ Segmentation successful")
            except Exception as e:
                logger.debug(f"  ✗ Segmentation failed: {e}")
                result['errors'].append(f"Segmentation: {str(e)}")
                result['segmentation_success'] = False
            
            # 2. Landmark Detection
            logger.debug(f"Processing: {image_name} [Landmarks]")
            landmark_data = None
            try:
                landmark_result = self.landmark_detector.detect(str(image_path))
                
                if landmark_result and 'landmarks' in landmark_result:
                    landmarks = landmark_result['landmarks']
                    
                    # Save landmarks JSON
                    landmark_data = {
                        'image_name': image_name,
                        'num_landmarks': len(landmarks),
                        'landmarks': [
                            {
                                'id': i,
                                'x': float(lm[0]) if isinstance(lm[0], (int, float)) else float(lm[0]),
                                'y': float(lm[1]) if isinstance(lm[1], (int, float)) else float(lm[1]),
                                'z': float(lm[2]) if len(lm) > 2 and isinstance(lm[2], (int, float)) else 0.0,
                                'group': self._get_landmark_group(i)
                            }
                            for i, lm in enumerate(landmarks)
                        ],
                        'bbox': self._calculate_bbox(landmarks)
                    }
                    
                    json_path = self.features_dir / "landmarks" / f"{image_name}_landmarks.json"
                    with open(json_path, 'w') as f:
                        json.dump(landmark_data, f, indent=2)
                    
                    # Save landmark visualization
                    vis_path = self.features_dir / "landmarks" / f"{image_name}_landmarks_visual.png"
                    landmark_vis = self._visualize_landmarks(img, landmarks)
                    landmark_vis.save(str(vis_path))
                    
                    result['landmarks_success'] = True
                    logger.debug(f"  ✓ Landmarks detected: {len(landmarks)} points")
                else:
                    result['landmarks_success'] = False
                    result['errors'].append("No landmarks detected")
                    logger.debug(f"  ✗ No landmarks detected")
            
            except Exception as e:
                logger.debug(f"  ✗ Landmark detection failed: {e}")
                result['errors'].append(f"Landmarks: {str(e)}")
                result['landmarks_success'] = False
            
            # 3. Extract individual features
            features_extracted = 0
            if result['segmentation_success'] and result['landmarks_success']:
                logger.debug(f"Processing: {image_name} [Feature Extraction]")
                try:
                    features_extracted = self._extract_features(
                        img, 
                        landmark_data,
                        image_name
                    )
                    result['features_extracted'] = features_extracted
                    logger.debug(f"  ✓ Extracted {features_extracted} features")
                except Exception as e:
                    logger.debug(f"  ✗ Feature extraction failed: {e}")
                    result['errors'].append(f"Features: {str(e)}")
            
            # Calculate time
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time_sec'] = processing_time
            result['success'] = result['segmentation_success'] and result['landmarks_success']
            
            # Log result
            self._log_result(result)
            
            status = "✓" if result['success'] else "✗"
            logger.info(f"{status} {image_name}: {features_extracted} features, {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process {image_name}: {e}")
            result['errors'].append(f"General: {str(e)}")
            self._log_result(result)
        
        return result
    
    def _extract_features(self, img: Image.Image, landmark_data: Dict, 
                         image_name: str) -> int:
        """Extract and save individual features."""
        features_extracted = 0
        
        if not landmark_data or 'landmarks' not in landmark_data:
            return 0
        
        landmarks = np.array([
            [lm['x'], lm['y']] for lm in landmark_data['landmarks']
        ])
        
        # Map feature names to landmark indices
        feature_mapping = {
            'eyes_left': 'left_eye',
            'eyes_right': 'right_eye',
            'eyebrows_left': 'left_eyebrow',
            'eyebrows_right': 'right_eyebrow',
            'nose': 'nose',
            'mouth_outer': 'mouth_outer',
            'mouth_inner': 'mouth_inner',
            'ears_left': 'left_ear',
            'ears_right': 'right_ear',
            'face_contour': 'face_contour',
            'jawline': 'jawline'
        }
        
        for feature_name, group_name in feature_mapping.items():
            try:
                if group_name in self.LANDMARK_GROUPS:
                    indices = self.LANDMARK_GROUPS[group_name]
                    feature_landmarks = landmarks[indices]
                    
                    # Get bounding box
                    bbox = self._get_feature_bbox(feature_landmarks, img.size)
                    
                    if bbox and self._is_valid_bbox(bbox):
                        # Extract feature
                        feature_img = img.crop(bbox)
                        feature_mask = self._create_feature_mask(img.size, feature_landmarks, bbox)
                        
                        # Save feature image
                        img_path = self.features_dir / feature_name / f"{image_name}_{feature_name}.png"
                        feature_img.save(str(img_path), quality=95)
                        
                        # Save feature mask
                        mask_path = self.features_dir / feature_name / f"{image_name}_{feature_name}_mask.png"
                        feature_mask.save(str(mask_path))
                        
                        features_extracted += 1
            
            except Exception as e:
                logger.debug(f"Could not extract {feature_name}: {e}")
        
        return features_extracted
    
    def _visualize_segmentation(self, seg_mask: np.ndarray) -> Image.Image:
        """Create colored segmentation visualization."""
        # Color map for 19 classes
        colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
            (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
            (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0)
        ]
        
        # Create RGB image
        height, width = seg_mask.shape
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(len(colors)):
            mask = seg_mask == class_id
            vis[mask] = colors[class_id]
        
        return Image.fromarray(vis)
    
    def _visualize_landmarks(self, img: Image.Image, landmarks: List) -> Image.Image:
        """Draw landmarks on image."""
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        
        for i, lm in enumerate(landmarks):
            x, y = int(lm[0]), int(lm[1])
            
            # Draw point
            r = 2
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0), outline=(0, 255, 0))
        
        return vis
    
    def _create_feature_mask(self, img_size: Tuple[int, int], 
                            landmarks: np.ndarray, bbox: Tuple) -> Image.Image:
        """Create binary mask for feature."""
        mask = Image.new('L', img_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Convert to image coordinates relative to full image
        points = [tuple(lm) for lm in landmarks]
        
        if len(points) >= 3:
            try:
                draw.polygon(points, fill=255)
            except:
                for pt in points:
                    draw.point(pt, fill=255)
        
        # Crop to bbox
        if bbox:
            mask = mask.crop(bbox)
        
        return mask
    
    def _get_feature_bbox(self, landmarks: np.ndarray, 
                         img_size: Tuple[int, int]) -> Optional[Tuple]:
        """Get bounding box from landmarks."""
        if landmarks.size == 0:
            return None
        
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        
        # Add margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(img_size[0], x_max + margin)
        y_max = min(img_size[1], y_max + margin)
        
        return (x_min, y_min, x_max, y_max)
    
    def _is_valid_bbox(self, bbox: Tuple, min_size: int = 20) -> bool:
        """Check if bbox is valid."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width >= min_size and height >= min_size
    
    def _calculate_bbox(self, landmarks: List) -> Dict:
        """Calculate bounding box from landmarks."""
        if not landmarks:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        
        return {
            'x': int(min(xs)),
            'y': int(min(ys)),
            'width': int(max(xs) - min(xs)),
            'height': int(max(ys) - min(ys))
        }
    
    def _get_landmark_group(self, landmark_id: int) -> str:
        """Get feature group for landmark ID."""
        for group_name, indices in self.LANDMARK_GROUPS.items():
            if landmark_id in indices:
                return group_name
        return 'other'
    
    def _log_result(self, result: Dict):
        """Log result to CSV."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['image_name'],
                    result['source'],
                    datetime.now().isoformat(),
                    result.get('segmentation_success', False),
                    result.get('landmarks_success', False),
                    result.get('features_extracted', 0),
                    result.get('processing_time_sec', 0),
                    '; '.join(result.get('errors', [])),
                    '; '.join(result.get('notes', []))
                ])
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
    
    def process_dataset(self, batch_size: int = 50, max_images: Optional[int] = None):
        """Process entire dataset."""
        logger.info("\n" + "="*60)
        logger.info("STARTING DATASET PROCESSING")
        logger.info("="*60)
        
        # Get all image paths
        image_paths = []
        
        # FFHQ
        ffhq_dir = self.input_dir / "ffhq"
        if ffhq_dir.exists():
            image_paths.extend([
                (p, 'ffhq') for p in sorted(ffhq_dir.glob("*.png"))
            ])
        
        # CelebA-HQ
        celeba_dir = self.input_dir / "celeba_hq"
        if celeba_dir.exists():
            image_paths.extend([
                (p, 'celeba_hq') for p in sorted(celeba_dir.glob("*.jpg"))
            ])
            image_paths.extend([
                (p, 'celeba_hq') for p in sorted(celeba_dir.glob("*.png"))
            ])
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        # FIXED: Handle case where no images found
        if len(image_paths) == 0:
            logger.warning("\n⚠️  NO IMAGES FOUND!")
            logger.warning("="*60)
            logger.warning("No images found in:")
            logger.warning(f"  - {ffhq_dir}")
            logger.warning(f"  - {celeba_dir}")
            logger.warning("\nPlease download datasets first:")
            logger.warning("  python -m src.dataset_downloader --dataset both")
            logger.warning("="*60 + "\n")
            return
        
        logger.info(f"Total images: {len(image_paths)}")
        
        # Process
        successful = 0
        failed = 0
        
        for img_path, source in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(img_path, source)
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Generate metadata
        self._generate_metadata(successful, failed, len(image_paths))
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        if len(image_paths) > 0:
            logger.info(f"Success rate: {successful/len(image_paths)*100:.1f}%")
        logger.info("="*60 + "\n")
    
    def _generate_metadata(self, successful: int, failed: int, total: int):
        """Generate final metadata."""
        logger.info("Generating metadata...")
        
        metadata = {
            "dataset_name": "Facial Features Dataset",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "models": {
                "segmentation": "nvidia/segformer-b0-finetuned-ade-512-512",
                "landmarks": "MediaPipe Face Mesh v1"
            },
            "feature_types": self.feature_types,
            "statistics": {
                "total_images": total,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total if total > 0 else 0
            }
        }
        
        with open(self.metadata_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✓ Metadata generated")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract facial features from dataset")
    parser.add_argument("--input", type=str, default="/DATA/facial_features_dataset/raw_images",
                       help="Input directory")
    parser.add_argument("--output", type=str, default="/DATA/facial_features_dataset",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    extractor = FacialFeaturesExtractor(
        input_dir=args.input,
        output_dir=args.output,
        device=args.device
    )
    
    extractor.process_dataset(max_images=args.max_images)


if __name__ == "__main__":
    main()
