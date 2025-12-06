"""
Complete Feature Extraction Pipeline - v16 NOSE TRIANGLE FIX
- nose_tip = vertical line through nose
- nose_base = horizontal line at base
- Create triangle: top of nose_tip + left of nose_base + right of nose_base
- Expand this triangle region by margins
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
from PIL import Image
from numpy.typing import NDArray
import cv2
from tqdm import tqdm
from datetime import datetime
import csv
from typing import Any
from typing import Dict, List, Optional, Tuple

from src.landmark_detector import LandmarkDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FacialFeaturesExtractor:
    """Extract facial features using nose triangle."""
    
    def __init__(self, 
                 input_dir: str = "/DATA/facial_features_dataset/raw_images",
                 output_dir: str = "/DATA/facial_features_dataset"):
        """Initialize feature extractor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.features_dir = self.output_dir / "features"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.feature_types = [
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
        
        for feature_type in self.feature_types:
            (self.features_dir / feature_type).mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing landmark detector...")
        self.landmark_detector = LandmarkDetector()
        logger.info("✓ Landmark detector initialized")
        
        self.log_file = self.metadata_dir / "processing_log.csv"
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize processing log CSV."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_name', 'source', 'processed_date',
                    'landmarks_success', 'features_extracted', 
                    'processing_time_sec', 'errors',
                ])
    
    def process_image(self, image_path: Path, source: str) -> Dict[str, object]:
        """Process single image and save full landmark metadata for later indexing."""
        start_time = datetime.now()
        image_name = image_path.stem

        result: Dict[str, Any] = {
            "image_name": image_name,
            "source": source,
            "success": False,
            "errors": [],
        }

        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            img_height, img_width = img_array.shape[:2]

            # Run landmark detector (unchanged call)
            landmark_result: Dict[str, Any] = self.landmark_detector.detect(  # type: ignore
                str(image_path),
                return_visualization=True,
                return_groups=True,
            )

            # === NEW: save FULL landmark + groups metadata ===
            # This will include groups[detector_group_name]['bbox'] and ['landmarks_pixel']
            landmark_data: Dict[str, Any] = {
                "image_name": image_name,
                "image_size": {"width": img_width, "height": img_height},
                "num_landmarks": landmark_result.get("num_landmarks", 0),
                "groups": landmark_result.get("groups", {}),
            }

            json_path = self.features_dir / "landmarks" / f"{image_name}_landmarks.json"
            with open(json_path, "w") as f:
                json.dump(landmark_data, f)

            # Save visualization (unchanged)
            if "visualization" in landmark_result:
                vis_path = self.features_dir / "landmarks" / f"{image_name}_landmarks_visual.png"
                landmark_result["visualization"].save(str(vis_path))

            result["landmarks_success"] = True

            # Extract feature crops/masks (your existing logic)
            features_extracted = self._extract_all_features(  # type: ignore
                img,
                img_array,
                landmark_result,
                image_name,
                img_width,
                img_height,
            )

            result["features_extracted"] = features_extracted
            result["success"] = True
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time_sec"] = processing_time

            self._log_result(result)
            status = "✓" if result["success"] else "✗"
            logger.info(
                f"{status} {image_name}: {features_extracted} features, {processing_time:.2f}s"
            )  # type: ignore

        except Exception as e:
            logger.error(f"Failed to process {image_name}: {e}")
            result["errors"].append(str(e))
            self._log_result(result)

        return result

    def _extract_all_features(self, img: Image.Image, img_array: NDArray[np.uint8],
                             landmark_result: Dict[str, Any], image_name: str,
                             img_width: int, img_height: int) -> int:
        """Extract features from all landmark groups."""
        
        features_extracted = 0
        groups = landmark_result.get('groups', {})
        
        feature_masks: Dict[str, NDArray[np.uint8]] = {}
        
        feature_mapping = {
            'left_eye': 'eyes_left',
            'right_eye': 'eyes_right',
            'left_eyebrow': 'eyebrows_left',
            'right_eyebrow': 'eyebrows_right',
            'nose_tip': 'nose',
            'nose_base': 'nose',
            'mouth_outer': 'mouth_outer',
            'mouth_inner': 'mouth_inner',
            'left_ear': 'ears_left',
            'right_ear': 'ears_right',
            'face_contour': 'face_contour',
            'jawline': 'jawline',
        }
        
        subtract_from_contour = [
            'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow',
            'nose_tip', 'nose_base',
            'mouth_outer', 'mouth_inner'
        ]
        
        # SPECIAL: Handle NOSE with TRIANGLE
        if 'nose_tip' in groups and 'nose_base' in groups:
            nose_tip_data = groups['nose_tip']
            nose_base_data = groups['nose_base']
            
            has_nose_tip = ('landmarks_pixel' in nose_tip_data and 
                           len(nose_tip_data['landmarks_pixel']) > 0)
            has_nose_base = ('landmarks_pixel' in nose_base_data and 
                            len(nose_base_data['landmarks_pixel']) > 0)
            
            if has_nose_tip and has_nose_base:
                nose_tip_lms = np.array([[int(lm[0]), int(lm[1])] 
                                         for lm in nose_tip_data['landmarks_pixel']], dtype=np.int32)
                nose_base_lms = np.array([[int(lm[0]), int(lm[1])] 
                                          for lm in nose_base_data['landmarks_pixel']], dtype=np.int32)
                
                if len(nose_tip_lms) > 0 and len(nose_base_lms) > 0:
                    # Find key points
                    # Top point = topmost point of nose_tip (minimum y)
                    top_point = nose_tip_lms[np.argmin(nose_tip_lms[:, 1])]
                    
                    # Left point = leftmost point of nose_base (minimum x)
                    left_point = nose_base_lms[np.argmin(nose_base_lms[:, 0])]
                    
                    # Right point = rightmost point of nose_base (maximum x)
                    right_point = nose_base_lms[np.argmax(nose_base_lms[:, 0])]
                    
                    logger.debug(f"  Nose triangle: top={top_point}, left={left_point}, right={right_point}")
                    
                    # Create triangle points
                    triangle_points = np.array([top_point, left_point, right_point], dtype=np.int32)
                    
                    # Get bbox of triangle
                    xs = triangle_points[:, 0]
                    ys = triangle_points[:, 1]
                    
                    x_min_tri = int(np.min(xs))
                    x_max_tri = int(np.max(xs))
                    y_min_tri = int(np.min(ys))
                    y_max_tri = int(np.max(ys))
                    
                    # Expand with margins
                    margin_x = 180
                    margin_y = 140
                    
                    x_min = max(0, x_min_tri - margin_x)
                    y_min = max(0, y_min_tri - margin_y)
                    x_max = min(img_width, x_max_tri + margin_x)
                    y_max = min(img_height, y_max_tri + margin_y)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    logger.debug(f"  Nose: triangle bbox ({x_min_tri}, {y_min_tri}, {x_max_tri}, {y_max_tri}) "
                                f"-> expanded ({x_min}, {y_min}, {x_max}, {y_max}) "
                                f"[{width}x{height}]")
                    
                    if width >= 10 and height >= 10:
                        # Crop image
                        cropped_img = img.crop((x_min, y_min, x_max, y_max))
                        
                        # Create mask with triangle
                        cropped_mask = self._create_triangle_mask(
                            triangle_points,
                            (x_min, y_min, x_max, y_max),
                            (img_height, img_width)
                        )
                        
                        # Store full-size mask for subtraction
                        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(full_mask, [triangle_points], color=(255,))
                        feature_masks['nose_tip'] = full_mask
                        feature_masks['nose_base'] = full_mask
                        
                        # Apply mask with transparency
                        masked_img = self._apply_mask_transparency(cropped_img, cropped_mask)
                        
                        # Save
                        img_path = self.features_dir / "nose" / f"{image_name}_nose.png"
                        masked_img.save(str(img_path), quality=95)
                        
                        mask_path = self.features_dir / "nose" / f"{image_name}_nose_mask.png"
                        cropped_mask.save(str(mask_path))
                        
                        features_extracted += 1
                        logger.debug(f"  ✓ Extracted NOSE TRIANGLE")
        
        # First pass: extract all other features
        for detector_group_name, group_data in groups.items():
            if detector_group_name == 'face_contour':
                continue
            
            # Skip nose - already handled
            if detector_group_name in ['nose_tip', 'nose_base']:
                continue
            
            try:
                if 'landmarks_pixel' not in group_data:
                    continue
                
                landmarks_pixel = group_data['landmarks_pixel']
                if len(landmarks_pixel) == 0:
                    continue
                
                if 'bbox' not in group_data or group_data['bbox'] is None:
                    continue
                
                bbox = group_data['bbox']
                feature_name = feature_mapping.get(detector_group_name, detector_group_name)
                
                # Extract x, y coordinates
                landmarks_2d = []
                for lm in landmarks_pixel:
                    landmarks_2d.append([int(lm[0]), int(lm[1])]) # type: ignore
                
                landmarks_2d = np.array(landmarks_2d, dtype=np.int32)
                
                # Regular bbox
                x_min = max(0, int(bbox['x_min']))
                y_min = max(0, int(bbox['y_min']))
                x_max = min(img_width, int(bbox['x_max']))
                y_max = min(img_height, int(bbox['y_max']))
                
                width = x_max - x_min
                height = y_max - y_min
                
                if width < 5 or height < 5:
                    continue
                
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                cropped_mask: Image.Image = self._create_mask_from_contour(
                    landmarks_2d,
                    (x_min, y_min, x_max, y_max),
                    (img_height, img_width)
                )
                
                # Store mask for subtraction
                if detector_group_name in subtract_from_contour:
                    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    cv2.fillPoly(full_mask, [landmarks_2d], color=(255,))
                    feature_masks[detector_group_name] = full_mask
                
                masked_img = self._apply_mask_transparency(cropped_img, cropped_mask)
                
                img_path = self.features_dir / str(feature_name) / f"{image_name}_{feature_name}.png"
                masked_img.save(str(img_path), quality=95)
                
                mask_path = self.features_dir / str(feature_name) / f"{image_name}_{feature_name}_mask.png"
                cropped_mask.save(str(mask_path))
                
                features_extracted += 1
                logger.debug(f"  ✓ Extracted {feature_name}")
            
            except Exception as e:
                logger.debug(f"  Error extracting {detector_group_name}: {e}")
        
        # Second pass: process face_contour
        try:
            if 'face_contour' in groups:
                logger.debug("  Processing face_contour...")
                
                face_contour_data = groups['face_contour']
                
                if 'landmarks_pixel' in face_contour_data and 'bbox' in face_contour_data:
                    
                    landmarks_pixel = face_contour_data['landmarks_pixel']
                    bbox = face_contour_data['bbox']
                    
                    if len(landmarks_pixel) > 0 and bbox:
                        landmarks_2d = []
                        for lm in landmarks_pixel:
                            if isinstance(lm, (list, tuple)):
                                landmarks_2d.append([int(lm[0]), int(lm[1])]) # type: ignore
                            else:
                                landmarks_2d.append([int(lm[0]), int(lm[1])]) # type: ignore
                        
                        landmarks_2d = np.array(landmarks_2d, dtype=np.int32)
                        
                        x_min = max(0, int(bbox['x_min']))
                        y_min = max(0, int(bbox['y_min']))
                        x_max = min(img_width, int(bbox['x_max']))
                        y_max = min(img_height, int(bbox['y_max']))
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        if width >= 5 and height >= 5:
                            face_contour_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.fillPoly(face_contour_mask, [landmarks_2d], color=(255,))
                            
                            # Subtract all features
                            subtracted_count = 0
                            for feature_name in subtract_from_contour:
                                if feature_name in feature_masks:
                                    face_contour_mask = cv2.subtract(face_contour_mask, feature_masks[feature_name])
                                    subtracted_count += 1
                            
                            logger.debug(f"    Subtracted {subtracted_count} features")
                            
                            cropped_mask = Image.fromarray(face_contour_mask[y_min:y_max, x_min:x_max], mode='L')
                            cropped_img = img.crop((x_min, y_min, x_max, y_max))
                            
                            masked_img = self._apply_mask_transparency(cropped_img, cropped_mask)
                            
                            img_path = self.features_dir / "face_contour" / f"{image_name}_face_contour.png"
                            masked_img.save(str(img_path), quality=95)
                            
                            mask_path = self.features_dir / "face_contour" / f"{image_name}_face_contour_mask.png"
                            with mask_path.open("wb") as f:
                                cropped_mask.save(f, format="PNG")
                            
                            logger.debug(f"  ✓ Extracted face_contour")
        
        except Exception as e:
            logger.debug(f"  Error processing face_contour: {e}")
        
        return features_extracted
    
    def _create_triangle_mask(self, triangle_points: NDArray[np.int32], bbox: Tuple[int, int, int, int],
                               img_size: Tuple[int, int]) -> Image.Image:
        """Create binary mask from triangle."""
        height, width = img_size
        x1, y1, x2, y2 = bbox
        
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw filled triangle
        cv2.fillPoly(full_mask, [triangle_points], color=(255,))
        
        # Crop to bbox
        cropped_mask_array = full_mask[y1:y2, x1:x2]
        cropped_mask = Image.fromarray(cropped_mask_array, mode='L')
        
        return cropped_mask
    
    def _apply_mask_transparency(self, cropped_img: Image.Image, 
                                mask: Image.Image) -> Image.Image:
        """Apply mask to image with transparency."""
        img_rgba = cropped_img.convert('RGBA')
        
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        alpha_channel = mask
        img_rgba.putalpha(alpha_channel)
        
        return img_rgba
    
    def _create_mask_from_contour(self, landmarks_2d: NDArray[np.int32], bbox: Tuple[int, int, int, int],
                                  img_size: Tuple[int, int]) -> Image.Image:
        """Create binary mask from 2D landmark contour."""
        height, width = img_size
        x1, y1, x2, y2 = bbox
        
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(landmarks_2d) >= 3:
            try:
                cv2.fillPoly(full_mask, [landmarks_2d], color=(255,))
            except Exception as e:
                logger.debug(f"fillPoly failed: {e}")
                for pt in landmarks_2d:
                    cv2.circle(full_mask, tuple(pt), 2, (255,), -1)
        else:
            for pt in landmarks_2d:
                cv2.circle(full_mask, tuple(pt), 3, (255,), -1)
        
        cropped_mask_array = full_mask[y1:y2, x1:x2]
        cropped_mask = Image.fromarray(cropped_mask_array, mode='L')
        
        return cropped_mask
    
    def _log_result(self, result: Dict[str, Any]):
        """Log result to CSV."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['image_name'],
                    result['source'],
                    datetime.now().isoformat(),
                    result.get('landmarks_success', False),
                    result.get('features_extracted', 0),
                    result.get('processing_time_sec', 0),
                    '; '.join(result.get('errors', []))
                ])
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
    
    def process_dataset(self, max_images: Optional[int] = None):
        """Process entire dataset."""
        logger.info("\n" + "="*60)
        logger.info("FACIAL FEATURE EXTRACTION (v16 NOSE TRIANGLE)")
        logger.info("="*60)
        logger.info("- Nose as TRIANGLE: top(nose_tip) + left(nose_base) + right(nose_base)")
        logger.info("- Triangle expanded by margins: ±80px width, ±100px height")
        logger.info("- Nose SUBTRACTED from face_contour")
        logger.info("="*60)
        
        image_paths: List[Tuple[Path, str]] = []
        
        ffhq_dir = self.input_dir / "ffhq"
        if ffhq_dir.exists():
            image_paths.extend([(p, 'ffhq') for p in sorted(ffhq_dir.glob("*.png"))])
        
        celeba_dir = self.input_dir / "celeba_hq"
        if celeba_dir.exists():
            image_paths.extend([(p, 'celeba_hq') for p in sorted(celeba_dir.glob("*.jpg"))])
            image_paths.extend([(p, 'celeba_hq') for p in sorted(celeba_dir.glob("*.png"))])
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        if len(image_paths) == 0:
            logger.error("NO IMAGES FOUND!")
            return
        
        logger.info(f"Total images: {len(image_paths)}\n")
        
        successful = 0
        failed = 0
        
        for img_path, source in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(img_path, source)
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Successful: {successful}/{len(image_paths)}")
        logger.info(f"Failed: {failed}/{len(image_paths)}")
        if len(image_paths) > 0:
            logger.info(f"Success rate: {successful/len(image_paths)*100:.1f}%")
        logger.info("="*60 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract facial features with nose triangle")
    parser.add_argument("--input", type=str, default="/DATA/facial_features_dataset/raw_images")
    parser.add_argument("--output", type=str, default="/DATA/facial_features_dataset")
    parser.add_argument("--max-images", type=int, default=None)
    
    args = parser.parse_args()
    
    extractor = FacialFeaturesExtractor(
        input_dir=args.input,
        output_dir=args.output
    )
    
    extractor.process_dataset(max_images=args.max_images)


if __name__ == "__main__":
    main()
