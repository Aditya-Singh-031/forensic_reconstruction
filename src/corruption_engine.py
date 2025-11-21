"""
Corruption Engine - FINAL FIX for RGBA transparency
Properly handles face_contour with transparent holes
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorruptionEngine:
    """Generate corrupted faces with precise feature positioning."""
    
    CORRUPTIBLE_FEATURES = [
        "eyes_left", "eyes_right", "eyebrows_left", "eyebrows_right",
        "nose", "mouth_outer", "mouth_inner"
    ]
    
    def __init__(self, feature_index: Dict, dataset_dir: str = "/home/teaching/G14/forensic_reconstruction/dataset"):
        self.feature_index = feature_index
        self.image_names = list(feature_index.keys())
        self.dataset_dir = Path(dataset_dir)
        logger.info(f"Initialized CorruptionEngine with {len(self.image_names)} images")
    
    def get_corruption_level_features(self, level: int) -> List[str]:
        if level == 1:
            num_features = random.randint(2, 3)
        elif level == 2:
            num_features = random.randint(4, 5)
        elif level == 3:
            num_features = random.randint(6, 7)
        else:
            num_features = 4
        
        return random.sample(self.CORRUPTIBLE_FEATURES, 
                           min(num_features, len(self.CORRUPTIBLE_FEATURES)))
    
    def load_image(self, path: str, mode: str = 'RGBA') -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            if img.mode != mode:
                img = img.convert(mode)
            return img
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None
    
    def create_corrupted_face(self,
                             image_name: str,
                             corruption_level: int = 2) -> Tuple[Image.Image, Image.Image, List[str]]:
        """Create corrupted face by compositing random features at correct positions."""
        base_data = self.feature_index[image_name]
        
        # Load base face_contour (RGBA with transparent holes)
        base_path = base_data["features"]["face_contour"]
        base_img = self.load_image(base_path, 'RGBA')
        
        if base_img is None:
            raise ValueError(f"Failed to load base: {base_path}")
        
        # Load target jawline
        target_path = base_data["features"]["jawline"]
        target_img = self.load_image(target_path, 'RGBA')
        
        if target_img is None:
            raise ValueError(f"Failed to load target: {target_path}")
        
        # Determine features to corrupt
        features_to_corrupt = self.get_corruption_level_features(corruption_level)
        
        # Start with base image (RGBA with holes)
        corrupted = base_img.copy()
        actually_corrupted = []
        
        # Composite each random feature
        for feature_type in features_to_corrupt:
            # Sample random image
            random_image_name = random.choice(self.image_names)
            random_data = self.feature_index[random_image_name]
            
            if feature_type not in random_data["features"]:
                continue
            
            # Load current image's feature as position template
            if feature_type not in base_data["features"]:
                continue
                
            current_feature_path = base_data["features"][feature_type]
            current_feature = self.load_image(current_feature_path, 'RGBA')
            
            # Load random feature to paste
            random_feature_path = random_data["features"][feature_type]
            random_feature = self.load_image(random_feature_path, 'RGBA')
            
            if current_feature is None or random_feature is None:
                continue
            
            # Resize random feature to match current feature size
            random_feature_resized = random_feature.resize(
                current_feature.size, 
                Image.Resampling.LANCZOS
            )
            
            # Use current feature's alpha as mask for positioning
            mask = current_feature.split()[3]  # Alpha channel
            
            # Paste random feature onto corrupted using alpha compositing
            corrupted.paste(random_feature_resized, (0, 0), mask)
            
            actually_corrupted.append(feature_type)
            logger.debug(f"  Composited {feature_type} from {random_image_name}")
        
        return corrupted, target_img, actually_corrupted
    
    def visualize_corruption(self,
                            image_name: str,
                            corruption_level: int = 2,
                            save_path: Optional[str] = None) -> Image.Image:
        """Visualize: face_contour | corrupted | jawline with white background."""
        base_data = self.feature_index[image_name]
        
        # Load base face_contour (RGBA with holes)
        base_path = base_data["features"]["face_contour"]
        base_img = self.load_image(base_path, 'RGBA')
        
        # Create corrupted
        corrupted, target, features = self.create_corrupted_face(
            image_name, corruption_level
        )
        
        # Create white background for transparent images
        width, height = base_img.size
        white_bg = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Composite each image onto white background
        base_on_white = white_bg.copy()
        base_on_white.paste(base_img, (0, 0), base_img)
        
        corrupted_on_white = white_bg.copy()
        corrupted_on_white.paste(corrupted, (0, 0), corrupted)
        
        target_on_white = white_bg.copy()
        target_on_white.paste(target, (0, 0), target)
        
        # Create side-by-side visualization
        vis = Image.new('RGB', (width * 3, height), (255, 255, 255))
        vis.paste(base_on_white, (0, 0))
        vis.paste(corrupted_on_white, (width, 0))
        vis.paste(target_on_white, (width * 2, 0))
        
        if save_path:
            vis.save(save_path)
            logger.info(f"Saved visualization to {save_path}")
            logger.info(f"  Corrupted features: {features}")
        
        return vis


def test_corruption_engine():
    """Test corruption engine."""
    dataset_dir = "/home/teaching/G14/forensic_reconstruction/dataset"
    index_path = Path(dataset_dir) / "metadata" / "feature_index.json"
    
    if not index_path.exists():
        logger.error(f"Feature index not found: {index_path}")
        logger.error("Please run: python src/dataset_indexer.py")
        return
    
    with open(index_path) as f:
        feature_index = json.load(f)
    
    logger.info(f"Loaded feature index with {len(feature_index)} images")
    
    # Initialize engine
    engine = CorruptionEngine(feature_index, dataset_dir)
    
    # Test on first 3 images
    test_images = list(feature_index.keys())[:3]
    
    # Create visualizations directory
    vis_dir = Path(dataset_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for image_name in test_images:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {image_name}")
        logger.info(f"{'='*60}")
        
        # Test all levels
        for level in [1, 2, 3]:
            logger.info(f"\nCorruption level {level}:")
            
            try:
                corrupted, target, features = engine.create_corrupted_face(
                    image_name, corruption_level=level
                )
                
                logger.info(f"  Features ({len(features)}): {features}")
                
                # Save visualization
                vis_path = vis_dir / f"corruption_level_{level}_{image_name}.png"
                engine.visualize_corruption(image_name, level, str(vis_path))
                
            except Exception as e:
                logger.error(f"Failed: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Complete! Check: {vis_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    test_corruption_engine()