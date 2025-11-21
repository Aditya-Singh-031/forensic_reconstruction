"""
Corruption Engine - Phase 2
Creates corrupted faces by randomly compositing features
Implements curriculum learning corruption levels
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorruptionEngine:
    """Generate corrupted faces for training."""
    
    # Feature types that can be corrupted
    CORRUPTIBLE_FEATURES = [
        "eyes_left",
        "eyes_right",
        "eyebrows_left", 
        "eyebrows_right",
        "nose",
        "mouth_outer",
        "mouth_inner"
    ]
    
    def __init__(self, feature_index: Dict, seed: int = 42):
        """Initialize corruption engine.
        
        Args:
            feature_index: Feature index from dataset_indexer
            seed: Random seed for reproducibility
        """
        self.feature_index = feature_index
        self.image_names = list(feature_index.keys())
        
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Initialized CorruptionEngine with {len(self.image_names)} images")
    
    def get_corruption_level_features(self, level: int) -> List[str]:
        """Get features to corrupt for given level.
        
        Args:
            level: Corruption level (1=easy, 2=medium, 3=hard)
            
        Returns:
            features: List of feature types to corrupt
        """
        if level == 1:
            # Easy: 2-3 features
            num_features = random.randint(2, 3)
        elif level == 2:
            # Medium: 4-5 features
            num_features = random.randint(4, 5)
        elif level == 3:
            # Hard: 6-7 features
            num_features = random.randint(6, 7)
        else:
            # Default to medium
            num_features = 4
        
        # Randomly select features to corrupt
        features = random.sample(self.CORRUPTIBLE_FEATURES, 
                                min(num_features, len(self.CORRUPTIBLE_FEATURES)))
        return features
    
    def load_image_rgba(self, path: str) -> Optional[Image.Image]:
        """Load image and ensure RGBA format.
        
        Args:
            path: Image path
            
        Returns:
            img: PIL Image in RGBA format, or None if load fails
        """
        try:
            img = Image.open(path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def composite_feature(self, 
                         base_img: Image.Image,
                         feature_img: Image.Image,
                         feature_mask: Optional[Image.Image] = None) -> Image.Image:
        """Composite feature onto base image using alpha channel.
        
        Args:
            base_img: Base image (face_contour)
            feature_img: Feature to composite (has alpha channel)
            feature_mask: Optional mask (not used if feature has alpha)
            
        Returns:
            result: Composited image
        """
        # Convert to numpy for easier manipulation
        base_array = np.array(base_img)
        feature_array = np.array(feature_img)
        
        # If feature has alpha channel, use it
        if feature_array.shape[2] == 4:
            alpha = feature_array[:, :, 3:4] / 255.0
            
            # Resize alpha to match base if needed
            if base_array.shape[:2] != feature_array.shape[:2]:
                # Feature is cropped, we need position info
                # For now, assume same size (will handle positioning in data_loader)
                return base_img
            
            # Blend using alpha
            result_array = base_array.copy()
            result_array[:, :, :3] = (
                alpha * feature_array[:, :, :3] + 
                (1 - alpha) * base_array[:, :, :3]
            ).astype(np.uint8)
            
            return Image.fromarray(result_array, mode='RGBA')
        
        # Fallback: direct paste
        result = base_img.copy()
        result.paste(feature_img, (0, 0), feature_img if feature_img.mode == 'RGBA' else None)
        return result
    
    def create_corrupted_face(self,
                             image_name: str,
                             corruption_level: int = 2,
                             spatial_augment: bool = False) -> Tuple[Image.Image, Image.Image, List[str]]:
        """Create corrupted face by random feature compositing.
        
        Args:
            image_name: Base image name
            corruption_level: Level of corruption (1-3)
            spatial_augment: Apply spatial augmentations
            
        Returns:
            corrupted_face: Corrupted image
            target_face: Ground truth (jawline)
            corrupted_features: List of corrupted feature names
        """
        # Load base face_contour
        base_data = self.feature_index[image_name]
        base_path = base_data["features"]["face_contour"]
        base_img = self.load_image_rgba(base_path)
        
        if base_img is None:
            raise ValueError(f"Failed to load base image for {image_name}")
        
        # Load target (jawline)
        target_path = base_data["features"]["jawline"]
        target_img = self.load_image_rgba(target_path)
        
        if target_img is None:
            raise ValueError(f"Failed to load target image for {image_name}")
        
        # Determine which features to corrupt
        features_to_corrupt = self.get_corruption_level_features(corruption_level)
        
        # Start with base image
        corrupted = base_img.copy()
        
        # Composite each corrupted feature
        for feature_type in features_to_corrupt:
            # Sample random image for this feature
            random_image_name = random.choice(self.image_names)
            random_data = self.feature_index[random_image_name]
            
            # Check if this random image has the feature
            if feature_type not in random_data["features"]:
                continue
            
            # Load random feature
            feature_path = random_data["features"][feature_type]
            feature_img = self.load_image_rgba(feature_path)
            
            if feature_img is None:
                continue
            
            # Load mask if available
            mask_img = None
            if feature_type in random_data["masks"]:
                mask_path = random_data["masks"][feature_type]
                mask_img = self.load_image_rgba(mask_path)
            
            # Composite onto corrupted image
            # Note: This is simplified - full version needs position info from landmarks
            # For now, we assume features are already positioned correctly
            corrupted = self.composite_feature(corrupted, feature_img, mask_img)
        
        return corrupted, target_img, features_to_corrupt
    
    def create_corruption_mask(self, 
                               image_size: Tuple[int, int],
                               corrupted_features: List[str]) -> Image.Image:
        """Create binary mask showing which regions were corrupted.
        
        Args:
            image_size: (width, height)
            corrupted_features: List of corrupted feature names
            
        Returns:
            mask: Binary mask (white=corrupted, black=original)
        """
        # Simplified: Return all-ones mask for now
        # Full version would use actual feature positions
        mask = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
        return Image.fromarray(mask, mode='L')
    
    def visualize_corruption(self,
                            image_name: str,
                            corruption_level: int = 2,
                            save_path: Optional[str] = None) -> Image.Image:
        """Visualize corruption process for debugging.
        
        Args:
            image_name: Image to corrupt
            corruption_level: Corruption level
            save_path: Optional path to save visualization
            
        Returns:
            vis: Visualization image (base | corrupted | target)
        """
        # Load base
        base_data = self.feature_index[image_name]
        base_img = self.load_image_rgba(base_data["features"]["face_contour"])
        
        # Create corrupted
        corrupted, target, features = self.create_corrupted_face(
            image_name, corruption_level
        )
        
        # Create side-by-side visualization
        width = base_img.size[0]
        height = base_img.size[1]
        
        vis = Image.new('RGB', (width * 3, height))
        vis.paste(base_img.convert('RGB'), (0, 0))
        vis.paste(corrupted.convert('RGB'), (width, 0))
        vis.paste(target.convert('RGB'), (width * 2, 0))
        
        if save_path:
            vis.save(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        return vis


def test_corruption_engine():
    """Test corruption engine with sample data."""
    import json
    
    # Load feature index
    index_path = "/DATA/facial_features_dataset/metadata/feature_index.json"
    
    if not Path(index_path).exists():
        logger.error(f"Feature index not found: {index_path}")
        logger.error("Please run dataset_indexer.py first")
        return
    
    with open(index_path) as f:
        feature_index = json.load(f)
    
    logger.info(f"Loaded feature index with {len(feature_index)} images")
    
    # Initialize engine
    engine = CorruptionEngine(feature_index)
    
    # Test corruption on first image
    image_name = list(feature_index.keys())[0]
    logger.info(f"\nTesting corruption on: {image_name}")
    
    # Test all corruption levels
    for level in [1, 2, 3]:
        logger.info(f"\nCorruption level {level}:")
        
        corrupted, target, features = engine.create_corrupted_face(
            image_name, corruption_level=level
        )
        
        logger.info(f"  Corrupted features: {features}")
        logger.info(f"  Corrupted image size: {corrupted.size}")
        logger.info(f"  Target image size: {target.size}")
        
        # Save visualization
        vis_dir = Path("/DATA/facial_features_dataset/visualizations")
        vis_dir.mkdir(exist_ok=True)
        
        vis_path = vis_dir / f"corruption_level_{level}_{image_name}.png"
        engine.visualize_corruption(image_name, level, str(vis_path))


if __name__ == "__main__":
    test_corruption_engine()
