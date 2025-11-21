"""
Corruption Engine Test - WITH IMAGE SELECTION
Test corruption engine on specific images or random selection
"""

import json
import logging
import random
from pathlib import Path
import argparse

# Import the CorruptionEngine
from corruption_engine import CorruptionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_corruption_engine(dataset_dir: str = "/home/teaching/G14/forensic_reconstruction/dataset",
                          test_image: str = None,
                          num_random: int = 1):
    """Test corruption engine with image selection.
    
    Args:
        dataset_dir: Dataset root directory
        test_image: Specific image name to test (e.g., "00001"), or None for random
        num_random: Number of random images to test if test_image is None
    """
    # Load feature index
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
    
    # Create visualizations directory
    vis_dir = Path(dataset_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Determine which images to test
    if test_image:
        # Test specific image
        if test_image not in feature_index:
            logger.error(f"Image {test_image} not found in index!")
            logger.info(f"Available images: {list(feature_index.keys())[:10]}...")
            return
        test_images = [test_image]
    else:
        # Test random images
        all_images = list(feature_index.keys())
        test_images = random.sample(all_images, min(num_random, len(all_images)))
    
    # Test each image
    for image_name in test_images:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing corruption on: {image_name}")
        logger.info(f"{'='*60}")
        
        # Check what files exist for this image
        image_data = feature_index[image_name]
        logger.info(f"Image {image_name} has:")
        logger.info(f"  face_contour: {image_data['features'].get('face_contour', 'MISSING')}")
        logger.info(f"  jawline: {image_data['features'].get('jawline', 'MISSING')}")
        
        # Test all corruption levels
        for level in [1, 2, 3]:
            logger.info(f"\n{'-'*60}")
            logger.info(f"Corruption level {level}:")
            logger.info(f"{'-'*60}")
            
            try:
                corrupted, target, features = engine.create_corrupted_face(
                    image_name, corruption_level=level
                )
                
                logger.info(f"  Corrupted features ({len(features)}): {features}")
                logger.info(f"  Corrupted size: {corrupted.size}")
                logger.info(f"  Target size: {target.size}")
                
                # Save visualization
                vis_path = vis_dir / f"corruption_level_{level}_{image_name}.png"
                engine.visualize_corruption(image_name, level, str(vis_path))
                
            except Exception as e:
                logger.error(f"Failed level {level} for {image_name}: {e}")
                import traceback
                traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Corruption test complete!")
    logger.info(f"Check visualizations in: {vis_dir}")
    logger.info(f"{'='*60}\n")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Test corruption engine")
    parser.add_argument("--dataset-dir", type=str,
                       default="/home/teaching/G14/forensic_reconstruction/dataset",
                       help="Dataset root directory")
    parser.add_argument("--image", type=str, default=None,
                       help="Specific image name to test (e.g., '00001')")
    parser.add_argument("--num-random", type=int, default=3,
                       help="Number of random images to test (if --image not specified)")
    parser.add_argument("--list-images", action='store_true',
                       help="List available images and exit")
    
    args = parser.parse_args()
    
    # List images if requested
    if args.list_images:
        index_path = Path(args.dataset_dir) / "metadata" / "feature_index.json"
        with open(index_path) as f:
            feature_index = json.load(f)
        
        logger.info(f"Available images ({len(feature_index)} total):")
        for i, img_name in enumerate(sorted(feature_index.keys())[:20]):
            logger.info(f"  {img_name}")
        if len(feature_index) > 20:
            logger.info(f"  ... and {len(feature_index) - 20} more")
        return
    
    # Run test
    test_corruption_engine(
        dataset_dir=args.dataset_dir,
        test_image=args.image,
        num_random=args.num_random
    )


if __name__ == "__main__":
    main()