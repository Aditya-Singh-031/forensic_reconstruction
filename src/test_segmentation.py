#!/usr/bin/env python3
"""
Test script for face segmentation module.

This script tests the FaceSegmenter class with sample images and
displays/saves the results.

Usage:
    python test_segmentation.py --image path/to/image.jpg
    python test_segmentation.py --image path/to/image.jpg --output_dir output/
    python test_segmentation.py --batch path/to/images/ --output_dir output/
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from face_segmentation import FaceSegmenter, create_segmenter

def test_single_image(image_path: str, output_dir: Path, segmenter: FaceSegmenter):
    """
    Test segmentation on a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        segmenter: FaceSegmenter instance
    """
    print(f"\n{'='*60}")
    print(f"Testing segmentation on: {image_path}")
    print(f"{'='*60}\n")
    
    try:
        # Run segmentation
        print("Running segmentation...")
        result = segmenter.segment(
            image_path,
            return_masks=True,
            return_colored=True,
            return_statistics=True
        )
        
        # Display results
        print(f"\n✓ Segmentation completed!")
        print(f"  Processing time: {result['processing_time']:.2f} seconds")
        print(f"  Original size: {result['original_size'][0]}x{result['original_size'][1]}")
        
        # Display statistics
        if 'statistics' in result:
            print(f"\nComponent Statistics:")
            print(f"  {'Component':<20} {'Pixels':<15} {'Percentage':<10}")
            print(f"  {'-'*45}")
            for component, stats in result['statistics'].items():
                if stats['pixel_count'] > 0:  # Only show non-zero components
                    print(f"  {component:<20} {stats['pixel_count']:<15} {stats['percentage']:<10.2f}%")
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save colored visualization
        colored_path = output_dir / f"segmentation_colored_{Path(image_path).stem}.png"
        result['colored'].save(colored_path)
        print(f"\n✓ Saved colored visualization: {colored_path}")
        
        # Save individual masks
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        if 'masks' in result:
            print(f"\nSaving individual component masks...")
            for component_name, mask in result['masks'].items():
                if mask.sum() > 0:  # Only save non-empty masks
                    from PIL import Image
                    mask_img = Image.fromarray(mask, mode='L')
                    mask_path = masks_dir / f"{component_name}_{Path(image_path).stem}.png"
                    mask_img.save(mask_path)
                    print(f"  ✓ {component_name}: {mask_path}")
        
        # Save raw segmentation mask (optional)
        import numpy as np
        from PIL import Image
        seg_img = Image.fromarray(result['segmentation'], mode='L')
        seg_path = output_dir / f"segmentation_raw_{Path(image_path).stem}.png"
        seg_img.save(seg_path)
        print(f"  ✓ Raw mask: {seg_path}")
        
        print(f"\n{'='*60}")
        print("✓ All outputs saved successfully!")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_images(image_dir: str, output_dir: Path, segmenter: FaceSegmenter):
    """
    Test segmentation on multiple images.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save outputs
        segmenter: FaceSegmenter instance
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"✗ Error: Directory not found: {image_dir}")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"✗ Error: No image files found in {image_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Batch Processing: {len(image_files)} images")
    print(f"{'='*60}\n")
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        if test_single_image(str(image_path), output_dir, segmenter):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    print(f"{'='*60}\n")

def verify_installation():
    """Verify that all required packages are installed."""
    print("Verifying installation...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False
    
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print("  ✓ Transformers library installed")
    except ImportError:
        print("  ✗ Transformers library not installed")
        print("    Install with: pip install transformers")
        return False
    
    try:
        import PIL
        print(f"  ✓ Pillow (PIL) installed")
    except ImportError:
        print("  ✗ Pillow not installed")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__} installed")
    except ImportError:
        print("  ✗ OpenCV not installed")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy installed")
    except ImportError:
        print("  ✗ NumPy not installed")
        return False
    
    print("\n✓ All required packages are installed!\n")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test face segmentation module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test_segmentation.py --image sample_face.jpg
  
  # Test with custom output directory
  python test_segmentation.py --image sample_face.jpg --output_dir output/
  
  # Test batch of images
  python test_segmentation.py --batch images/ --output_dir output/
  
  # Use CPU instead of GPU
  python test_segmentation.py --image sample_face.jpg --device cpu
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to directory containing images for batch processing'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/segmentation',
        help='Output directory for results (default: output/segmentation)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='nvidia/segformer-b0-finetuned-ade-512-512',
        help='SegFormer model name (default: nvidia/segformer-b0-finetuned-ade-512-512)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify installation, do not run segmentation'
    )
    
    args = parser.parse_args()
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed. Please install missing packages.")
        return 1
    
    if args.verify:
        print("\n✓ Installation verified. Ready to use!")
        return 0
    
    # Check arguments
    if not args.image and not args.batch:
        parser.print_help()
        print("\n✗ Error: Must specify either --image or --batch")
        return 1
    
    # Determine device
    device = None if args.device == 'auto' else args.device
    
    # Create segmenter
    print(f"\nInitializing FaceSegmenter...")
    print(f"  Model: {args.model}")
    print(f"  Device: {device or 'auto-detect'}")
    
    try:
        segmenter = create_segmenter(model_name=args.model, device=device)
        print("✓ FaceSegmenter initialized successfully\n")
    except Exception as e:
        print(f"\n✗ Failed to initialize FaceSegmenter: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    if args.image:
        success = test_single_image(args.image, output_dir, segmenter)
        return 0 if success else 1
    elif args.batch:
        test_batch_images(args.batch, output_dir, segmenter)
        return 0

if __name__ == "__main__":
    sys.exit(main())

