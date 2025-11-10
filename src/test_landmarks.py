#!/usr/bin/env python3
"""
Test script for facial landmark detection module.

This script tests the LandmarkDetector class with sample images and
displays/saves the results.

Usage:
    python test_landmarks.py --image path/to/image.jpg
    python test_landmarks.py --image path/to/image.jpg --output_dir output/
    python test_landmarks.py --batch path/to/images/ --output_dir output/
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from landmark_detector import LandmarkDetector, create_detector

def test_single_image(image_path: str, output_dir: Path, detector: LandmarkDetector):
    """
    Test landmark detection on a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        detector: LandmarkDetector instance
    """
    print(f"\n{'='*60}")
    print(f"Testing landmark detection on: {image_path}")
    print(f"{'='*60}\n")
    
    try:
        # Run detection
        print("Running landmark detection...")
        result = detector.detect(
            image_path,
            return_visualization=True,
            return_groups=True,
            return_coordinates=True
        )
        
        # Display results
        print(f"\n✓ Landmark detection completed!")
        print(f"  Processing time: {result['processing_time']:.2f} seconds")
        print(f"  Image size: {result['image_size'][0]}x{result['image_size'][1]}")
        print(f"  Faces detected: {result['num_faces']}")
        if result['num_faces'] > 1:
            print(f"  Selected face: {result['face_index'] + 1} (largest)")
        print(f"  Total landmarks: {len(result['landmarks'])}")
        
        # Display feature groups
        if 'groups' in result:
            print(f"\nFeature Groups:")
            print(f"  {'Feature':<20} {'Landmarks':<15} {'BBox':<30}")
            print(f"  {'-'*65}")
            for feature_name, group_data in result['groups'].items():
                if group_data['bbox']:
                    bbox = group_data['bbox']
                    bbox_str = f"({bbox['x_min']},{bbox['y_min']})-({bbox['x_max']},{bbox['y_max']})"
                    print(f"  {feature_name:<20} {group_data['count']:<15} {bbox_str:<30}")
        
        # Display sample coordinates
        print(f"\nSample Landmark Coordinates (first 5):")
        print(f"  {'Index':<10} {'X (px)':<15} {'Y (px)':<15} {'Z (px)':<15}")
        print(f"  {'-'*55}")
        for i in range(min(5, len(result['landmarks']))):
            x, y, z = result['landmarks'][i]
            print(f"  {i:<10} {x:<15.2f} {y:<15.2f} {z:<15.2f}")
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        vis_path = output_dir / f"landmarks_{Path(image_path).stem}.png"
        result['visualization'].save(vis_path)
        print(f"\n✓ Saved visualization: {vis_path}")
        
        # Save coordinates as NumPy array
        coords_path = output_dir / f"landmarks_coords_{Path(image_path).stem}.npy"
        np.save(coords_path, result['landmarks'])
        print(f"✓ Saved coordinates (NumPy): {coords_path}")
        
        # Save coordinates as CSV
        csv_path = output_dir / f"landmarks_coords_{Path(image_path).stem}.csv"
        np.savetxt(csv_path, result['landmarks'], delimiter=',', fmt='%.2f',
                   header='x,y,z', comments='')
        print(f"✓ Saved coordinates (CSV): {csv_path}")
        
        # Save feature groups info
        if 'groups' in result:
            groups_dir = output_dir / "feature_groups"
            groups_dir.mkdir(exist_ok=True)
            
            for feature_name, group_data in result['groups'].items():
                if len(group_data['landmarks_pixel']) > 0:
                    feature_coords = np.array(group_data['landmarks_pixel'])
                    feature_path = groups_dir / f"{feature_name}_{Path(image_path).stem}.npy"
                    np.save(feature_path, feature_coords)
            
            print(f"✓ Saved feature group coordinates: {groups_dir}/")
        
        # Print summary statistics
        print(f"\nLandmark Statistics:")
        landmarks = result['landmarks']
        print(f"  X range: [{landmarks[:, 0].min():.2f}, {landmarks[:, 0].max():.2f}]")
        print(f"  Y range: [{landmarks[:, 1].min():.2f}, {landmarks[:, 1].max():.2f}]")
        print(f"  Z range: [{landmarks[:, 2].min():.2f}, {landmarks[:, 2].max():.2f}]")
        print(f"  Mean position: ({landmarks[:, 0].mean():.2f}, {landmarks[:, 1].mean():.2f})")
        
        print(f"\n{'='*60}")
        print("✓ All outputs saved successfully!")
        print(f"{'='*60}\n")
        
        return True
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("  Tips:")
        print("    - Ensure image contains a clear face")
        print("    - Try a different image")
        print("    - Check image is not corrupted")
        return False
    except Exception as e:
        print(f"\n✗ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_images(image_dir: str, output_dir: Path, detector: LandmarkDetector):
    """
    Test landmark detection on multiple images.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save outputs
        detector: LandmarkDetector instance
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
        if test_single_image(str(image_path), output_dir, detector):
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
        import mediapipe as mp
        print(f"  ✓ MediaPipe {mp.__version__} installed")
    except ImportError:
        print("  ✗ MediaPipe not installed")
        print("    Install with: pip install mediapipe")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy installed")
    except ImportError:
        print("  ✗ NumPy not installed")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__} installed")
    except ImportError:
        print("  ✗ OpenCV not installed")
        return False
    
    try:
        from PIL import Image
        print(f"  ✓ Pillow (PIL) installed")
    except ImportError:
        print("  ✗ Pillow not installed")
        return False
    
    print("\n✓ All required packages are installed!\n")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test facial landmark detection module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test_landmarks.py --image sample_face.jpg
  
  # Test with custom output directory
  python test_landmarks.py --image sample_face.jpg --output_dir output/
  
  # Test batch of images
  python test_landmarks.py --batch images/ --output_dir output/
  
  # Allow multiple faces (will still pick largest)
  python test_landmarks.py --image sample_face.jpg --max_faces 5
  
  # Disable landmark refinement (faster)
  python test_landmarks.py --image sample_face.jpg --no_refine
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
        default='output/landmarks',
        help='Output directory for results (default: output/landmarks)'
    )
    
    parser.add_argument(
        '--max_faces',
        type=int,
        default=1,
        help='Maximum number of faces to detect (default: 1)'
    )
    
    parser.add_argument(
        '--no_refine',
        action='store_true',
        help='Disable landmark refinement (faster but less accurate)'
    )
    
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence [0.0, 1.0] (default: 0.5)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify installation, do not run detection'
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
    
    # Create detector
    print(f"\nInitializing LandmarkDetector...")
    print(f"  Max faces: {args.max_faces}")
    print(f"  Refine landmarks: {not args.no_refine}")
    print(f"  Min confidence: {args.min_confidence}")
    
    try:
        detector = create_detector(
            static_image_mode=True,
            max_num_faces=args.max_faces,
            refine_landmarks=not args.no_refine
        )
        print("✓ LandmarkDetector initialized successfully\n")
    except Exception as e:
        print(f"\n✗ Failed to initialize LandmarkDetector: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    if args.image:
        success = test_single_image(args.image, output_dir, detector)
        return 0 if success else 1
    elif args.batch:
        test_batch_images(args.batch, output_dir, detector)
        return 0

if __name__ == "__main__":
    sys.exit(main())

