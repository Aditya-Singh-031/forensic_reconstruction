#!/usr/bin/env python3
"""
Test script for face inpainting module.

This script demonstrates how to:
  - Load a face image
  - Generate occlusion masks (using mask_generator)
  - Run inpainting with Stable Diffusion
  - Test with default and custom text prompts
  - Show before/after comparisons
  - Save results and measure performance

Usage:
  # Basic usage (uses default prompts)
  python test_inpainting.py --image path/to/image.jpg

  # Custom prompt for specific feature
  python test_inpainting.py --image path/to/image.jpg --mask_type mouth --prompt "wide smile, natural"

  # Custom inference parameters
  python test_inpainting.py --image path/to/image.jpg --steps 40 --guidance_scale 10.0

  # Batch mode (inpaint multiple masks)
  python test_inpainting.py --image path/to/image.jpg --batch

  # Use pre-computed mask file
  python test_inpainting.py --image path/to/image.jpg --mask path/to/mask.png
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from face_segmentation import create_segmenter, FaceSegmenter  # noqa: E402
from landmark_detector import create_detector, LandmarkDetector  # noqa: E402
from mask_generator import MaskGenerator  # noqa: E402
from face_inpainter import FaceInpainter, create_inpainter  # noqa: E402


def _ensure_output_dir(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _load_image(image_path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _load_mask(mask_path: str) -> np.ndarray:
    """Load mask as grayscale numpy array."""
    mask = Image.open(mask_path).convert("L")
    return np.array(mask)


def _create_side_by_side(
    original: np.ndarray,
    inpainted: Image.Image,
    mask: np.ndarray,
    title: str = "Before / After"
) -> Image.Image:
    """
    Create side-by-side comparison image.
    
    Args:
        original: Original image (numpy array, RGB)
        inpainted: Inpainted image (PIL Image)
        mask: Occlusion mask (numpy array, HxW)
        title: Title text for the comparison
    
    Returns:
        PIL Image with side-by-side comparison
    """
    # Convert inpainted to numpy
    inpainted_np = np.array(inpainted)
    
    # Ensure same height
    h_orig = original.shape[0]
    h_inp = inpainted_np.shape[0]
    
    if h_orig != h_inp:
        # Resize inpainted to match original
        inpainted_np = cv2.resize(inpainted_np, (original.shape[1], original.shape[0]))
    
    # Create overlay showing mask region
    overlay = original.copy()
    mask_bool = mask > 128  # Threshold mask
    overlay[mask_bool] = [255, 0, 0]  # Red overlay
    overlay = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    
    # Concatenate horizontally: original | overlay | inpainted
    comparison = np.hstack([original, overlay, inpainted_np])
    
    # Add title text
    comparison_pil = Image.fromarray(comparison)
    
    return comparison_pil


def _get_segmentation_and_landmarks(
    image_path: str,
    segmenter: Optional[FaceSegmenter],
    detector: Optional[LandmarkDetector],
) -> Tuple[np.ndarray, Dict[str, int], Dict]:
    """
    Get segmentation mask and landmarks for an image.
    
    Returns:
        Tuple of (segmentation_mask, component_ids, landmark_groups)
    """
    seg_mask = None
    comp_ids = {}
    landmarks = {}
    
    if segmenter:
        print("  Running segmentation...")
        result = segmenter.segment(
            image_path,
            return_masks=False,
            return_colored=False,
            return_statistics=False,
        )
        seg_mask = result['segmentation']
        comp_ids = segmenter.face_component_ids
    
    if detector:
        print("  Running landmark detection...")
        try:
            result = detector.detect(
                image_path,
                return_visualization=False,
                return_groups=True,
                return_coordinates=False,
            )
            landmarks = result.get('groups', {})
        except Exception as e:
            print(f"  Warning: Landmark detection failed: {e}")
            print("  Continuing with segmentation-only masks...")
    
    return seg_mask, comp_ids, landmarks


def test_single_inpainting(
    image_path: str,
    mask_path: Optional[str],
    mask_type: Optional[str],
    prompt: Optional[str],
    output_dir: Path,
    inpainter: FaceInpainter,
    segmenter: Optional[FaceSegmenter],
    detector: Optional[LandmarkDetector],
    num_steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> bool:
    """
    Test inpainting on a single image with one mask.
    
    Args:
        image_path: Path to input image
        mask_path: Optional path to pre-computed mask file
        mask_type: Type of mask to generate ('mouth', 'eyes', etc.)
        prompt: Custom text prompt (uses default if None)
        output_dir: Output directory
        inpainter: FaceInpainter instance
        segmenter: FaceSegmenter instance (for generating masks)
        detector: LandmarkDetector instance (for generating masks)
        num_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Inpainting test: {Path(image_path).name}")
    print(f"{'='*60}\n")
    
    try:
        # Load image
        print("Loading image...")
        image_np = _load_image(image_path)
        image_pil = Image.fromarray(image_np)
        h, w = image_np.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Get or generate mask
        if mask_path:
            print(f"Loading mask from: {mask_path}")
            mask = _load_mask(mask_path)
            mask_type_name = Path(mask_path).stem
        elif mask_type:
            print(f"Generating {mask_type} mask...")
            
            # Get segmentation and landmarks
            seg_mask, comp_ids, landmarks = _get_segmentation_and_landmarks(
                image_path, segmenter, detector
            )
            
            if seg_mask is None:
                print("  ✗ Error: Segmentation required but not available")
                return False
            
            # Generate mask
            mg = MaskGenerator(
                segmentation=seg_mask,
                component_ids=comp_ids,
                landmarks=landmarks if landmarks else None,
            )
            
            mask = mg.generate([mask_type], margin_px=4, feather_px=2)
            mask_type_name = mask_type
            
            # Save generated mask
            mask_dir = output_dir / "masks"
            mask_dir.mkdir(exist_ok=True)
            mask_img = Image.fromarray(mask, mode='L')
            mask_path_saved = mask_dir / f"mask_{mask_type_name}_{Path(image_path).stem}.png"
            mask_img.save(mask_path_saved)
            print(f"  ✓ Saved mask: {mask_path_saved}")
        else:
            print("  ✗ Error: Must specify either --mask or --mask_type")
            return False
        
        # Check mask validity
        mask_pixels = np.sum(mask > 128)
        total_pixels = mask.size
        mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        print(f"\nMask statistics:")
        print(f"  Masked pixels: {mask_pixels:,} ({mask_ratio*100:.2f}% of image)")
        
        if mask_pixels == 0:
            print("  ✗ Warning: Mask is empty (no pixels to inpaint)")
            return False
        
        if mask_ratio > 0.8:
            print("  ⚠ Warning: Mask covers >80% of image (may produce poor results)")
        
        # Get prompt
        if prompt is None:
            # Use default prompt for this feature type
            prompt = inpainter.get_default_prompt(mask_type_name)
            print(f"\nUsing default prompt for '{mask_type_name}':")
        else:
            print(f"\nUsing custom prompt:")
        
        print(f"  \"{prompt}\"")
        
        # Run inpainting
        print(f"\nRunning inpainting...")
        print(f"  Steps: {num_steps}, Guidance: {guidance_scale:.1f}")
        if seed is not None:
            print(f"  Seed: {seed}")
        
        start_time = time.time()
        
        result = inpainter.inpaint(
            image=image_pil,
            mask=mask,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            show_progress=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Inpainting completed in {elapsed:.2f} seconds")
        
        # Create side-by-side comparison
        print("\nCreating comparison image...")
        comparison = _create_side_by_side(image_np, result, mask, title=f"{mask_type_name} inpainting")
        
        # Save results
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save inpainted result
        result_path = results_dir / f"inpainted_{mask_type_name}_{Path(image_path).stem}.png"
        result.save(result_path)
        print(f"  ✓ Saved inpainted image: {result_path}")
        
        # Save comparison
        comp_path = results_dir / f"comparison_{mask_type_name}_{Path(image_path).stem}.png"
        comparison.save(comp_path)
        print(f"  ✓ Saved comparison: {comp_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("✓ Inpainting test completed successfully!")
        print(f"{'='*60}")
        print(f"  Input: {image_path}")
        print(f"  Mask type: {mask_type_name}")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  Output: {result_path}")
        print(f"  Comparison: {comp_path}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inpainting(
    image_path: str,
    output_dir: Path,
    inpainter: FaceInpainter,
    segmenter: Optional[FaceSegmenter],
    detector: Optional[LandmarkDetector],
    num_steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> bool:
    """
    Test inpainting multiple mask types on the same image.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        inpainter: FaceInpainter instance
        segmenter: FaceSegmenter instance
        detector: LandmarkDetector instance
        num_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Batch inpainting test: {Path(image_path).name}")
    print(f"{'='*60}\n")
    
    try:
        # Load image
        print("Loading image...")
        image_np = _load_image(image_path)
        image_pil = Image.fromarray(image_np)
        h, w = image_np.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Get segmentation and landmarks
        print("\nGenerating masks for multiple features...")
        seg_mask, comp_ids, landmarks = _get_segmentation_and_landmarks(
            image_path, segmenter, detector
        )
        
        if seg_mask is None:
            print("  ✗ Error: Segmentation required but not available")
            return False
        
        # Generate masks for multiple features
        mask_types = ['mouth', 'eyes', 'mustache', 'nose', 'hair']
        mg = MaskGenerator(
            segmentation=seg_mask,
            component_ids=comp_ids,
            landmarks=landmarks if landmarks else None,
        )
        
        print(f"  Generating {len(mask_types)} masks...")
        masks = {}
        for mask_type in mask_types:
            try:
                mask = mg.generate([mask_type], margin_px=4, feather_px=2)
                if np.sum(mask > 128) > 0:  # Only include non-empty masks
                    masks[mask_type] = mask
                    print(f"    ✓ {mask_type}: {np.sum(mask > 128):,} pixels")
                else:
                    print(f"    ⚠ {mask_type}: empty mask (skipped)")
            except Exception as e:
                print(f"    ✗ {mask_type}: failed ({e})")
        
        if not masks:
            print("  ✗ Error: No valid masks generated")
            return False
        
        print(f"\n  Generated {len(masks)} valid masks")
        
        # Create prompts for each mask type
        prompts = {}
        for mask_type in masks.keys():
            prompts[mask_type] = inpainter.get_default_prompt(mask_type)
        
        # Run batch inpainting
        print(f"\nRunning batch inpainting ({len(masks)} masks)...")
        print(f"  Steps: {num_steps}, Guidance: {guidance_scale:.1f}")
        if seed is not None:
            print(f"  Seed: {seed} (will increment for each mask)")
        
        start_time = time.time()
        
        results = inpainter.inpaint_batch(
            image=image_pil,
            masks=masks,
            prompts=prompts,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            show_progress=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Batch inpainting completed in {elapsed:.2f} seconds")
        print(f"  Average time per mask: {elapsed/len(masks):.2f} seconds")
        
        # Save all results
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving results...")
        for mask_type, result in results.items():
            # Save inpainted image
            result_path = results_dir / f"inpainted_{mask_type}_{Path(image_path).stem}.png"
            result.save(result_path)
            print(f"  ✓ {mask_type}: {result_path}")
            
            # Save mask
            mask_img = Image.fromarray(masks[mask_type], mode='L')
            mask_path = masks_dir / f"mask_{mask_type}_{Path(image_path).stem}.png"
            mask_img.save(mask_path)
            
            # Create and save comparison
            comparison = _create_side_by_side(image_np, result, masks[mask_type])
            comp_path = results_dir / f"comparison_{mask_type}_{Path(image_path).stem}.png"
            comparison.save(comp_path)
        
        print(f"\n{'='*60}")
        print("✓ Batch inpainting test completed successfully!")
        print(f"{'='*60}")
        print(f"  Processed {len(results)} masks")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Results saved to: {results_dir}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during batch inpainting: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_installation() -> bool:
    """Verify that all required packages are installed."""
    print("Verifying installation for inpainting...")
    
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
        from diffusers import StableDiffusionInpaintPipeline
        print("  ✓ Diffusers library installed")
    except ImportError:
        print("  ✗ Diffusers library not installed")
        print("    Install with: pip install diffusers transformers accelerate")
        return False
    
    try:
        import numpy
        print("  ✓ NumPy installed")
    except ImportError:
        print("  ✗ NumPy not installed")
        return False
    
    try:
        from PIL import Image
        print("  ✓ Pillow (PIL) installed")
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
        from tqdm import tqdm
        print("  ✓ tqdm installed (progress bars)")
    except ImportError:
        print("  ⚠ tqdm not installed (progress bars disabled)")
    
    print("\n✓ All required packages are installed!\n")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test face inpainting with Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (generate mask automatically)
  python test_inpainting.py --image testimage.jpeg --mask_type mouth

  # Custom prompt
  python test_inpainting.py --image testimage.jpeg --mask_type mustache \\
      --prompt "thick black mustache, detailed, realistic"

  # Use pre-computed mask
  python test_inpainting.py --image testimage.jpeg --mask output/masks/mask_mouth.png

  # Batch mode (inpaint multiple features)
  python test_inpainting.py --image testimage.jpeg --batch

  # High quality (more steps)
  python test_inpainting.py --image testimage.jpeg --mask_type eyes --steps 50 --guidance_scale 10.0

  # Reproducible results
  python test_inpainting.py --image testimage.jpeg --mask_type mouth --seed 42
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        help='Path to pre-computed occlusion mask (PNG, grayscale)'
    )
    
    parser.add_argument(
        '--mask_type',
        type=str,
        choices=['eyes', 'eyebrows', 'mouth', 'mustache', 'nose', 'hair', 'upper_face', 'lower_face'],
        help='Type of mask to generate (requires segmentation/landmarks)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Custom text prompt for inpainting (e.g., "thick black mustache, detailed")'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: inpaint multiple mask types on the same image'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/inpainting',
        help='Output directory for results (default: output/inpainting)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of inference steps (20-50, default: 30)'
    )
    
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=7.5,
        help='Guidance scale (7.5-15, default: 7.5)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (default: random)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--no_segmentation',
        action='store_true',
        help='Skip segmentation (only works with --mask)'
    )
    
    parser.add_argument(
        '--no_landmarks',
        action='store_true',
        help='Skip landmark detection (masks will use segmentation only)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify installation, do not run inpainting'
    )
    
    args = parser.parse_args()
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed. Please install missing packages.")
        return 1
    
    if args.verify:
        print("\n✓ Installation verified. Ready to use!")
        return 0
    
    # Validate arguments
    if not args.mask and not args.mask_type and not args.batch:
        parser.print_help()
        print("\n✗ Error: Must specify either --mask, --mask_type, or --batch")
        return 1
    
    if args.mask and args.mask_type:
        print("⚠ Warning: Both --mask and --mask_type specified. Using --mask.")
    
    # Determine device
    device = None if args.device == 'auto' else args.device
    
    # Initialize models
    print("\nInitializing models...")
    
    # Initialize inpainter
    try:
        inpainter = create_inpainter(device=device, use_half_precision=True)
        print("  ✓ FaceInpainter ready")
    except Exception as e:
        print(f"\n✗ Failed to initialize FaceInpainter: {e}")
        print("  Tip: First-time model download may take 10-30 minutes")
        return 1
    
    # Initialize segmenter and detector (if needed)
    segmenter = None
    detector = None
    
    if not args.no_segmentation and (args.mask_type or args.batch):
        try:
            from face_segmentation import create_segmenter
            segmenter = create_segmenter(device=device)
            print("  ✓ FaceSegmenter ready")
        except Exception as e:
            print(f"  ⚠ FaceSegmenter failed: {e}")
            print("  Continuing without segmentation (masks may be limited)")
    
    if not args.no_landmarks and (args.mask_type or args.batch):
        try:
            from landmark_detector import create_detector
            detector = create_detector()
            print("  ✓ LandmarkDetector ready")
        except Exception as e:
            print(f"  ⚠ LandmarkDetector failed: {e}")
            print("  Continuing without landmarks (masks will use segmentation only)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)
    
    # Run test
    if args.batch:
        success = test_batch_inpainting(
            image_path=args.image,
            output_dir=output_dir,
            inpainter=inpainter,
            segmenter=segmenter,
            detector=detector,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    else:
        success = test_single_inpainting(
            image_path=args.image,
            mask_path=args.mask,
            mask_type=args.mask_type,
            prompt=args.prompt,
            output_dir=output_dir,
            inpainter=inpainter,
            segmenter=segmenter,
            detector=detector,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
