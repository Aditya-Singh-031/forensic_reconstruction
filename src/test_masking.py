#!/usr/bin/env python3
"""
Test script for occlusion mask generation.

This script demonstrates how to:
  - Load a face image
  - Obtain segmentation and landmarks (or accept precomputed files)
  - Generate occlusion masks for multiple face components
  - Save masks and overlays
  - Print summary of masked pixels/regions and timing

Usage:
  # Minimal (compute segmentation + landmarks automatically)
  python test_masking.py --image path/to/image.jpg

  # Custom output dir and feathering
  python test_masking.py --image path/to/image.jpg --output_dir output/masking --feather 3

  # Batch mode (directory of images)
  python test_masking.py --batch images/ --output_dir output/masking
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
from mask_generator import MaskGenerator, PRESETS, RegionSpec  # noqa: E402


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _overlay_mask_on_image(image: np.ndarray, mask255: np.ndarray, color=(0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """
    Overlay mask (red by default) on image for visualization.
    image: RGB uint8 (H, W, 3)
    mask255: uint8 (H, W) where 255 means occluded.
    """
    overlay = image.copy()
    color_img = np.zeros_like(image)
    color_img[:, :] = color
    mask_bool = mask255 > 0
    overlay = cv2.addWeighted(overlay, 1.0, color_img, alpha, 0)
    result = image.copy()
    result[mask_bool] = overlay[mask_bool]
    return result


def _load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _get_segmentation(image_path: str, seg: Optional[FaceSegmenter]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Run segmentation and return (segmentation_mask, component_ids).
    """
    result = seg.segment(image_path, return_masks=False, return_colored=False, return_statistics=False)
    return result['segmentation'], seg.face_component_ids


def _get_landmarks(image_path: str, det: Optional[LandmarkDetector]) -> Dict:
    """
    Run landmark detection and return groups dict.
    """
    res = det.detect(image_path, return_visualization=False, return_groups=True, return_coordinates=False)
    return res.get('groups', {})


def generate_and_save_masks(
    image_path: str,
    output_dir: Path,
    segmenter: FaceSegmenter,
    detector: LandmarkDetector,
    feather_px: int,
    default_margin: int,
) -> bool:
    """
    Generate masks for all requested features and save outputs.
    """
    print(f"\n{'='*60}")
    print(f"Generating masks for: {image_path}")
    print(f"{'='*60}")

    t0 = time.time()
    img_rgb = _load_image(image_path)
    h, w = img_rgb.shape[:2]

    # Segmentation
    t_seg0 = time.time()
    seg_mask, comp_ids = _get_segmentation(image_path, segmenter)
    t_seg = time.time() - t_seg0

    # Landmarks
    t_lm0 = time.time()
    groups = {}
    try:
        groups = _get_landmarks(image_path, detector)
    except Exception as e:
        print(f"  Warning: landmarks not available ({e}). Proceeding with segmentation-only.")
    t_lm = time.time() - t_lm0

    # Create generator
    mg = MaskGenerator(segmentation=seg_mask, component_ids=comp_ids, landmarks=groups)

    # Define the set of masks to create (one per feature)
    # Using presets + side-specific eyes/eyebrows
    requests = {
        'eyes_left': [RegionSpec('eyes', side='left', margin_px=default_margin)],
        'eyes_right': [RegionSpec('eyes', side='right', margin_px=default_margin)],
        'eyes_both': PRESETS['eyes_both'],
        'eyebrows_left': [RegionSpec('eyebrows', side='left', margin_px=default_margin)],
        'eyebrows_right': [RegionSpec('eyebrows', side='right', margin_px=default_margin)],
        'eyebrows_both': PRESETS['eyebrows_both'],
        'mouth': PRESETS['mouth'],
        'mustache': PRESETS['mustache'],
        'nose': PRESETS['nose'],
        'hair_top': PRESETS['hair_top'],
        'upper_face': PRESETS['upper_face'],
        'lower_face': PRESETS['lower_face'],
    }

    # Generate all masks (batch API)
    t_gen0 = time.time()
    masks = mg.generate_batch(requests, margin_px=default_margin, feather_px=feather_px, hair_top_ratio=0.5)
    t_gen = time.time() - t_gen0
    total_time = time.time() - t0

    # Save outputs
    _ensure_output_dir(output_dir)
    base = Path(image_path).stem
    img_out = Image.fromarray(img_rgb)
    img_path_out = output_dir / f"image_{base}.png"
    img_out.save(img_path_out)

    print(f"\nSaved base image: {img_path_out}")
    print(f"Segmentation time: {t_seg:.2f}s, Landmarks time: {t_lm:.2f}s, Mask gen: {t_gen:.2f}s, Total: {total_time:.2f}s")

    # Save individual masks and overlays
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    _ensure_output_dir(masks_dir)
    _ensure_output_dir(overlays_dir)

    for name, m in masks.items():
        # Save mask
        mask_img = Image.fromarray(m, mode='L')
        mask_path = masks_dir / f"{name}_{base}.png"
        mask_img.save(mask_path)

        # Save overlay
        overlay = _overlay_mask_on_image(img_rgb, m, color=(255, 0, 0), alpha=0.45)
        overlay_img = Image.fromarray(overlay)
        overlay_path = overlays_dir / f"{name}_overlay_{base}.png"
        overlay_img.save(overlay_path)

        # Print stats
        num_masked = int((m > 0).sum())
        pct = 100.0 * num_masked / (h * w)
        print(f"  - {name:<14} -> {num_masked:>8} px masked ({pct:5.2f}%), mask: {mask_path.name}, overlay: {overlay_path.name}")

    print(f"\n{'='*60}")
    print("✓ All masks and overlays saved.")
    print(f"{'='*60}\n")
    return True


def verify_installation() -> bool:
    """
    Verify required packages for this script.
    """
    print("Verifying installation for masking...")
    try:
        import numpy  # noqa: F401
        print("  ✓ NumPy installed")
    except Exception:
        print("  ✗ NumPy not installed")
        return False
    try:
        import cv2  # noqa: F401
        print("  ✓ OpenCV installed")
    except Exception:
        print("  ✗ OpenCV not installed")
        return False
    try:
        from PIL import Image  # noqa: F401
        print("  ✓ Pillow installed")
    except Exception:
        print("  ✗ Pillow not installed")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test occlusion mask generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python test_masking.py --image sample_face.jpg

  # With feathering and custom margin
  python test_masking.py --image sample_face.jpg --feather 3 --margin 6

  # Batch directory
  python test_masking.py --batch images/ --output_dir output/masking
        """,
    )
    parser.add_argument('--image', type=str, help='Path to input image file')
    parser.add_argument('--batch', type=str, help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, default='output/masking', help='Output directory (default: output/masking)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto', help='Device for segmentation (default: auto)')
    parser.add_argument('--model', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512', help='SegFormer model name')
    parser.add_argument('--feather', type=int, default=0, help='Feather size in pixels (0 to disable)')
    parser.add_argument('--margin', type=int, default=4, help='Default margin/padding in pixels for regions')

    args = parser.parse_args()

    if not verify_installation():
        print("\n✗ Missing dependencies. Please install required packages.")
        return 1

    if not args.image and not args.batch:
        parser.print_help()
        print("\n✗ Error: Must specify either --image or --batch")
        return 1

    # Initialize segmenter and detector
    device = None if args.device == 'auto' else args.device
    print("\nInitializing models...")
    try:
        segmenter = create_segmenter(model_name=args.model, device=device)
        print("  ✓ FaceSegmenter ready")
    except Exception as e:
        print(f"  ✗ Failed to initialize FaceSegmenter: {e}")
        return 1
    try:
        detector = create_detector(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        print("  ✓ LandmarkDetector ready")
    except Exception as e:
        print(f"  ✗ Failed to initialize LandmarkDetector: {e}")
        print("    Proceeding without landmarks; masks will rely on segmentation and heuristics.")
        detector = None  # type: ignore

    out_dir = Path(args.output_dir)
    _ensure_output_dir(out_dir)

    # Process single image
    if args.image:
        ok = generate_and_save_masks(
            image_path=args.image,
            output_dir=out_dir,
            segmenter=segmenter,
            detector=detector,  # type: ignore
            feather_px=args.feather,
            default_margin=args.margin,
        )
        return 0 if ok else 1

    # Batch processing
    image_dir = Path(args.batch)
    if not image_dir.exists():
        print(f"✗ Error: Directory not found: {image_dir}")
        return 1
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]
    if not images:
        print(f"✗ Error: No image files found in {image_dir}")
        return 1

    ok_all = True
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_path.name}")
        ok = generate_and_save_masks(
            image_path=str(img_path),
            output_dir=out_dir,
            segmenter=segmenter,
            detector=detector,  # type: ignore
            feather_px=args.feather,
            default_margin=args.margin,
        )
        ok_all = ok_all and ok

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())


