#!/usr/bin/env python3
"""
Test script for text-to-face generation using Stable Diffusion.

This script:
  - Loads one or more descriptions
  - Generates 1..N images per description
  - Saves images with sanitized filenames
  - Prints parameters and timing

Usage examples:
  # Single description
  python -m src.test_text_to_face --description "Adult male, thick mustache" --num_images 2

  # From file (one description per line)
  python -m src.test_text_to_face --descriptions_file data/sample_descriptions.txt --num_images 2

  # Custom steps/guidance/seed
  python -m src.test_text_to_face \
      --description "Middle-aged Indian female, bindi, smiling" \
      --steps 40 --guidance_scale 10 --seed 123
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from text_to_face import TextToFaceGenerator, sanitize_filename  # noqa: E402


def load_descriptions(file_path: Path) -> List[str]:
    lines: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate faces from text descriptions (Stable Diffusion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tips for better prompts:
  - Be specific: age, gender, ethnicity, key features (e.g., "thick mustache")
  - Add photographic cues: "photorealistic", "natural lighting", "high detail"
  - Avoid: style words like "cartoon", "sketch", "painting" unless desired

Bad prompt examples:
  - "a face" (too vague)
  - "beautiful" (subjective, weak signal)
  - "photo" (too generic)

For forensic descriptions:
  - Include demographics (age range, gender, ethnicity)
  - Distinctive marks (scars, moles, glasses, beard/mustache)
  - Hair (style, color, length), facial structure, expression
  - Lighting and realism cues
        """,
    )
    parser.add_argument("--description", type=str, help="Single text description")
    parser.add_argument("--descriptions_file", type=str, help="Path to file with multiple descriptions")
    parser.add_argument("--num_images", type=int, default=1, help="Images per description (default: 1)")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (20-50, default: 30)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (7.5-15, default: 7.5)")
    parser.add_argument("--seed", type=int, help="Random seed (reproducibility)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="output/text_to_face", help="Output directory")

    args = parser.parse_args()

    if not args.description and not args.descriptions_file:
        parser.print_help()
        print("\n✗ Error: Provide --description or --descriptions_file")
        return 1

    # Device
    device = None if args.device == "auto" else args.device

    # Initialize generator
    print("Initializing generator...")
    try:
        gen = TextToFaceGenerator(device=device)
        print("  ✓ Generator ready")
    except Exception as e:
        print(f"✗ Failed to initialize generator: {e}")
        return 1

    # Build description list
    descriptions: List[str] = []
    if args.description:
        descriptions.append(args.description.strip())
    if args.descriptions_file:
        fp = Path(args.descriptions_file)
        if not fp.exists():
            print(f"✗ Descriptions file not found: {fp}")
            return 1
        descriptions.extend(load_descriptions(fp))

    # Output paths
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.num_images} image(s) per description...")

    total_start = time.time()
    for idx, desc in enumerate(descriptions, 1):
        print(f"\n[{idx}/{len(descriptions)}] {desc}")
        start = time.time()
        try:
            images = gen.generate_batch(
                description=desc,
                num_images=args.num_images,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                show_progress=True,
            )
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            continue

        elapsed = time.time() - start
        print(f"  ✓ Generated {len(images)} image(s) in {elapsed:.2f}s")

        # Save images
        folder = out_root / sanitize_filename(desc, 80)
        folder.mkdir(exist_ok=True)
        for i, img in enumerate(images, 1):
            out_path = folder / f"{i:02d}.png"
            img.save(out_path)
            print(f"    - Saved: {out_path}")

    total_elapsed = time.time() - total_start
    print(f"\n✓ All done in {total_elapsed:.2f}s. Results in: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
