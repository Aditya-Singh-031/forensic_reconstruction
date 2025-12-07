"""
FAST Regeneration of Jawlines (Target) using MULTIPROCESSING.
Fix: Masks the raw image to show ONLY the face oval (removes background),
matching the style of the face_contour input.
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_jawline(args):
    """Worker function to process one masked jawline."""
    lm_path, dataset_dir = args
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    jawline_dir = features_dir / "jawline"
    
    try:
        # 1. Load Landmarks
        with open(lm_path) as f:
            lm_data = json.load(f)
        
        image_name = lm_path.stem.replace("_landmarks", "")
        groups = lm_data.get("groups", {})
        
        # 2. Find Raw Image
        raw_path = None
        raw_dirs = [
            dataset_path / "raw_images" / "ffhq",
            dataset_path / "raw_images" / "celeba_hq",
        ]
        for d in raw_dirs:
            if (d / f"{image_name}.png").exists(): raw_path = d / f"{image_name}.png"; break
            if (d / f"{image_name}.jpg").exists(): raw_path = d / f"{image_name}.jpg"; break
        
        if not raw_path:
            return False
        
        # Load Raw Image
        raw_img = Image.open(raw_path).convert("RGBA")
        w, h = raw_img.size
        
        # 3. Create Jawline Mask (Face Oval)
        # Start with full transparent (black)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if "face_contour" in groups:
            fc_data = groups["face_contour"]
            if "landmarks_pixel" in fc_data:
                # Convert landmarks to numpy array
                pts = np.array(
                    [[int(p[0]), int(p[1])] for p in fc_data["landmarks_pixel"]],
                    dtype=np.int32
                )
                # Fill the face oval with white (opaque)
                cv2.fillPoly(mask, [pts], 255)
        else:
            # Fallback: If no contour detected, keep full image (or you could skip)
            mask.fill(255)
            
        # 4. Apply Mask to Raw Image
        # This keeps the face but makes the background transparent
        jawline_img = raw_img.copy()
        jawline_img.putalpha(Image.fromarray(mask, mode='L'))
        
        # 5. Save
        output_path = jawline_dir / f"{image_name}_jawline.png"
        jawline_img.save(output_path)
        
        return True
        
    except Exception:
        return False

def regenerate_jawlines_fast(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    landmarks_dir = features_dir / "landmarks"
    jawline_dir = features_dir / "jawline"
    
    jawline_dir.mkdir(parents=True, exist_ok=True)
    
    landmark_files = sorted(landmarks_dir.glob("*_landmarks.json"))
    logger.info(f"Found {len(landmark_files)} landmark files")
    
    tasks = [(p, dataset_dir) for p in landmark_files]
    
    num_cores = max(1, int(cpu_count() * 0.8))
    logger.info(f"Starting multiprocessing pool with {num_cores} cores...")
    
    success_count = 0
    with Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_jawline, tasks),
            total=len(tasks),
            desc="Masking jawlines"
        ))
        success_count = sum(results)
    
    logger.info(f"Complete! Success: {success_count}/{len(landmark_files)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/teaching/G14/forensic_reconstruction/dataset")
    args = parser.parse_args()
    regenerate_jawlines_fast(args.dataset_dir)

if __name__ == "__main__":
    main()