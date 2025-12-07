"""
FAST Regeneration of Face Contours using MULTIPROCESSING.
Uses all available CPU cores to speed up PNG encoding/decoding.
"""

import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# WORKER FUNCTION (Running on multiple cores)
# ---------------------------------------------------------
def process_single_image(args):
    """Worker function to process one image."""
    lm_path, dataset_dir = args
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    face_contour_dir = features_dir / "face_contour"
    
    try:
        # Load landmarks
        with open(lm_path) as f:
            lm_data = json.load(f)
        groups = lm_data.get("groups", {})
        image_name = lm_path.stem.replace("_landmarks", "")

        # Find raw image (Replicated logic for worker)
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

        raw_img = Image.open(raw_path).convert("RGBA")
        img_width, img_height = raw_img.size
        
        # 1. Base Mask
        face_contour_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        if "face_contour" in groups:
            fc_pts = groups["face_contour"].get("landmarks_pixel", [])
            if fc_pts:
                pts = np.array([[int(p[0]), int(p[1])] for p in fc_pts], dtype=np.int32)
                cv2.fillPoly(face_contour_mask, [pts], 255)
        else:
            face_contour_mask.fill(255)

        # 2. Cut Holes
        features_to_cut = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth_outer", "mouth_inner"]
        for feat_name in features_to_cut:
            if feat_name in groups:
                ft_pts = groups[feat_name].get("landmarks_pixel", [])
                if ft_pts:
                    pts = np.array([[int(p[0]), int(p[1])] for p in ft_pts], dtype=np.int32)
                    cv2.fillPoly(face_contour_mask, [pts], 0)

        # 3. Cut Nose Triangle
        if "nose_tip" in groups and "nose_base" in groups:
            nt_pts = groups["nose_tip"].get("landmarks_pixel", [])
            nb_pts = groups["nose_base"].get("landmarks_pixel", [])
            if nt_pts and nb_pts:
                nt, nb = np.array(nt_pts), np.array(nb_pts)
                top = nt[np.argmin(nt[:, 1])]
                left = nb[np.argmin(nb[:, 0])]
                right = nb[np.argmax(nb[:, 0])]
                tri_pts = np.array([[int(top[0]), int(top[1])], [int(left[0]), int(left[1])], [int(right[0]), int(right[1])]], dtype=np.int32)
                cv2.fillPoly(face_contour_mask, [tri_pts], 0)

        # Apply and Save
        face_contour_img = raw_img.copy()
        face_contour_img.putalpha(Image.fromarray(face_contour_mask, mode='L'))
        
        face_contour_img.save(face_contour_dir / f"{image_name}_face_contour.png")
        Image.fromarray(face_contour_mask, mode='L').save(face_contour_dir / f"{image_name}_face_contour_mask.png")
        
        return True

    except Exception:
        return False

# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------
def regenerate_fast(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    landmarks_dir = features_dir / "landmarks"
    face_contour_dir = features_dir / "face_contour"
    face_contour_dir.mkdir(parents=True, exist_ok=True)
    
    landmark_files = sorted(landmarks_dir.glob("*_landmarks.json"))
    logger.info(f"Found {len(landmark_files)} landmark files")
    
    # Prepare arguments for workers
    # (We pass the path and dir to every worker)
    tasks = [(p, dataset_dir) for p in landmark_files]
    
    # Use 80% of available cores to avoid freezing the system
    num_cores = max(1, int(cpu_count() * 0.8))
    logger.info(f"Starting multiprocessing pool with {num_cores} cores...")
    
    success_count = 0
    with Pool(num_cores) as pool:
        # Map list of files to workers
        results = list(tqdm(pool.imap(process_single_image, tasks), total=len(tasks), desc="Processing"))
        success_count = sum(results)

    logger.info(f"Complete! Success: {success_count}/{len(landmark_files)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/teaching/G14/forensic_reconstruction/dataset")
    args = parser.parse_args()
    regenerate_fast(args.dataset_dir)

if __name__ == "__main__":
    main()