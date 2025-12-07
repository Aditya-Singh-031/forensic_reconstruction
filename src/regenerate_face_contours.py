"""
Regenerate face_contour AND jawline images as FULL SIZE.
- Fixes "Shifted Jawline" by making it 512x512 (matching raw image).
- Fixes "Missing Nose" by reapplying the Triangle Cutout logic.
"""

import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_raw_image(dataset_dir: Path, image_name: str) -> Path:
    """Locate the original raw image in ffhq or celeba_hq."""
    raw_dirs = [
        dataset_dir / "raw_images" / "ffhq",
        dataset_dir / "raw_images" / "celeba_hq",
        dataset_dir.parent / "dataset" / "raw_images" / "ffhq",
        dataset_dir.parent / "dataset" / "raw_images" / "celeba_hq"
    ]
    for d in raw_dirs:
        if d.exists():
            if (d / f"{image_name}.png").exists(): return d / f"{image_name}.png"
            if (d / f"{image_name}.jpg").exists(): return d / f"{image_name}.jpg"
    return None

def regenerate_all(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    landmarks_dir = features_dir / "landmarks"
    
    # Output directories
    face_contour_dir = features_dir / "face_contour"
    jawline_dir = features_dir / "jawline"
    face_contour_dir.mkdir(parents=True, exist_ok=True)
    jawline_dir.mkdir(parents=True, exist_ok=True)
    
    landmark_files = sorted(landmarks_dir.glob("*_landmarks.json"))
    logger.info(f"Found {len(landmark_files)} landmark files")
    
    success = 0
    
    for lm_path in tqdm(landmark_files, desc="Regenerating Full-Size Images"):
        image_name = lm_path.stem.replace("_landmarks", "")
        
        try:
            with open(lm_path) as f:
                lm_data = json.load(f)
            groups = lm_data.get("groups", {})
            
            # 1. Load RAW Image (Full Size)
            raw_path = find_raw_image(dataset_path, image_name)
            if not raw_path:
                continue
            
            raw_img = Image.open(raw_path).convert("RGBA")
            w, h = raw_img.size
            
            # 2. Create Base Mask (The Face Oval/Jawline)
            # This is the "container" for the face
            base_mask = np.zeros((h, w), dtype=np.uint8)
            if "face_contour" in groups:
                fc_pts = groups["face_contour"].get("landmarks_pixel", [])
                if fc_pts:
                    pts = np.array([[int(p[0]), int(p[1])] for p in fc_pts], dtype=np.int32)
                    cv2.fillPoly(base_mask, [pts], 255)
            
            # --- SAVE TARGET (JAWLINE) ---
            # Jawline = Raw Image masked by Base Mask (NO holes)
            # This creates the full-size "Ground Truth"
            jawline_img = raw_img.copy()
            jawline_img.putalpha(Image.fromarray(base_mask, mode='L'))
            jawline_img.save(jawline_dir / f"{image_name}_jawline.png")
            
            # --- CREATE INPUT (FACE_CONTOUR) ---
            # Face Contour = Base Mask MINUS Features (Holes)
            contour_mask = base_mask.copy()
            
            # A. Subtract Standard Features
            standard_feats = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth_outer", "mouth_inner"]
            for ft in standard_feats:
                if ft in groups:
                    ft_pts = groups[ft].get("landmarks_pixel", [])
                    if ft_pts:
                        pts = np.array([[int(p[0]), int(p[1])] for p in ft_pts], dtype=np.int32)
                        cv2.fillPoly(contour_mask, [pts], 0) # Fill with 0 (Black) to erase
            
            # B. Subtract NOSE TRIANGLE (The Fix)
            if 'nose_tip' in groups and 'nose_base' in groups:
                nt_pts = groups['nose_tip'].get("landmarks_pixel", [])
                nb_pts = groups['nose_base'].get("landmarks_pixel", [])
                
                if nt_pts and nb_pts:
                    nt = np.array(nt_pts)
                    nb = np.array(nb_pts)
                    
                    # Calculate Triangle Vertices
                    top = nt[np.argmin(nt[:, 1])]       # Topmost point
                    left = nb[np.argmin(nb[:, 0])]      # Leftmost point
                    right = nb[np.argmax(nb[:, 0])]     # Rightmost point
                    
                    # Create Triangle
                    tri_pts = np.array([[int(top[0]), int(top[1])], 
                                      [int(left[0]), int(left[1])], 
                                      [int(right[0]), int(right[1])]], dtype=np.int32)
                    
                    # Fill Triangle with 0 (Black) to erase
                    cv2.fillPoly(contour_mask, [tri_pts], 0)

            # Save Face Contour
            contour_img = raw_img.copy()
            contour_img.putalpha(Image.fromarray(contour_mask, mode='L'))
            contour_img.save(face_contour_dir / f"{image_name}_face_contour.png")
            
            # Save Mask (useful for debugging)
            Image.fromarray(contour_mask, mode='L').save(face_contour_dir / f"{image_name}_face_contour_mask.png")
            
            success += 1
            
        except Exception as e:
            logger.error(f"Error {image_name}: {e}")

    logger.info(f"Regenerated {success} image pairs (Contour + Jawline)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/teaching/G14/forensic_reconstruction/dataset")
    args = parser.parse_args()
    regenerate_all(args.dataset_dir)