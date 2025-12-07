"""
FAST Regeneration of Jawlines using MULTIPROCESSING.
Simply copies raw images to jawline directory at full size.
"""

import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_single_jawline(args):
    """Worker function to process one jawline."""
    lm_path, dataset_dir = args
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    jawline_dir = features_dir / "jawline"
    
    try:
        image_name = lm_path.stem.replace("_landmarks", "")
        
        # Find raw image
        raw_path = None
        raw_dirs = [
            dataset_path / "raw_images" / "ffhq",
            dataset_path / "raw_images" / "celeba_hq",
        ]
        
        for d in raw_dirs:
            if (d / f"{image_name}.png").exists():
                raw_path = d / f"{image_name}.png"
                break
            if (d / f"{image_name}.jpg").exists():
                raw_path = d / f"{image_name}.jpg"
                break
        
        if not raw_path:
            return False
        
        # Load and save as RGBA (jawline = full original face)
        raw_img = Image.open(raw_path).convert("RGBA")
        output_path = jawline_dir / f"{image_name}_jawline.png"
        raw_img.save(output_path)
        
        return True
        
    except Exception:
        return False


def regenerate_jawlines_fast(dataset_dir: str):
    """Regenerate jawlines using multiprocessing."""
    dataset_path = Path(dataset_dir)
    features_dir = dataset_path / "features"
    landmarks_dir = features_dir / "landmarks"
    jawline_dir = features_dir / "jawline"
    
    jawline_dir.mkdir(parents=True, exist_ok=True)
    
    landmark_files = sorted(landmarks_dir.glob("*_landmarks.json"))
    logger.info(f"Found {len(landmark_files)} landmark files")
    
    # Prepare tasks
    tasks = [(p, dataset_dir) for p in landmark_files]
    
    # Use 80% of cores
    num_cores = max(1, int(cpu_count() * 0.8))
    logger.info(f"Starting multiprocessing pool with {num_cores} cores...")
    
    success_count = 0
    with Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_jawline, tasks),
            total=len(tasks),
            desc="Processing jawlines"
        ))
        success_count = sum(results)
    
    logger.info(f"Complete! Success: {success_count}/{len(landmark_files)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast jawline regeneration")
    parser.add_argument(
        "--dataset-dir",
        default="/home/teaching/G14/forensic_reconstruction/dataset"
    )
    args = parser.parse_args()
    regenerate_jawlines_fast(args.dataset_dir)


if __name__ == "__main__":
    main()
