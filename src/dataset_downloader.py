"""
Complete Dataset Downloader for FFHQ and CelebA-HQ
Downloads and organizes facial image datasets from multiple sources
"""

import os
from pathlib import Path
import logging
from typing import Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FFHQDownloader:
    """Download FFHQ dataset from Kaggle or Google Drive."""
    
    def __init__(self, output_dir: str = "/DATA/facial_features_dataset/raw_images/ffhq"):
        """Initialize FFHQ downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FFHQ output directory: {self.output_dir}")
    
    def download_via_kaggle(self, dataset_name: str = "arnaud58/flickrfaceshq-dataset-ffhq"):
        """
        Download FFHQ via Kaggle API.
        
        Requires: pip install kaggle
        Setup: Place kaggle.json in ~/.kaggle/
        """
        try:
            import kaggle  # type: ignore
            logger.info(f"Downloading FFHQ from Kaggle: {dataset_name}") # type: ignore
            
            kaggle.api.dataset_download_files( # type: ignore
                dataset_name,
                path=str(self.output_dir),
                unzip=True
            )
            logger.info("✓ FFHQ downloaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
            logger.info("To use Kaggle API:")
            logger.info("  1. pip install kaggle")
            logger.info("  2. Download kaggle.json from https://www.kaggle.com/settings/account")
            logger.info("  3. Place in ~/.kaggle/")
            return False
    
    def download_via_google_drive(self):
        """Download FFHQ from Google Drive."""
        try:
            import gdown  # type: ignore
            logger.info("Downloading FFHQ from Google Drive...") # type: ignore
            
            # FFHQ 1024x1024 on Google Drive
            url = "https://drive.google.com/uc?id=1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP"
            output = str(self.output_dir / "ffhq_1024.zip")
            
            gdown.download(url, output, quiet=False) # type: ignore
            
            # Unzip
            import zipfile
            logger.info("Extracting...")
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            
            os.remove(output)
            logger.info("✓ FFHQ downloaded from Google Drive")
            return True
        
        except Exception as e:
            logger.warning(f"Google Drive download failed: {e}")
            return False
    
    def download_manually_guided(self):
        """Guide user to download manually."""
        logger.info("\n" + "="*60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
        logger.info("="*60)
        logger.info("\n1. Visit: https://github.com/NVlabs/ffhq-dataset")
        logger.info("2. Download the dataset")
        logger.info("3. Extract to: " + str(self.output_dir))
        logger.info("\nOr use Kaggle:")
        logger.info("  kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq")
        logger.info("  unzip -q flickrfaceshq-dataset-ffhq.zip -d " + str(self.output_dir))
        logger.info("="*60 + "\n")
    
    def verify(self) -> int:
        """Verify downloaded images."""
        image_count = len(list(self.output_dir.glob("*.png")))
        logger.info(f"Found {image_count} FFHQ images")
        return image_count


class CelebAHQDownloader:
    """Download CelebA-HQ dataset."""
    
    def __init__(self, output_dir: str = "/DATA/facial_features_dataset/raw_images/celeba_hq"):
        """Initialize CelebA-HQ downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CelebA-HQ output directory: {self.output_dir}")
    
    def download_via_kaggle(self, dataset_name: str = "lamsimon/celebahq"):
        """Download CelebA-HQ via Kaggle."""
        try:
            import kaggle  # type: ignore
            logger.info(f"Downloading CelebA-HQ from Kaggle: {dataset_name}") # type: ignore
            
            kaggle.api.dataset_download_files( # type: ignore
                dataset_name,
                path=str(self.output_dir),
                unzip=True
            )
            logger.info("✓ CelebA-HQ downloaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
            return False
    
    def verify(self) -> int:
        """Verify downloaded images."""
        jpg_count = len(list(self.output_dir.glob("*.jpg")))
        png_count = len(list(self.output_dir.glob("*.png")))
        total = jpg_count + png_count
        logger.info(f"Found {total} CelebA-HQ images ({jpg_count} JPG, {png_count} PNG)")
        return total


class DatasetDownloader:
    """Main dataset downloader orchestrating both FFHQ and CelebA-HQ."""
    
    def __init__(self, output_dir: str = "/DATA/facial_features_dataset/raw_images"):
        """Initialize main downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ffhq_downloader = FFHQDownloader(str(self.output_dir / "ffhq"))
        self.celeba_downloader = CelebAHQDownloader(str(self.output_dir / "celeba_hq"))
        
        logger.info(f"Dataset downloader initialized")
        logger.info(f"Output base: {self.output_dir}")
    
    def download_ffhq(self, method: str = "auto"):
        """
        Download FFHQ dataset.
        
        Args:
            method: 'kaggle', 'google_drive', 'manual', or 'auto'
        """
        logger.info("\n" + "="*60)
        logger.info("DOWNLOADING FFHQ DATASET")
        logger.info("="*60)
        
        if method == "kaggle" or method == "auto":
            if self.ffhq_downloader.download_via_kaggle():
                return
        
        if method == "google_drive" or method == "auto":
            if self.ffhq_downloader.download_via_google_drive():
                return
        
        self.ffhq_downloader.download_manually_guided()
    
    def download_celeba_hq(self, method: str = "auto"):
        """
        Download CelebA-HQ dataset.
        
        Args:
            method: 'kaggle' or 'auto'
        """
        logger.info("\n" + "="*60)
        logger.info("DOWNLOADING CELEBA-HQ DATASET")
        logger.info("="*60)
        
        if method == "kaggle" or method == "auto":
            if self.celeba_downloader.download_via_kaggle():
                return
        
        logger.info("\n" + "="*60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
        logger.info("="*60)
        logger.info("\n1. Visit: https://github.com/IIGROUP/MM-CelebA-HQ-Dataset")
        logger.info("2. Download images")
        logger.info("3. Extract to: " + str(self.output_dir / "celeba_hq"))
        logger.info("\nOr use Kaggle:")
        logger.info("  kaggle datasets download -d lamsimon/celebahq")
        logger.info("="*60 + "\n")
    
    def verify_downloads(self) -> Dict[str, int]:
        """Verify all downloads."""
        logger.info("\n" + "="*60)
        logger.info("VERIFYING DOWNLOADS")
        logger.info("="*60)
        
        ffhq_count = self.ffhq_downloader.verify()
        celeba_count = self.celeba_downloader.verify()
        
        total = ffhq_count + celeba_count
        
        logger.info(f"\nTotal images: {total}")
        logger.info(f"  FFHQ: {ffhq_count}")
        logger.info(f"  CelebA-HQ: {celeba_count}")
        
        return { # type: ignore
            'ffhq': ffhq_count, # type: ignore
            'celeba_hq': celeba_count,
            'total': total
        }
    
    def create_manifest(self) -> Dict[str, Any]:
        """Create JSON manifest of all downloaded images."""
        logger.info("\nCreating manifest...")
        
        manifest: Dict[str, Any] = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "description": "Manifest of downloaded face images"
            },
            "ffhq": [],
            "celeba_hq": []
        }
        
        # FFHQ images
        ffhq_dir = self.output_dir / "ffhq"
        if ffhq_dir.exists():
            for img_path in sorted(ffhq_dir.glob("*.png")):
                manifest["ffhq"].append({
                    "filename": img_path.name,
                    "path": str(img_path),
                    "source": "ffhq"
                })
        
        # CelebA-HQ images
        celeba_dir = self.output_dir / "celeba_hq"
        if celeba_dir.exists():
            for img_path in sorted(celeba_dir.glob("*.jpg")):
                manifest["celeba_hq"].append({
                    "filename": img_path.name,
                    "path": str(img_path),
                    "source": "celeba_hq"
                })
            for img_path in sorted(celeba_dir.glob("*.png")):
                manifest["celeba_hq"].append({
                    "filename": img_path.name,
                    "path": str(img_path),
                    "source": "celeba_hq"
                })
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_images = len(manifest["ffhq"]) + len(manifest["celeba_hq"])
        logger.info(f"✓ Manifest created: {total_images} images")
        logger.info(f"  Saved to: {manifest_path}")
        
        return manifest


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download FFHQ and CelebA-HQ facial datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/DATA/facial_features_dataset/raw_images",
        help="Output directory for raw images"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ffhq", "celeba_hq", "both"],
        default="both",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["kaggle", "google_drive", "manual", "auto"],
        default="auto",
        help="Download method"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output)
    
    if args.verify_only:
        downloader.verify_downloads()
        downloader.create_manifest()
        return
    
    # Download
    if args.dataset in ["ffhq", "both"]:
        downloader.download_ffhq(args.method)
    
    if args.dataset in ["celeba_hq", "both"]:
        downloader.download_celeba_hq(args.method)
    
    # Verify and create manifest
    stats = downloader.verify_downloads()
    downloader.create_manifest()
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total: {stats['total']} images")
    logger.info(f"Location: {args.output}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
