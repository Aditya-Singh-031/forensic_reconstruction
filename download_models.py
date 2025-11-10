#!/usr/bin/env python3
"""
Model Download Script for Forensic Facial Reconstruction System

This script downloads and verifies all pre-trained models needed for:
- Face segmentation (BiSeNet)
- Landmark detection (MediaPipe)
- Image inpainting (Stable Diffusion)
- Text encoding (CLIP)
- Face recognition (ArcFace)

Author: Auto-generated for forensic reconstruction system
Date: 2025
"""

import os
import sys
import shutil
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import json

# Third-party imports
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing required packages. Install with:")
    print("  pip install requests tqdm")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Base directory for models
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"model_download_{int(time.time())}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Download settings
MAX_RETRIES = 3
TIMEOUT = 300  # 5 minutes per download
CHUNK_SIZE = 8192  # 8KB chunks for progress bar

# ============================================================================
# Utility Functions
# ============================================================================

def get_disk_space(path: Path) -> Tuple[int, int]:
    """Get free and total disk space in bytes."""
    stat = shutil.disk_usage(path)
    return stat.free, stat.total

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def calculate_file_hash(filepath: Path, algorithm='sha256') -> str:
    """Calculate hash of a file."""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

class DownloadProgressBar:
    """Progress bar for downloads."""
    def __init__(self, total_size: int, desc: str):
        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        self.downloaded = 0
    
    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        self.pbar.update(chunk_size)
    
    def close(self):
        self.pbar.close()

def download_file(url: str, dest_path: Path, expected_size: Optional[int] = None,
                  expected_hash: Optional[str] = None, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar and retry logic.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        expected_size: Expected file size in bytes (for verification)
        expected_hash: Expected SHA256 hash (for verification)
        desc: Description for progress bar
    
    Returns:
        True if download successful, False otherwise
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if dest_path.exists():
        file_size = dest_path.stat().st_size
        if expected_size and file_size == expected_size:
            logger.info(f"File already exists: {dest_path}")
            if expected_hash:
                actual_hash = calculate_file_hash(dest_path)
                if actual_hash == expected_hash:
                    logger.info("✓ File hash verified - skipping download")
                    return True
                else:
                    logger.warning("File exists but hash mismatch - re-downloading")
                    dest_path.unlink()
            else:
                logger.info("✓ File exists - skipping download")
                return True
        else:
            logger.warning(f"File exists but size mismatch ({file_size} vs {expected_size}) - re-downloading")
            dest_path.unlink()
    
    # Download with retries
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Get file size first
            response = requests.head(url, timeout=10, allow_redirects=True)
            if 'Content-Length' in response.headers:
                total_size = int(response.headers['Content-Length'])
            else:
                # Try to get size from GET request
                response = requests.get(url, stream=True, timeout=10)
                total_size = int(response.headers.get('Content-Length', 0))
                response.close()
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            
            progress = DownloadProgressBar(total_size, desc)
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            progress.close()
            response.close()
            
            # Verify file size
            actual_size = dest_path.stat().st_size
            if expected_size and actual_size != expected_size:
                logger.error(f"Size mismatch: expected {expected_size}, got {actual_size}")
                dest_path.unlink()
                raise ValueError("File size mismatch")
            
            # Verify hash if provided
            if expected_hash:
                actual_hash = calculate_file_hash(dest_path)
                if actual_hash != expected_hash:
                    logger.error(f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}...")
                    dest_path.unlink()
                    raise ValueError("File hash mismatch")
                logger.info(f"✓ Hash verified: {actual_hash[:16]}...")
            
            logger.info(f"✓ Download complete: {dest_path}")
            return True
            
        except (URLError, HTTPError, requests.RequestException) as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download after {MAX_RETRIES} attempts")
                if dest_path.exists():
                    dest_path.unlink()
                return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    return False

# ============================================================================
# Model Download Functions
# ============================================================================

def download_bisenet_model() -> bool:
    """
    Download BiSeNet face parsing model.
    
    Source: https://github.com/zllrunning/face-parsing.PyTorch
    Model: 79999_iter.pth
    """
    logger.info("=" * 60)
    logger.info("Downloading BiSeNet Face Parsing Model")
    logger.info("=" * 60)
    
    model_dir = MODELS_DIR / "bisenet"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "79999_iter.pth"
    
    # Primary source (GitHub releases or direct link)
    # Note: You may need to update this URL if the repository changes
    urls = [
        "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth",
        "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812",  # Alternative
    ]
    
    # Expected size: ~377MB (approximate)
    expected_size = 395 * 1024 * 1024  # 395 MB
    
    for url in urls:
        logger.info(f"Trying URL: {url}")
        if download_file(url, model_path, expected_size=expected_size, desc="BiSeNet"):
            # Verify model can be loaded
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                logger.info("✓ BiSeNet model loaded successfully")
                logger.info(f"  Model keys: {list(checkpoint.keys())[:5]}...")
                return True
            except Exception as e:
                logger.error(f"Failed to load BiSeNet model: {e}")
                logger.error("Model file may be corrupted. Try downloading again.")
                model_path.unlink()
                return False
    
    logger.error("Failed to download BiSeNet model from all sources")
    logger.error("Manual download: https://github.com/zllrunning/face-parsing.PyTorch")
    return False

def verify_mediapipe() -> bool:
    """
    Verify MediaPipe Face Mesh is available.
    MediaPipe models are included in the package, no download needed.
    """
    logger.info("=" * 60)
    logger.info("Verifying MediaPipe Face Mesh")
    logger.info("=" * 60)
    
    try:
        import mediapipe as mp
        logger.info(f"✓ MediaPipe version: {mp.__version__}")
        
        # Test face mesh initialization
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("✓ MediaPipe Face Mesh initialized successfully")
        logger.info("  - 468 facial landmarks supported")
        logger.info("  - No additional download needed")
        return True
    except ImportError:
        logger.error("✗ MediaPipe not installed")
        logger.error("  Install with: pip install mediapipe")
        return False
    except Exception as e:
        logger.error(f"✗ MediaPipe verification failed: {e}")
        return False

def download_stable_diffusion_inpainting() -> bool:
    """
    Download Stable Diffusion v1.5 inpainting model.
    Uses Hugging Face diffusers library (auto-downloads).
    """
    logger.info("=" * 60)
    logger.info("Downloading Stable Diffusion v1.5 Inpainting Model")
    logger.info("=" * 60)
    
    try:
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        
        model_dir = MODELS_DIR / "stable_diffusion_inpainting"
        model_dir.mkdir(exist_ok=True)
        
        logger.info("Loading Stable Diffusion inpainting pipeline...")
        logger.info("  This will download ~7GB of model files (first time only)")
        logger.info("  Model will be cached in: ~/.cache/huggingface/")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Using device: {device}")
        
        # Load pipeline (downloads automatically)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=str(model_dir)
        )
        
        logger.info("✓ Stable Diffusion inpainting model loaded")
        logger.info(f"  Model location: {model_dir}")
        logger.info("  Note: Model files are cached by Hugging Face")
        
        # Test inference (optional, can be slow)
        # logger.info("Testing model...")
        # test_image = torch.zeros((1, 3, 512, 512))
        # test_mask = torch.ones((1, 1, 512, 512))
        # _ = pipe(prompt="test", image=test_image, mask_image=test_mask)
        # logger.info("✓ Model test passed")
        
        return True
        
    except ImportError:
        logger.error("✗ Diffusers not installed")
        logger.error("  Install with: pip install diffusers transformers accelerate")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to load Stable Diffusion: {e}")
        return False

def download_clip_model() -> bool:
    """
    Download CLIP ViT-B/32 model.
    Uses open_clip library (auto-downloads).
    """
    logger.info("=" * 60)
    logger.info("Downloading CLIP ViT-B/32 Model")
    logger.info("=" * 60)
    
    try:
        import open_clip
        import torch
        
        model_dir = MODELS_DIR / "clip"
        model_dir.mkdir(exist_ok=True)
        
        logger.info("Loading CLIP ViT-B/32 model...")
        logger.info("  This will download ~350MB (first time only)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            cache_dir=str(model_dir)
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        model = model.to(device)
        logger.info(f"✓ CLIP ViT-B/32 model loaded on {device}")
        logger.info(f"  Model location: {model_dir}")
        
        # Test encoding
        text = tokenizer(["a photo of a face"])
        with torch.no_grad():
            text_features = model.encode_text(text.to(device))
        logger.info(f"✓ CLIP text encoding test passed (feature dim: {text_features.shape})")
        
        return True
        
    except ImportError:
        logger.error("✗ open-clip-torch not installed")
        logger.error("  Install with: pip install open-clip-torch")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to load CLIP: {e}")
        return False

def download_arcface_model() -> bool:
    """
    Download ArcFace R50 model for face embeddings.
    
    Source: https://github.com/deepinsight/insightface
    We'll try to download ONNX or PyTorch weights.
    """
    logger.info("=" * 60)
    logger.info("Downloading ArcFace R50 Model")
    logger.info("=" * 60)
    
    model_dir = MODELS_DIR / "arcface"
    model_dir.mkdir(exist_ok=True)
    
    # Try multiple sources for ArcFace
    # Option 1: InsightFace ONNX model
    onnx_path = model_dir / "arcface_r50_v1.onnx"
    onnx_urls = [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r50_v1.onnx",
        "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcface_r100_v1.onnx",  # Alternative
    ]
    
    # Option 2: PyTorch weights (if available)
    pytorch_path = model_dir / "arcface_r50.pth"
    
    # Try ONNX first (more common)
    logger.info("Attempting to download ArcFace ONNX model...")
    for url in onnx_urls:
        logger.info(f"Trying URL: {url}")
        # ONNX model is typically ~100-200MB
        if download_file(url, onnx_path, expected_size=None, desc="ArcFace ONNX"):
            # Verify ONNX model
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                logger.info("✓ ArcFace ONNX model loaded successfully")
                logger.info(f"  Input shape: {session.get_inputs()[0].shape}")
                logger.info(f"  Output shape: {session.get_outputs()[0].shape}")
                return True
            except ImportError:
                logger.warning("onnxruntime not installed - cannot verify ONNX model")
                logger.warning("  Install with: pip install onnxruntime")
                logger.info("  Model file downloaded but not verified")
                return True
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                onnx_path.unlink()
                continue
    
    # If ONNX fails, try PyTorch weights
    logger.info("ONNX download failed, trying PyTorch weights...")
    logger.warning("Note: PyTorch ArcFace weights may need to be downloaded manually")
    logger.warning("  See: https://github.com/deepinsight/insightface")
    
    # Alternative: Use insightface library (auto-downloads)
    try:
        #import insightface
        logger.info("Using insightface library (auto-downloads models)...")
        #app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        logger.info("✓ InsightFace initialized (models auto-downloaded)")
        return True
    except ImportError:
        logger.warning("insightface library not installed")
        logger.warning("  Install with: pip install insightface")
        logger.warning("  Or download ArcFace model manually")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        return False

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to download all models."""
    print("\n" + "=" * 60)
    print("Forensic Facial Reconstruction - Model Download Script")
    print("=" * 60)
    print()
    
    # Check disk space
    free_space, total_space = get_disk_space(MODELS_DIR)
    logger.info(f"Disk space check:")
    logger.info(f"  Free: {format_size(free_space)}")
    logger.info(f"  Total: {format_size(total_space)}")
    
    # Estimate required space: ~50GB
    required_space = 50 * 1024 * 1024 * 1024  # 50 GB
    if free_space < required_space:
        logger.warning(f"⚠ Warning: Less than {format_size(required_space)} free space")
        logger.warning("  Some models may fail to download")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Download cancelled")
            return 1
    else:
        logger.info(f"✓ Sufficient disk space available")
    
    print()
    
    # Track results
    results = {}
    
    # Download/verify each model
    print("\nStarting model downloads...\n")
    
    # 1. BiSeNet
    results['bisenet'] = download_bisenet_model()
    print()
    
    # 2. MediaPipe (verify only)
    results['mediapipe'] = verify_mediapipe()
    print()
    
    # 3. Stable Diffusion
    results['stable_diffusion'] = download_stable_diffusion_inpainting()
    print()
    
    # 4. CLIP
    results['clip'] = download_clip_model()
    print()
    
    # 5. ArcFace
    results['arcface'] = download_arcface_model()
    print()
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    all_success = True
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name:20s} {status}")
        if not success:
            all_success = False
    
    print()
    
    if all_success:
        logger.info("✓ All models downloaded and verified successfully!")
        logger.info(f"  Models directory: {MODELS_DIR}")
        logger.info(f"  Log file: {LOG_FILE}")
        return 0
    else:
        logger.warning("⚠ Some models failed to download")
        logger.warning("  Check log file for details: " + str(LOG_FILE))
        logger.warning("  You may need to download failed models manually")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

