#!/usr/bin/env python3
"""
Quick verification script to test that all models are downloaded and loadable.
Run this after download_models.py to verify everything works.
"""

import sys
from pathlib import Path

def test_bisenet():
    """Test BiSeNet model."""
    print("Testing BiSeNet...", end=" ")
    try:
        import torch
        model_path = Path("models/bisenet/79999_iter.pth")
        if not model_path.exists():
            print("✗ Model file not found")
            return False
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ Loaded (keys: {len(checkpoint)} items)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe."""
    print("Testing MediaPipe...", end=" ")
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
        print(f"✓ Loaded (version: {mp.__version__})")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_stable_diffusion():
    """Test Stable Diffusion."""
    print("Testing Stable Diffusion...", end=" ")
    try:
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        print("Loading (this may take a minute)...", end=" ")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✓ Loaded")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_clip():
    """Test CLIP."""
    print("Testing CLIP...", end=" ")
    try:
        import open_clip
        import torch
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        print("✓ Loaded")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_arcface():
    """Test ArcFace."""
    print("Testing ArcFace...", end=" ")
    try:
        # Try ONNX first
        onnx_path = Path("models/arcface/arcface_r50_v1.onnx")
        if onnx_path.exists():
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            print("✓ ONNX model loaded")
            return True
        else:
            # Try InsightFace library
            import insightface
            app = insightface.app.FaceAnalysis()
            print("✓ InsightFace initialized")
            return True
    except ImportError:
        print("⚠ Not installed (install onnxruntime or insightface)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Model Verification")
    print("=" * 60)
    print()
    
    results = {
        "BiSeNet": test_bisenet(),
        "MediaPipe": test_mediapipe(),
        "Stable Diffusion": test_stable_diffusion(),
        "CLIP": test_clip(),
        "ArcFace": test_arcface(),
    }
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = True
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if not success:
            all_ok = False
    
    print()
    if all_ok:
        print("✓ All models verified successfully!")
        return 0
    else:
        print("✗ Some models failed verification")
        print("  Run download_models.py to download missing models")
        return 1

if __name__ == "__main__":
    sys.exit(main())

