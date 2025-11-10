# Model Download Guide

Complete guide for downloading and verifying all pre-trained models for the Forensic Facial Reconstruction System.

---

## Overview

This guide covers downloading 5 essential models:

1. **BiSeNet** - Face parsing (19 semantic segments)
2. **MediaPipe Face Mesh** - Facial landmarks (already included)
3. **Stable Diffusion v1.5** - Image inpainting (~7GB)
4. **CLIP ViT-B/32** - Text encoding (~350MB)
5. **ArcFace R50** - Face embeddings (~200MB)

**Total estimated size: ~50GB** (including Hugging Face cache)

---

## Quick Start

### Automated Download

```bash
# 1. Activate your conda environment
conda activate forensic310

# 2. Navigate to project directory
cd /home/teaching/G14/forensic_reconstruction

# 3. Run the download script
python download_models.py
```

The script will:
- Check disk space
- Download all models with progress bars
- Verify file integrity
- Test that each model loads correctly
- Log everything to `logs/model_download_*.log`

---

## Prerequisites

Before running the download script, ensure:

- ✅ Python 3.10 environment activated (`forensic310`)
- ✅ PyTorch 2.5.1+cu118 installed
- ✅ All packages from `requirements.txt` installed
- ✅ At least 50GB free disk space
- ✅ Stable internet connection (some models are large)

**Check prerequisites:**
```bash
# Verify environment
conda activate forensic310
python --version  # Should show 3.10.x

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check disk space
df -h .
```

---

## Model Details

### 1. BiSeNet Face Parsing Model

**Purpose:** Semantic face segmentation into 19 components

**File:** `models/bisenet/79999_iter.pth`

**Size:** ~395 MB

**Source:** 
- Primary: https://github.com/zllrunning/face-parsing.PyTorch
- Alternative: Google Drive (if GitHub fails)

**Verification:**
```bash
python -c "import torch; ckpt = torch.load('models/bisenet/79999_iter.pth', map_location='cpu'); print('BiSeNet loaded:', list(ckpt.keys())[:3])"
```

**Manual Download (if script fails):**
```bash
# Option 1: Clone repository
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
cp face-parsing.PyTorch/res/cp/79999_iter.pth models/bisenet/

# Option 2: Direct download
wget -O models/bisenet/79999_iter.pth \
  https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth
```

---

### 2. MediaPipe Face Mesh

**Purpose:** 468 facial landmark detection

**Status:** Already included in MediaPipe package (no download needed)

**Verification:**
```bash
python -c "import mediapipe as mp; fm = mp.solutions.face_mesh.FaceMesh(); print('MediaPipe OK')"
```

**If not installed:**
```bash
pip install mediapipe
```

---

### 3. Stable Diffusion v1.5 Inpainting

**Purpose:** Realistic facial feature reconstruction

**Model ID:** `runwayml/stable-diffusion-inpainting`

**Size:** ~7GB (downloaded automatically by diffusers)

**Location:** Cached in `~/.cache/huggingface/` (or `models/stable_diffusion_inpainting/`)

**Verification:**
```bash
python -c "from diffusers import StableDiffusionInpaintPipeline; import torch; \
  pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting'); \
  print('Stable Diffusion loaded')"
```

**First-time download:**
- Automatically downloads when you first load the pipeline
- Takes 10-30 minutes depending on internet speed
- Progress shown in terminal

**Manual cache location:**
```bash
# Check Hugging Face cache
ls -lh ~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/
```

**Alternative models (if needed):**
- `stabilityai/stable-diffusion-2-inpainting` (SD 2.0)
- `runwayml/stable-diffusion-v1-5` (base model, not inpainting)

---

### 4. CLIP ViT-B/32

**Purpose:** Text-to-image encoding for witness descriptions

**Model:** OpenAI CLIP ViT-B/32

**Size:** ~350MB (downloaded automatically)

**Location:** Cached in `models/clip/` or `~/.cache/torch/`

**Verification:**
```bash
python -c "import open_clip; import torch; \
  model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
  print('CLIP loaded:', model.visual.output_dim)"
```

**Alternative CLIP models:**
- `ViT-L/14` - Larger, more accurate (but slower)
- `ViT-B/16` - Faster alternative

---

### 5. ArcFace R50

**Purpose:** Face embedding extraction for database matching

**Format:** ONNX or PyTorch weights

**Size:** ~100-200MB

**Sources:**
- Primary: InsightFace ONNX model
- Alternative: InsightFace library (auto-downloads)

**Verification (ONNX):**
```bash
python -c "import onnxruntime as ort; \
  session = ort.InferenceSession('models/arcface/arcface_r50_v1.onnx'); \
  print('ArcFace ONNX loaded')"
```

**Verification (InsightFace library):**
```bash
python -c "import insightface; \
  app = insightface.app.FaceAnalysis(); \
  print('InsightFace initialized')"
```

**Manual download (if script fails):**

**Option 1: Using InsightFace library (recommended)**
```bash
pip install insightface
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=0)"
# Models auto-download to ~/.insightface/models/
```

**Option 2: Direct ONNX download**
```bash
mkdir -p models/arcface
wget -O models/arcface/arcface_r50_v1.onnx \
  https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r50_v1.onnx
```

**Option 3: Clone InsightFace repository**
```bash
git clone https://github.com/deepinsight/insightface.git
# Models are in insightface/models/
```

---

## Directory Structure

After downloading, your `models/` directory should look like:

```
models/
├── bisenet/
│   └── 79999_iter.pth              (~395 MB)
├── stable_diffusion_inpainting/    (Hugging Face cache, ~7GB)
├── clip/                           (CLIP cache, ~350MB)
└── arcface/
    └── arcface_r50_v1.onnx         (~200MB)
```

**Note:** Some models (Stable Diffusion, CLIP) are cached by their respective libraries in:
- `~/.cache/huggingface/` (Hugging Face models)
- `~/.cache/torch/` (PyTorch Hub models)
- `~/.insightface/models/` (InsightFace models)

---

## Verification

### Quick Verification

Run the verification script:
```bash
python verify_models.py  # (if you create this script)
```

Or verify manually:
```bash
# Test all models
python -c "
import torch
print('1. BiSeNet:', torch.load('models/bisenet/79999_iter.pth', map_location='cpu') is not None)

import mediapipe as mp
print('2. MediaPipe:', mp.__version__)

from diffusers import StableDiffusionInpaintPipeline
print('3. Stable Diffusion: Loading...')
pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting')
print('   Stable Diffusion: OK')

import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print('4. CLIP: OK')

import onnxruntime as ort
session = ort.InferenceSession('models/arcface/arcface_r50_v1.onnx')
print('5. ArcFace: OK')

print('\n✓ All models verified!')
"
```

---

## Troubleshooting

### Error: "Out of disk space"

**Problem:** Not enough space for large models (~50GB needed)

**Solutions:**
```bash
# Check disk space
df -h .

# Clean pip cache
pip cache purge

# Clean Hugging Face cache (if needed)
rm -rf ~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/

# Download to different location
export HF_HOME=/path/to/larger/disk/.cache/huggingface
```

---

### Error: "Network timeout" or "Connection refused"

**Problem:** Slow/unstable internet or firewall blocking

**Solutions:**
```bash
# Increase timeout in download_models.py
TIMEOUT = 600  # 10 minutes

# Use proxy (if behind firewall)
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Download models one at a time manually
# (see manual download sections above)
```

---

### Error: "Model file corrupted" or "Hash mismatch"

**Problem:** Download was interrupted or corrupted

**Solutions:**
```bash
# Delete corrupted file and re-download
rm models/bisenet/79999_iter.pth
python download_models.py  # Re-run script

# Or download manually (see manual download sections)
```

---

### Error: "CUDA out of memory" when loading models

**Problem:** GPU doesn't have enough VRAM (24GB should be enough, but check)

**Solutions:**
```bash
# Check GPU memory
nvidia-smi

# Load models on CPU first, then move to GPU
# (modify your code to use map_location='cpu' first)

# Use half precision (FP16) for Stable Diffusion
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=torch.float16
)
```

---

### Error: "ModuleNotFoundError: No module named 'X'"

**Problem:** Missing Python package

**Solutions:**
```bash
# Install missing package
pip install X

# Or reinstall all requirements
pip install -r requirements.txt

# Common missing packages:
pip install onnxruntime  # For ArcFace ONNX
pip install insightface  # For ArcFace alternative
pip install open-clip-torch  # For CLIP
```

---

### Error: "Permission denied" when saving models

**Problem:** No write permission to models directory

**Solutions:**
```bash
# Fix permissions
chmod -R 755 models/

# Or run with appropriate permissions
sudo python download_models.py  # (not recommended, better to fix permissions)
```

---

## Manual Download Instructions

If the automated script fails, you can download models manually:

### BiSeNet

```bash
mkdir -p models/bisenet
cd models/bisenet

# Option 1: From GitHub releases
wget https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth

# Option 2: Clone repository
cd ../..
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
cp face-parsing.PyTorch/res/cp/79999_iter.pth models/bisenet/
```

### Stable Diffusion

```python
# In Python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    cache_dir="./models/stable_diffusion_inpainting"
)
```

### CLIP

```python
# In Python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai',
    cache_dir="./models/clip"
)
```

### ArcFace

```bash
# Option 1: ONNX model
mkdir -p models/arcface
wget -O models/arcface/arcface_r50_v1.onnx \
  https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r50_v1.onnx

# Option 2: InsightFace library (auto-downloads)
pip install insightface
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=0)"
```

---

## Disk Space Requirements

| Model | Size | Location |
|-------|------|----------|
| BiSeNet | ~395 MB | `models/bisenet/` |
| Stable Diffusion | ~7 GB | `~/.cache/huggingface/` |
| CLIP | ~350 MB | `models/clip/` or `~/.cache/torch/` |
| ArcFace | ~200 MB | `models/arcface/` or `~/.insightface/models/` |
| MediaPipe | Included | Package files |
| **Total** | **~50 GB** | (including cache) |

**Note:** Hugging Face cache can grow large. You can set custom cache directory:
```bash
export HF_HOME=/path/to/large/disk/.cache/huggingface
```

---

## Next Steps

After successful download:

1. ✅ **Verify all models load:**
   ```bash
   python -c "import torch; print('Testing models...')"
   # Run verification commands from above
   ```

2. ✅ **Test with sample code:**
   - Create a simple test script to load each model
   - Verify they work with your GPU/CPU setup

3. ✅ **Proceed to Step 3:** Build facial segmentation module

---

## Additional Resources

- **BiSeNet:** https://github.com/zllrunning/face-parsing.PyTorch
- **MediaPipe:** https://google.github.io/mediapipe/
- **Stable Diffusion:** https://huggingface.co/runwayml/stable-diffusion-inpainting
- **CLIP:** https://github.com/mlfoundations/open_clip
- **ArcFace/InsightFace:** https://github.com/deepinsight/insightface

---

## Support

If you encounter issues:

1. Check the log file: `logs/model_download_*.log`
2. Verify prerequisites are met
3. Try manual download for failed models
4. Check disk space and permissions
5. Review error messages carefully

---

**Last Updated:** 2025
**Environment:** Python 3.10, PyTorch 2.5.1+cu118, CUDA 11.8

