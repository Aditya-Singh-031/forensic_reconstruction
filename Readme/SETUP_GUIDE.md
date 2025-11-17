# 🛠️ SETUP & DEPLOYMENT GUIDE

## Quick Installation (5 minutes)

### Step 1: Verify Prerequisites
```bash
python3 --version  # Must be 3.10+
nvidia-smi         # Check GPU (if available)
df -h ~/           # Check ~50GB free space
```

### Step 2: Create Environment
```bash
cd ~/G14/forensic_reconstruction
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)"
```

---

## Running the System

### 1. Test Individual Modules

**Test Face Segmentation:**
```bash
python -m src.test_masking --image testimage.jpeg
```

**Test Landmarks:**
```bash
python src/test_masking.py  # Also tests landmarks
```

**Test Text-to-Face:**
```bash
python -m src.test_text_to_face --description "Adult male, 40 years old" --num_images 2
```

**Test Iterative Refinement:**
```bash
python src/test_iterative_refinement.py
# Select: 6 (Run all demos)
```

**Test Database & Matching:**
```bash
python src/test_database_matching.py
# Select: 6 (Run all demos)
```

### 2. Run Complete Pipeline

**Generate and refine face from description:**
```bash
python -m src.forensic_reconstruction_pipeline \
  --description "Adult male, 35-40 years, mustache, Indian complexion" \
  --num_faces 1 \
  --refine
```

**Without refinement (faster):**
```bash
python -m src.forensic_reconstruction_pipeline \
  --description "Young woman, 25-30, fair skin, glasses" \
  --num_faces 2
```

---

## Output Structure

```
output/
├── masking/                          # Face segmentation results
│   ├── image_testimage.png
│   ├── eyes_left_testimage.png
│   ├── mouth_testimage.png
│   └── ... (18+ feature masks)
│
├── inpainting/                       # Inpainted features
│   ├── masks/
│   └── results/
│
├── text_to_face/                     # Generated faces
│   ├── Adult_male__40_years__thick_mustache/
│   │   ├── 01.png
│   │   └── 02.png
│   └── ... (other descriptions)
│
├── iterative_refinement/             # Refinement sessions
│   ├── iteration_000_base.png
│   ├── iteration_001_mustache_thicker.png
│   ├── comparison_001.png
│   └── demo_*_session.json
│
├── pipeline_generated_faces/         # Pipeline outputs
│   ├── generated_face_00.png
│   └── generated_face_01.png
│
├── parsed_descriptions/              # Extracted attributes
│   └── parsed_results.json
│
├── face_database.json               # Simple database (STEP 10)
├── forensic_faces.db                # Multi-face database (SQLite)
├── forensic_database_export.json    # Exported faces
│
└── pipeline_results/                # Final results
    └── pipeline_results.json
```

---

## Module Dependencies

### Level 1: Core Libraries
```
torch → PyTorch (deep learning framework)
transformers → HuggingFace (model loading)
diffusers → Stable Diffusion (generation & inpainting)
```

### Level 2: Vision & Processing
```
opencv (cv2) → Image processing
pillow (PIL) → Image I/O
numpy → Numerical operations
scipy → Scientific computing
mediapipe → Facial landmarks
```

### Level 3: Application
```
src/face_segmentation.py
  ├─ Uses: transformers (SegFormer)
  ├─ Requires: torch, PIL, numpy
  └─ Output: Facial component masks

src/landmark_detector.py
  ├─ Uses: mediapipe
  ├─ Requires: cv2, numpy
  └─ Output: 468 facial keypoints

src/text_to_face.py
  ├─ Uses: diffusers (Stable Diffusion v1.5)
  ├─ Requires: torch, PIL
  └─ Output: 512×512 face image

src/face_inpainter.py
  ├─ Uses: diffusers (Stable Diffusion Inpaint)
  ├─ Requires: torch, PIL, numpy
  └─ Output: Inpainted face region

src/description_parser.py
  ├─ Uses: regex (built-in)
  ├─ Requires: nothing special
  └─ Output: Structured attributes

src/iterative_refinement.py
  ├─ Uses: text_to_face, face_inpainter
  ├─ Requires: PIL, Path, json
  └─ Output: Series of refined faces

src/multi_face_database.py
  ├─ Uses: transformers (CLIP)
  ├─ Requires: sqlite3, numpy, torch, PIL
  └─ Output: SQLite database with embeddings

src/advanced_matching.py
  ├─ Uses: multi_face_database, description_parser
  ├─ Requires: numpy, dataclasses
  └─ Output: Ranked match results
```

---

## Model Downloads (First Run)

Models are automatically downloaded on first use. Monitor disk space:

```bash
# Watch model downloads
watch -n 5 'ls -lh ~/.cache/huggingface/hub/ && du -sh ~/.cache/huggingface/'

# Free up space if needed
rm -rf ~/.cache/huggingface/hub/*inference*
rm -rf ~/.cache/torch/hub/*
```

### Expected Download Order

1. **SegFormer** (350 MB) - Face segmentation
2. **CLIP** (600 MB) - Text-image embeddings
3. **MediaPipe** (100 MB) - Facial landmarks
4. **Stable Diffusion v1.5** (~5.5 GB) - Text-to-face generation
5. **Stable Diffusion Inpaint** (~5.5 GB) - Feature refinement
6. **Whisper** (~3 GB) - Speech transcription (optional)

**Total: ~15-17 GB**

---

## GPU Optimization

### Check GPU Status
```bash
nvidia-smi -l 1  # Update every 1 second
```

### Monitor During Inference
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Reduce Memory Usage
```python
# Enable optimization techniques
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.enable_attention_slicing()  # Trade speed for memory
pipe.enable_vae_slicing()        # VAE memory optimization
```

### CPU Fallback (if GPU OOM)
```python
import torch
device = "cpu"  # Force CPU

pipe = StableDiffusionPipeline.from_pretrained(...).to(device)
# Much slower but will work
```

---

## Scaling to Production

### Single GPU Server
```bash
# Suitable for: < 100 requests/day
python -m src.forensic_reconstruction_pipeline --description "..."
```

### Multi-GPU Server
```bash
# Suitable for: < 1000 requests/day
# Use job queue (Celery/RQ) to distribute tasks
pip install celery redis
```

### Cloud Deployment (AWS EC2)
```bash
# AMI: Deep Learning AMI (Ubuntu 20.04)
# Instance: g4dn.xlarge (NVIDIA T4) or g4dn.2xlarge (2x T4)
# Storage: 100 GB EBS gp3
# Cost: ~$1/hour

# Setup
chmod +x setup.sh
./setup.sh  # Installs all dependencies

# Start API
python src/api/main.py --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build image (~10 GB uncompressed)
docker build -t forensic-ai:latest .

# Run container
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  forensic-ai:latest

# Access at http://localhost:8000/docs
```

### Database Scaling
```python
# SQLite: Good for < 100K faces
db = MultiFaceDatabase(db_path='forensic_faces.db')

# PostgreSQL + pgvector: For 1M+ faces
# pip install psycopg2-binary pgvector
# CREATE EXTENSION vector;
```

---

## Common Workflows

### Workflow 1: Single Face Reconstruction

```bash
# 1. Describe suspect
DESCRIPTION="Adult male, 40 years, thick mustache, Indian"

# 2. Generate face
python -m src.forensic_reconstruction_pipeline \
  --description "$DESCRIPTION" \
  --num_faces 1

# 3. View results
ls output/pipeline_generated_faces/
```

### Workflow 2: Batch Processing Multiple Descriptions

```python
# Create file: descriptions.txt
# Line 1: Adult male, 40 years old, mustache
# Line 2: Young female, 25-30, glasses
# ...

with open('descriptions.txt') as f:
    for desc in f:
        os.system(f"python -m src.forensic_reconstruction_pipeline --description '{desc.strip()}' --num_faces 1")
```

### Workflow 3: Database Search

```python
from src.multi_face_database import MultiFaceDatabase
from src.advanced_matching import AdvancedMatchingEngine
from src.description_parser import ForensicDescriptionParser

db = MultiFaceDatabase()
parser = ForensicDescriptionParser()
matcher = AdvancedMatchingEngine(db, parser)

query = "Adult male, 35 years, dark complexion"
results = matcher.match_description(query, top_k=10)

for result in results:
    print(f"{result.record_id}: {result.similarity_score:.3f}")
```

---

## Performance Tuning

### For Speed (Accuracy trade-off)
```python
generator.generate(
  description,
  num_inference_steps=20,  # Default 30, min 20
  guidance_scale=5.0       # Default 7.5, lower = faster
)
```

### For Quality (Speed trade-off)
```python
generator.generate(
  description,
  num_inference_steps=50,  # Higher = better, slower
  guidance_scale=10.0      # Higher = follow prompt more strictly
)
```

### Batch Processing
```python
descriptions = ["Male, 40", "Female, 25", "Male, 50"]

for desc in descriptions:
    # Each takes ~15 seconds
    result = pipeline.process_witness_description(desc)
```

---

## Monitoring & Logs

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# View logs
tail -f output/logs/forensic.log

# Monitor system resources
htop  # CPU/RAM
nvidia-smi  # GPU memory
```

---

## Backup & Recovery

### Backup Database
```bash
cp output/forensic_faces.db output/forensic_faces.db.backup
cp output/forensic_database_export.json output/forensic_database_export.backup.json
```

### Backup Generated Faces
```bash
tar -czf forensic_faces_backup.tar.gz output/text_to_face/
tar -czf iterative_backup.tar.gz output/iterative_refinement/
```

### Recovery
```bash
# Restore database
cp output/forensic_faces.db.backup output/forensic_faces.db

# Restore faces
tar -xzf forensic_faces_backup.tar.gz -C output/
```

---

## Troubleshooting Commands

```bash
# Check installation
python -c "import torch; print('✓ PyTorch'); import transformers; print('✓ Transformers'); from diffusers import StableDiffusionPipeline; print('✓ Diffusers')"

# Test GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check disk usage
du -sh ~/.cache/huggingface/
du -sh output/

# Test individual modules
python -m pytest src/  # Run unit tests
```

---

## Support Resources

- **Documentation:** See README.md
- **Issues:** Check Troubleshooting section in README.md
- **Model Docs:** https://huggingface.co/runwayml/stable-diffusion-v1-5
- **CLIP Docs:** https://huggingface.co/openai/clip-vit-base-patch32

---

**Last Updated:** November 13, 2025  
**Version:** 1.0
