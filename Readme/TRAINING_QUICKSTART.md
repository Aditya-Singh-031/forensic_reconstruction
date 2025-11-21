# Training Pipeline - Quick Start Guide

## 📋 Prerequisites

Before starting training, ensure:
- ✅ Feature extraction complete (`src/create_features_dataset.py` run successfully)
- ✅ Dataset structure exists: `/DATA/facial_features_dataset/features/`
- ✅ GPU available (8GB+ VRAM recommended)
- ✅ Python packages installed (see Installation section)

## 🚀 Step-by-Step Setup

### **Step 1: Install Dependencies**

```bash
cd /path/to/forensic_reconstruction

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install -r requirements_training.txt
```

**requirements_training.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.23.0
opencv-python>=4.7.0
tqdm>=4.65.0
tensorboard>=2.12.0
lpips>=0.1.4
```

---

### **Step 2: Index Your Dataset**

This creates fast lookup databases for training:

```bash
python src/dataset_indexer.py
```

**Expected output:**
```
================================================================
BUILDING FEATURE INDEX
================================================================
Found 1000 face contour images
Indexing features: 100%|██████████| 1000/1000
Complete: 950
Incomplete: 50

Statistics:
  Total images: 950
  Complete triplets (contour+jawline): 950

Split sizes:
  Train: 760 (80.0%)
  Val:   95 (10.0%)
  Test:  95 (10.0%)

✓ Saved index (950 images)
================================================================
```

**Outputs created:**
- `/DATA/facial_features_dataset/metadata/feature_index.json`
- `/DATA/facial_features_dataset/metadata/statistics.json`
- `/DATA/facial_features_dataset/metadata/train_val_test_split.json`

---

### **Step 3: Test Corruption Engine**

Verify feature corruption works correctly:

```bash
python src/corruption_engine.py
```

**Expected output:**
```
Loaded feature index with 950 images
Testing corruption on: 00001

Corruption level 1:
  Corrupted features: ['eyes_left', 'nose']
  Corrupted image size: (1024, 1024)
  Target image size: (1024, 1024)
  Saved visualization to /DATA/.../corruption_level_1_00001.png

Corruption level 2:
  Corrupted features: ['eyes_left', 'eyes_right', 'nose', 'mouth_outer']
  ...
```

**Check visualizations:**
```bash
ls /DATA/facial_features_dataset/visualizations/
# Should see: corruption_level_1_00001.png, corruption_level_2_00001.png, etc.
```

---

### **Step 4: Test DataLoader**

Verify PyTorch data pipeline works:

```bash
python src/data_loader.py
```

**Expected output:**
```
Creating dataloaders:
  Batch size: 2
  Num workers: 0
  Corruption level: 2

Created dataloaders:
  Train: 380 batches (760 samples)
  Val:   47 batches (95 samples)
  Test:  47 batches (95 samples)

Sample batch:
  Corrupted shape: torch.Size([2, 3, 256, 256])
  Target shape: torch.Size([2, 3, 256, 256])
  Mask shape: torch.Size([2, 1, 256, 256])

Dataloader test passed! ✓
```

---

### **Step 5: Verify Dataset Integrity**

Run comprehensive checks:

```bash
python src/verify_dataset.py
```

This will:
- Check all required files exist
- Validate image sizes and formats
- Verify feature-target pairs match
- Report any missing or corrupted files

---

## 🎯 Next Steps

Once all tests pass, you're ready for:

1. **Loss Functions** (`src/losses.py`) - Multi-task loss implementation
2. **Model Architecture** (`src/models.py`) - U-Net or Stable Diffusion setup
3. **Training Script** (`src/train.py`) - Main training loop with curriculum learning
4. **Evaluation** (`src/evaluation.py`) - Metrics and validation

---

## 📊 Expected Dataset Structure After Indexing

```
/DATA/facial_features_dataset/
├── features/
│   ├── eyes_left/
│   ├── eyes_right/
│   ├── eyebrows_left/
│   ├── eyebrows_right/
│   ├── nose/
│   ├── mouth_outer/
│   ├── mouth_inner/
│   ├── ears_left/
│   ├── ears_right/
│   ├── face_contour/      # Base images
│   └── jawline/           # Ground truth targets
├── metadata/
│   ├── feature_index.json         # ← NEW: Fast lookup
│   ├── statistics.json            # ← NEW: Dataset stats
│   ├── train_val_test_split.json  # ← NEW: Train/val/test splits
│   └── processing_log.csv         # From extraction
├── visualizations/                # ← NEW: Corruption examples
│   ├── corruption_level_1_00001.png
│   ├── corruption_level_2_00001.png
│   └── corruption_level_3_00001.png
└── raw_images/
    ├── ffhq/
    └── celeba_hq/
```

---

## ⚠️ Troubleshooting

### Issue: "Feature index not found"
**Solution:** Run `python src/dataset_indexer.py` first

### Issue: "No complete samples found"
**Solution:** Check feature extraction completed successfully. Re-run:
```bash
python -m src.create_features_dataset --max-images 10
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in training config:
```python
batch_size = 4  # Instead of 8
```

### Issue: "Missing feature for image"
**Solution:** Some images may have failed during extraction. Check:
```bash
cat /DATA/facial_features_dataset/metadata/processing_log.csv | grep "False"
```

---

## 🔍 Validation Checklist

Before training, verify:

- [ ] `feature_index.json` exists with >500 images
- [ ] Train/val/test splits created (80/10/10)
- [ ] Corruption visualizations look correct
- [ ] DataLoader test passes
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Disk space sufficient (models ~2-5GB)

---

## 📞 Need Help?

If tests fail, check:
1. Dataset extraction logs
2. File permissions
3. GPU drivers
4. Python package versions

**Ready to train?** Proceed to `src/train.py` setup next!
