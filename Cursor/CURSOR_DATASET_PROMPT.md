# CURSOR PROMPT: Create Facial Features Dataset from FFHQ and CelebA-HQ

## OBJECTIVE
Create a comprehensive facial features dataset by:
1. Downloading images from FFHQ and Multi-Modal-CelebA-HQ
2. Extracting facial features using existing segmentation and landmark detection models
3. Storing results on `/dev/sda` (9.1TB available)
4. Creating symlinks in current directory for easy access
5. Preparing structured dataset for future model fine-tuning

## CONTEXT FILES
- `src/face_segmentation.py` - FaceSegmenter class (SegFormer model)
- `src/landmark_detector.py` - LandmarkDetector class (MediaPipe 468 landmarks)
- `src/mask_generator.py` - MaskGenerator for feature extraction

## REQUIREMENTS

### Storage Structure
```
/DATA/facial_features_dataset/
├── raw_images/
│   ├── ffhq/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── celeba_hq/
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ...
├── features/
│   ├── segmentation/
│   │   ├── 00000_segmentation.png
│   │   ├── 00000_segmentation_mask.npy
│   │   └── ...
│   ├── landmarks/
│   │   ├── 00000_landmarks.json
│   │   ├── 00000_landmarks_visual.png
│   │   └── ...
│   ├── eyes_left/
│   │   ├── 00000_eyes_left.png
│   │   ├── 00000_eyes_left_mask.png
│   │   └── ...
│   ├── eyes_right/
│   ├── eyebrows_left/
│   ├── eyebrows_right/
│   ├── nose/
│   ├── mouth_outer/
│   ├── mouth_inner/
│   ├── ears_left/
│   ├── ears_right/
│   ├── face_contour/
│   └── jawline/
├── metadata/
│   ├── dataset_info.json
│   ├── processing_log.csv
│   └── statistics.json
└── annotations/
    ├── landmarks_annotations.csv
    └── features_summary.csv
```

### Symlink in Current Directory
```
~/G14/forensic_reconstruction/dataset -> /DATA/facial_features_dataset/
```

## IMPLEMENTATION REQUIREMENTS

### 1. Dataset Downloader Module (`src/dataset_downloader.py`)

Create a module that:
- Downloads FFHQ dataset (70,000 high-quality face images 1024×1024)
  - Source: https://github.com/NVlabs/ffhq-dataset
  - Use Kaggle API or direct download
- Downloads Multi-Modal-CelebA-HQ (30,000 images)
  - Source: https://github.com/IIGROUP/MM-CelebA-HQ-Dataset
- Implements resume capability (continue interrupted downloads)
- Shows progress bar with download speed
- Validates downloaded files (check corruption)
- Stores in `/DATA/facial_features_dataset/raw_images/`

### 2. Feature Extraction Pipeline (`src/create_features_dataset.py`)

Main pipeline that:

**For each image:**
1. Load image from raw_images/
2. Run FaceSegmenter:
   - Save full segmentation mask as `{image_name}_segmentation.png`
   - Save numpy array as `{image_name}_segmentation_mask.npy`
   - Extract individual feature masks (eyes, nose, mouth, etc.)
3. Run LandmarkDetector:
   - Save 468 landmarks as JSON `{image_name}_landmarks.json`
   - Save visualization as `{image_name}_landmarks_visual.png`
   - Group landmarks by feature (eyes, eyebrows, nose, etc.)
4. For each facial feature:
   - Extract region using landmarks + segmentation
   - Save cropped feature image: `{image_name}_{feature_type}.png`
   - Save feature mask: `{image_name}_{feature_type}_mask.png`
   - Save bounding box coordinates in metadata

**Features to extract:**
- `eyes_left` (16 landmarks)
- `eyes_right` (16 landmarks)
- `eyebrows_left` (10 landmarks)
- `eyebrows_right` (10 landmarks)
- `nose` (multiple landmarks)
- `mouth_outer` (multiple landmarks)
- `mouth_inner` (multiple landmarks)
- `ears_left` (8 landmarks)
- `ears_right` (8 landmarks)
- `face_contour` (36 landmarks)
- `jawline` (subset of face contour)

### 3. Metadata Management (`src/dataset_metadata.py`)

Create comprehensive metadata:

**dataset_info.json:**
```json
{
  "dataset_name": "Facial Features Dataset",
  "version": "1.0",
  "created_date": "2025-11-17",
  "total_images": 100000,
  "sources": ["FFHQ", "CelebA-HQ"],
  "features_extracted": [...],
  "storage_path": "/DATA/facial_features_dataset/",
  "models_used": {
    "segmentation": "nvidia/segformer-b0-finetuned-ade-512-512",
    "landmarks": "MediaPipe Face Mesh v1"
  }
}
```

**processing_log.csv:**
```csv
image_name,source,processed_date,segmentation_success,landmarks_success,features_extracted,processing_time_sec,errors
00000.png,ffhq,2025-11-17 22:30:00,True,True,11,2.5,None
```

**landmarks_annotations.csv:**
```csv
image_name,landmark_id,x,y,z,feature_group
00000.png,0,120.5,200.3,0.1,face_contour
00000.png,1,121.0,201.0,0.2,face_contour
```

**features_summary.csv:**
```csv
image_name,feature_type,bbox_x,bbox_y,bbox_w,bbox_h,mask_path,image_path
00000.png,eyes_left,100,150,50,30,features/eyes_left/00000_eyes_left_mask.png,features/eyes_left/00000_eyes_left.png
```

### 4. Batch Processing System

Implement efficient batch processing:
- Process images in batches of 10-50 (optimize for GPU)
- Multiprocessing for CPU-bound tasks
- Progress tracking with tqdm
- Error handling: skip corrupted images, log errors
- Resume capability: save checkpoint every 100 images
- Memory management: clear cache periodically

### 5. Quality Validation (`src/validate_dataset.py`)

Validate extracted features:
- Check all expected files exist
- Verify image dimensions
- Validate landmark coordinates (within bounds)
- Check for corrupted files
- Generate quality report

### 6. Symlink Setup Script (`scripts/setup_dataset_symlink.sh`)

```bash
#!/bin/bash
# Create symlink from /DATA to current directory
ln -s /DATA/facial_features_dataset ~/G14/forensic_reconstruction/dataset
echo "Symlink created: ~/G14/forensic_reconstruction/dataset -> /DATA/facial_features_dataset/"
```

## TECHNICAL SPECIFICATIONS

### Image Processing
- Input formats: PNG, JPG, JPEG
- Output format: PNG for images, NPY for arrays, JSON for metadata
- Image size: Maintain original resolution, save downscaled versions too
- Color space: RGB

### Landmark Format (JSON)
```json
{
  "image_name": "00000.png",
  "num_landmarks": 468,
  "landmarks": [
    {"id": 0, "x": 120.5, "y": 200.3, "z": 0.1, "group": "face_contour"},
    ...
  ],
  "feature_groups": {
    "eyes_left": [33, 133, 160, ...],
    "eyes_right": [362, 263, 387, ...],
    ...
  },
  "bbox": {"x": 50, "y": 100, "width": 400, "height": 500}
}
```

### Segmentation Mask Format
- NumPy array: (H, W) with class labels 0-18
- PNG visualization: RGB colored by class
- Individual feature masks: Binary (0=background, 255=feature)

## PERFORMANCE REQUIREMENTS

### Speed Targets
- Process 1 image in 1-3 seconds
- Batch of 50 images in 60-150 seconds
- Complete dataset (100K images) in 48-72 hours

### Memory Management
- Max GPU memory: 10GB
- Max CPU memory per worker: 4GB
- Clear cache every 1000 images

### Error Handling
- Skip images that fail processing
- Log all errors with traceback
- Continue processing rest of dataset
- Generate error summary report

## CODE STRUCTURE

Create these files:

1. `src/dataset_downloader.py` - Download FFHQ and CelebA-HQ
2. `src/create_features_dataset.py` - Main pipeline
3. `src/dataset_metadata.py` - Metadata management
4. `src/validate_dataset.py` - Quality validation
5. `src/batch_processor.py` - Efficient batch processing
6. `scripts/setup_dataset_symlink.sh` - Symlink setup
7. `scripts/run_dataset_creation.sh` - Full pipeline runner

## USAGE EXAMPLE

```bash
# 1. Setup storage
bash scripts/setup_dataset_symlink.sh

# 2. Download datasets
python -m src.dataset_downloader --dataset ffhq --output /DATA/facial_features_dataset/raw_images/

# 3. Create features dataset
python -m src.create_features_dataset \
  --input /DATA/facial_features_dataset/raw_images/ \
  --output /DATA/facial_features_dataset/ \
  --batch-size 50 \
  --num-workers 4

# 4. Validate
python -m src.validate_dataset --path /DATA/facial_features_dataset/

# 5. Access via symlink
ls ~/G14/forensic_reconstruction/dataset/
```

## TESTING

Create `tests/test_dataset_creation.py`:
- Test single image processing
- Test batch processing
- Test metadata generation
- Test symlink access
- Test resume capability

## DOCUMENTATION

Create `DATASET_README.md`:
- Dataset structure explanation
- Feature extraction methodology
- Usage examples
- Statistics and visualizations
- Known issues and limitations

## GITHUB INTEGRATION

For Git Large Files:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "dataset/raw_images/*.png"
git lfs track "dataset/features/**/*.png"
git lfs track "dataset/**/*.npy"

# Add .gitattributes
git add .gitattributes
```

## OPTIMIZATION TIPS

1. Use GPU for segmentation (FaceSegmenter)
2. Use CPU for landmarks (MediaPipe is CPU-optimized)
3. Multiprocessing for I/O operations
4. Cache frequently used models in memory
5. Write files in batches, not one-by-one
6. Use SSD for temporary files, HDD for final storage

## DELIVERABLES

1. ✅ Fully structured dataset on /DATA/
2. ✅ Symlink in current directory
3. ✅ Comprehensive metadata files
4. ✅ Processing logs and statistics
5. ✅ Validation report
6. ✅ Documentation
7. ✅ Scripts for easy replication

## ESTIMATED TIMELINE

- Setup & download: 6-12 hours
- Feature extraction (100K images): 48-72 hours
- Validation & documentation: 2-4 hours
- **Total: ~3-4 days**

---

**IMPORTANT NOTES:**
- Use existing FaceSegmenter and LandmarkDetector classes (already implemented)
- Store everything on /DATA/ (9.1TB available)
- Create symlink for easy access
- Prepare for future fine-tuning (clean structure)
- Handle errors gracefully (skip bad images)

**START BY:**
1. Creating storage structure on /DATA/
2. Setting up symlink
3. Implementing downloader
4. Testing on 10-100 images first
5. Then scale to full dataset

Generate all code, scripts, and documentation following these requirements exactly.
