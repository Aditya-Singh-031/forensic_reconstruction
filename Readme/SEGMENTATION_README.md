# Face Segmentation Module Documentation

Complete guide for using the SegFormer-based face segmentation module.

---

## Overview

The `FaceSegmenter` class provides semantic segmentation of face images into components like:
- Face skin
- Hair
- Eyes
- Nose
- Mouth
- Ears
- Neck
- Eyebrows
- Background

**Model:** SegFormer-B0 (nvidia/segformer-b0-finetuned-ade-512-512)
**Framework:** Hugging Face Transformers
**Input:** JPG, PNG, or other image formats
**Output:** Segmentation masks, colored visualizations, statistics

---

## Quick Start

### 1. Basic Usage

```python
from src.face_segmentation import FaceSegmenter

# Initialize segmenter (auto-detects GPU/CPU)
segmenter = FaceSegmenter()

# Segment an image
result = segmenter.segment("path/to/face/image.jpg")

# Save colored visualization
result['colored'].save("output_segmentation.png")

# Print statistics
print(result['statistics'])
```

### 2. Command Line Testing

```bash
# Activate environment
conda activate forensic310

# Navigate to project
cd /home/teaching/G14/forensic_reconstruction

# Test single image
python src/test_segmentation.py --image path/to/image.jpg

# Test with custom output directory
python src/test_segmentation.py --image path/to/image.jpg --output_dir output/

# Test batch of images
python src/test_segmentation.py --batch path/to/images/ --output_dir output/

# Use CPU instead of GPU
python src/test_segmentation.py --image path/to/image.jpg --device cpu
```

---

## Installation Verification

Before using, verify all dependencies are installed:

```bash
python src/test_segmentation.py --verify
```

**Required packages:**
- PyTorch 2.5.1+ (with CUDA support if using GPU)
- transformers
- Pillow (PIL)
- OpenCV (cv2)
- NumPy

---

## API Reference

### FaceSegmenter Class

#### Initialization

```python
segmenter = FaceSegmenter(
    model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    device=None,  # 'cuda', 'cpu', or None for auto-detect
    use_half_precision=True  # Use FP16 for faster inference (GPU only)
)
```

**Parameters:**
- `model_name` (str): Hugging Face model identifier
- `device` (str, optional): Device to use ('cuda', 'cpu', or None)
- `use_half_precision` (bool): Use FP16 precision for faster inference

#### Methods

##### `segment(image_path, return_masks=True, return_colored=True, return_statistics=True)`

Segment a face image into semantic components.

**Parameters:**
- `image_path` (str or Path): Path to input image
- `return_masks` (bool): Return individual component masks
- `return_colored` (bool): Return colored visualization
- `return_statistics` (bool): Return pixel statistics

**Returns:**
Dictionary with:
- `segmentation` (numpy.ndarray): Segmentation mask (H, W)
- `colored` (PIL.Image): Colored visualization (if return_colored=True)
- `masks` (dict): Individual component masks (if return_masks=True)
- `statistics` (dict): Pixel counts per component (if return_statistics=True)
- `processing_time` (float): Time taken in seconds
- `original_size` (tuple): Original image size (width, height)

**Example:**
```python
result = segmenter.segment("face.jpg")

# Access results
print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Statistics: {result['statistics']}")

# Save outputs
result['colored'].save("segmentation.png")

# Access individual masks
hair_mask = result['masks']['hair']
```

##### `segment_batch(image_paths, batch_size=4)`

Segment multiple images in batches.

**Parameters:**
- `image_paths` (list): List of image paths
- `batch_size` (int): Number of images per batch

**Returns:**
List of segmentation results (one per image)

**Example:**
```python
image_paths = ["face1.jpg", "face2.jpg", "face3.jpg"]
results = segmenter.segment_batch(image_paths, batch_size=2)
```

---

## Output Format

### Segmentation Mask

The segmentation mask is a numpy array of shape `(H, W)` where each pixel value corresponds to a face component:

```python
# Component IDs
0: background
1: face_skin
2: hair
3: eyes
4: nose
5: mouth
6: ears
7: neck
8: eyebrows
```

### Component Masks

Individual binary masks for each component (0 = not present, 255 = present):

```python
masks = {
    'background': numpy.ndarray,  # (H, W) uint8
    'face_skin': numpy.ndarray,
    'hair': numpy.ndarray,
    'eyes': numpy.ndarray,
    # ... etc
}
```

### Statistics

Pixel counts and percentages for each component:

```python
statistics = {
    'background': {'pixel_count': 12345, 'percentage': 45.67},
    'face_skin': {'pixel_count': 5678, 'percentage': 21.23},
    # ... etc
}
```

### Colored Visualization

PIL Image with colored overlay showing segmentation:

- Background: Black
- Face skin: Light skin tone
- Hair: Dark brown/black
- Eyes: Blue
- Nose: Light skin
- Mouth: Red
- Ears: Light skin
- Neck: Medium skin
- Eyebrows: Dark brown

---

## Example Usage

### Example 1: Basic Segmentation

```python
from src.face_segmentation import FaceSegmenter

# Initialize
segmenter = FaceSegmenter()

# Segment
result = segmenter.segment("sample_face.jpg")

# Display statistics
for component, stats in result['statistics'].items():
    if stats['pixel_count'] > 0:
        print(f"{component}: {stats['percentage']:.2f}%")

# Save visualization
result['colored'].save("output.png")
```

### Example 2: Extract Specific Component

```python
from src.face_segmentation import FaceSegmenter
from PIL import Image

segmenter = FaceSegmenter()
result = segmenter.segment("face.jpg", return_masks=True)

# Extract hair mask
hair_mask = result['masks']['hair']

# Save as image
hair_img = Image.fromarray(hair_mask, mode='L')
hair_img.save("hair_mask.png")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from src.face_segmentation import FaceSegmenter

segmenter = FaceSegmenter()

# Get all images in directory
image_dir = Path("images/")
image_paths = list(image_dir.glob("*.jpg"))

# Process in batches
results = segmenter.segment_batch(image_paths, batch_size=4)

# Process results
for i, result in enumerate(results):
    if result:
        result['colored'].save(f"output_{i}.png")
```

### Example 4: Custom Device

```python
from src.face_segmentation import FaceSegmenter

# Force CPU usage
segmenter = FaceSegmenter(device='cpu')

# Or force GPU
segmenter = FaceSegmenter(device='cuda')
```

---

## Performance

### Typical Processing Times

- **GPU (RTX A5000, FP16):** ~0.1-0.3 seconds per image
- **GPU (RTX A5000, FP32):** ~0.2-0.5 seconds per image
- **CPU (modern):** ~2-5 seconds per image

### Memory Usage

- **Model size:** ~50MB (SegFormer-B0)
- **GPU memory:** ~500MB-1GB (depending on image size)
- **CPU memory:** ~1-2GB

### Optimization Tips

1. **Use FP16 precision** (default on GPU) for 2x speedup
2. **Batch processing** for multiple images
3. **Resize large images** before processing if needed
4. **Use GPU** when available for 10-20x speedup

---

## Troubleshooting

### Error: "CUDA out of memory"

**Problem:** GPU doesn't have enough VRAM

**Solutions:**
```python
# Use CPU instead
segmenter = FaceSegmenter(device='cpu')

# Or use FP32 instead of FP16
segmenter = FaceSegmenter(use_half_precision=False)

# Or resize image before processing
from PIL import Image
img = Image.open("large_image.jpg")
img = img.resize((512, 512))
img.save("resized.jpg")
```

### Error: "Model not found"

**Problem:** SegFormer model not downloaded

**Solution:**
```python
# Model will auto-download on first use
# Make sure you have internet connection
segmenter = FaceSegmenter()
```

### Error: "No person detected"

**Problem:** Image doesn't contain a clear face

**Solutions:**
- Use images with clear, front-facing faces
- Ensure good lighting
- Try different image sizes
- Check if image is corrupted

### Error: "ImportError: No module named 'transformers'"

**Problem:** Missing dependencies

**Solution:**
```bash
pip install transformers pillow opencv-python numpy torch
```

### Slow Performance

**Solutions:**
1. Check if GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. Use FP16 precision (default on GPU)

3. Process images in batches

4. Resize very large images before processing

---

## Advanced Usage

### Custom Model

```python
# Use different SegFormer model
segmenter = FaceSegmenter(
    model_name="nvidia/segformer-b1-finetuned-ade-512-512"  # Larger model
)
```

### Save/Load Segmentation

```python
import numpy as np
from PIL import Image

# Save segmentation mask
np.save("segmentation.npy", result['segmentation'])

# Load segmentation mask
seg = np.load("segmentation.npy")

# Convert to image
seg_img = Image.fromarray(seg, mode='L')
seg_img.save("segmentation.png")
```

### Combine with Other Modules

```python
from src.face_segmentation import FaceSegmenter
# ... other imports

segmenter = FaceSegmenter()
result = segmenter.segment("face.jpg")

# Use segmentation mask for inpainting
hair_mask = result['masks']['hair']
# ... use with inpainting module
```

---

## Expected Output

When running `test_segmentation.py`, you should see:

```
============================================================
Testing segmentation on: sample_face.jpg
============================================================

Running segmentation...
✓ Segmentation completed!
  Processing time: 0.25 seconds
  Original size: 512x512

Component Statistics:
  Component            Pixels          Percentage
  ---------------------------------------------
  background           12345           45.67%
  face_skin            5678            21.23%
  hair                 3456            12.89%
  eyes                 890             3.33%
  nose                 567             2.12%
  mouth                234             0.87%
  ears                 123             0.46%
  neck                 456             1.71%

✓ Saved colored visualization: output/segmentation_colored_sample_face.png

Saving individual component masks...
  ✓ background: output/masks/background_sample_face.png
  ✓ face_skin: output/masks/face_skin_sample_face.png
  ✓ hair: output/masks/hair_sample_face.png
  ...

============================================================
✓ All outputs saved successfully!
============================================================
```

---

## File Structure

After running tests, output directory structure:

```
output/segmentation/
├── segmentation_colored_sample_face.png  # Colored visualization
├── segmentation_raw_sample_face.png      # Raw mask
└── masks/
    ├── background_sample_face.png
    ├── face_skin_sample_face.png
    ├── hair_sample_face.png
    ├── eyes_sample_face.png
    ├── nose_sample_face.png
    ├── mouth_sample_face.png
    ├── ears_sample_face.png
    ├── neck_sample_face.png
    └── eyebrows_sample_face.png
```

---

## Next Steps

After successful segmentation:

1. ✅ **Verify outputs:** Check that masks look correct
2. ✅ **Test with different images:** Try various face images
3. ✅ **Proceed to Step 4:** Add facial landmark detection
4. ✅ **Integrate with pipeline:** Use segmentation in full pipeline

---

## Support

If you encounter issues:

1. Check this documentation
2. Run verification: `python src/test_segmentation.py --verify`
3. Check log files in `logs/` directory
4. Review error messages carefully
5. Ensure all dependencies are installed

---

**Last Updated:** 2025
**Module Version:** 1.0.0
**Compatible with:** Python 3.10+, PyTorch 2.5.1+, CUDA 11.8+

