# Facial Landmark Detection Module Documentation

Complete guide for using the MediaPipe Face Mesh-based landmark detection module.

---

## Overview

The `LandmarkDetector` class provides 468 3D facial landmark detection using MediaPipe Face Mesh. It identifies and highlights key facial features:

- **Eyes** (left, right) - 16 landmarks each
- **Eyebrows** (left, right) - 10 landmarks each
- **Nose** (tip, base) - Multiple landmarks
- **Mouth** (outer, inner) - Multiple landmarks
- **Ears** (left, right) - 8 landmarks each
- **Face contour/jawline** - 36 landmarks

**Model:** MediaPipe Face Mesh (built-in, no download needed)
**Input:** JPG, PNG, or other image formats
**Output:** 468 3D landmarks (x, y, z coordinates), feature groups, visualizations

---

## Quick Start

### 1. Basic Usage

```python
from src.landmark_detector import LandmarkDetector

# Initialize detector
detector = LandmarkDetector()

# Detect landmarks
result = detector.detect("path/to/face/image.jpg")

# Access results
print(f"Detected {len(result['landmarks'])} landmarks")
print(f"Processing time: {result['processing_time']:.2f}s")

# Save visualization
result['visualization'].save("output_landmarks.png")

# Access feature groups
left_eye = result['groups']['left_eye']
print(f"Left eye center: {left_eye['center']}")
```

### 2. Command Line Testing

```bash
# Activate environment
conda activate forensic310

# Navigate to project
cd /home/teaching/G14/forensic_reconstruction

# Test single image
python src/test_landmarks.py --image path/to/image.jpg

# Test with custom output directory
python src/test_landmarks.py --image path/to/image.jpg --output_dir output/

# Test batch of images
python src/test_landmarks.py --batch path/to/images/ --output_dir output/

# Allow multiple faces (picks largest)
python src/test_landmarks.py --image path/to/image.jpg --max_faces 5
```

---

## Installation Verification

Before using, verify all dependencies are installed:

```bash
python src/test_landmarks.py --verify
```

**Required packages:**
- MediaPipe (included in requirements.txt)
- NumPy
- OpenCV (cv2)
- Pillow (PIL)

---

## API Reference

### LandmarkDetector Class

#### Initialization

```python
detector = LandmarkDetector(
    static_image_mode=True,      # True for images, False for video
    max_num_faces=1,             # Maximum faces to detect
    refine_landmarks=True,        # Refine around eyes/lips
    min_detection_confidence=0.5, # Min confidence for detection
    min_tracking_confidence=0.5   # Min confidence for tracking
)
```

**Parameters:**
- `static_image_mode` (bool): True for static images, False for video streams
- `max_num_faces` (int): Maximum number of faces to detect (default: 1)
- `refine_landmarks` (bool): Refine landmarks around eyes and lips (default: True)
- `min_detection_confidence` (float): Minimum confidence [0.0, 1.0] (default: 0.5)
- `min_tracking_confidence` (float): Minimum tracking confidence [0.0, 1.0] (default: 0.5)

#### Methods

##### `detect(image_path, return_visualization=True, return_groups=True, return_coordinates=True)`

Detect facial landmarks in an image.

**Parameters:**
- `image_path` (str or Path): Path to input image
- `return_visualization` (bool): Return annotated image
- `return_groups` (bool): Return feature groups (eyes, nose, etc.)
- `return_coordinates` (bool): Return raw coordinates

**Returns:**
Dictionary with:
- `landmarks` (numpy.ndarray): All 468 landmarks in pixel coordinates (N, 3) - [x, y, z]
- `landmarks_normalized` (numpy.ndarray): Normalized coordinates [0.0, 1.0] (N, 3)
- `groups` (dict): Feature groups (eyes, nose, mouth, etc.)
- `visualization` (PIL.Image): Annotated image with landmarks
- `num_faces` (int): Number of faces detected
- `face_index` (int): Index of selected face (if multiple)
- `processing_time` (float): Time taken in seconds
- `image_size` (tuple): Original image size (width, height)
- `coordinates` (dict): Both pixel and normalized coordinates

**Example:**
```python
result = detector.detect("face.jpg")

# Access all landmarks
landmarks = result['landmarks']  # Shape: (468, 3)
print(f"First landmark: {landmarks[0]}")  # [x, y, z]

# Access feature groups
left_eye = result['groups']['left_eye']
print(f"Left eye landmarks: {left_eye['landmarks_pixel']}")
print(f"Left eye center: {left_eye['center']}")
print(f"Left eye bbox: {left_eye['bbox']}")

# Save visualization
result['visualization'].save("output.png")
```

##### `get_landmark_coordinates(result, format='numpy')`

Get landmark coordinates in various formats.

**Parameters:**
- `result`: Result dictionary from `detect()` method
- `format`: Output format ('numpy', 'list', 'csv')

**Returns:**
Landmarks in requested format

**Example:**
```python
result = detector.detect("face.jpg")

# Get as NumPy array (default)
coords = detector.get_landmark_coordinates(result, format='numpy')

# Get as Python list
coords_list = detector.get_landmark_coordinates(result, format='list')

# Get as CSV string
coords_csv = detector.get_landmark_coordinates(result, format='csv')
print(coords_csv)
```

---

## Feature Groups

The detector automatically identifies and groups landmarks for key facial features:

### Available Groups

| Group | Description | Landmark Count |
|-------|-------------|----------------|
| `left_eye` | Left eye region | 16 |
| `right_eye` | Right eye region | 16 |
| `left_eyebrow` | Left eyebrow | 10 |
| `right_eyebrow` | Right eyebrow | 10 |
| `nose_tip` | Nose tip region | 10 |
| `nose_base` | Nose base | 4 |
| `mouth_outer` | Outer mouth contour | 29 |
| `mouth_inner` | Inner mouth | 10 |
| `left_ear` | Left ear region | 8 |
| `right_ear` | Right ear region | 8 |
| `face_contour` | Face outline/jawline | 36 |
| `jawline` | Alias for face_contour | 36 |

### Accessing Feature Groups

```python
result = detector.detect("face.jpg")
groups = result['groups']

# Access specific feature
left_eye = groups['left_eye']

# Get landmarks for this feature
landmarks = left_eye['landmarks_pixel']  # List of [x, y, z] coordinates

# Get bounding box
bbox = left_eye['bbox']
print(f"Left eye bbox: x={bbox['x_min']}-{bbox['x_max']}, y={bbox['y_min']}-{bbox['y_max']}")

# Get center point
center = left_eye['center']
print(f"Left eye center: ({center['x']}, {center['y']})")
```

---

## Output Format

### Landmark Coordinates

Landmarks are returned in two coordinate systems:

1. **Pixel Coordinates** (`landmarks`): Absolute pixel positions
   - X: [0, image_width]
   - Y: [0, image_height]
   - Z: Relative depth (scaled to image width)

2. **Normalized Coordinates** (`landmarks_normalized`): Relative positions [0.0, 1.0]
   - X: [0.0, 1.0] (0.0 = left, 1.0 = right)
   - Y: [0.0, 1.0] (0.0 = top, 1.0 = bottom)
   - Z: Relative depth

**Example:**
```python
result = detector.detect("face.jpg")

# Pixel coordinates
pixel_coords = result['landmarks']  # Shape: (468, 3)
# [[x1, y1, z1], [x2, y2, z2], ...]

# Normalized coordinates
norm_coords = result['landmarks_normalized']  # Shape: (468, 3)
# [[x1_norm, y1_norm, z1_norm], ...]
```

### Visualization

The visualization includes:
- All 468 landmarks drawn as points
- Face mesh connections
- Feature group bounding boxes (different colors)
- Feature group labels
- Center points for each feature

**Color Scheme:**
- Eyes: Blue
- Eyebrows: Green
- Nose: Red
- Mouth: Magenta
- Ears: Cyan
- Face contour: Yellow

---

## Example Usage

### Example 1: Basic Detection

```python
from src.landmark_detector import LandmarkDetector

# Initialize
detector = LandmarkDetector()

# Detect
result = detector.detect("sample_face.jpg")

# Print summary
print(f"Detected {len(result['landmarks'])} landmarks")
print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Faces detected: {result['num_faces']}")

# Save visualization
result['visualization'].save("landmarks_output.png")
```

### Example 2: Extract Specific Features

```python
from src.landmark_detector import LandmarkDetector
import numpy as np

detector = LandmarkDetector()
result = detector.detect("face.jpg", return_groups=True)

# Extract left eye landmarks
left_eye = result['groups']['left_eye']
eye_landmarks = np.array(left_eye['landmarks_pixel'])

# Calculate eye center
eye_center_x = eye_landmarks[:, 0].mean()
eye_center_y = eye_landmarks[:, 1].mean()

print(f"Left eye center: ({eye_center_x:.2f}, {eye_center_y:.2f})")
```

### Example 3: Save Coordinates

```python
from src.landmark_detector import LandmarkDetector
import numpy as np

detector = LandmarkDetector()
result = detector.detect("face.jpg")

# Save as NumPy array
np.save("landmarks.npy", result['landmarks'])

# Save as CSV
np.savetxt("landmarks.csv", result['landmarks'], 
           delimiter=',', fmt='%.2f', header='x,y,z', comments='')

# Save using built-in method
csv_string = detector.get_landmark_coordinates(result, format='csv')
with open("landmarks.csv", "w") as f:
    f.write(csv_string)
```

### Example 4: Multiple Faces

```python
from src.landmark_detector import LandmarkDetector

# Allow up to 5 faces
detector = LandmarkDetector(max_num_faces=5)

result = detector.detect("group_photo.jpg")

print(f"Detected {result['num_faces']} faces")
print(f"Selected face index: {result['face_index']} (largest)")

# The detector automatically selects the largest face
# All landmarks are from the selected face
```

### Example 5: Combine with Segmentation

```python
from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector

# Initialize both
segmenter = FaceSegmenter()
detector = LandmarkDetector()

# Process same image
image_path = "face.jpg"

# Segmentation
seg_result = segmenter.segment(image_path)

# Landmark detection
landmark_result = detector.detect(image_path)

# Use together (e.g., align masks with landmarks)
hair_mask = seg_result['masks']['hair']
left_eye_landmarks = landmark_result['groups']['left_eye']['landmarks_pixel']
```

---

## Performance

### Typical Processing Times

- **CPU (modern):** ~50-200ms per image
- **CPU (older):** ~200-500ms per image
- **GPU:** MediaPipe uses CPU by default (optimized for CPU)

### Memory Usage

- **Model size:** Included in MediaPipe package (~10MB)
- **Memory per image:** ~50-100MB (depending on image size)

### Optimization Tips

1. **Disable refinement** for faster processing:
   ```python
   detector = LandmarkDetector(refine_landmarks=False)
   ```

2. **Lower confidence threshold** if faces aren't detected:
   ```python
   detector = LandmarkDetector(min_detection_confidence=0.3)
   ```

3. **Resize large images** before processing:
   ```python
   from PIL import Image
   img = Image.open("large_image.jpg")
   img = img.resize((512, 512))
   img.save("resized.jpg")
   ```

---

## Troubleshooting

### Error: "No face detected in image"

**Problem:** MediaPipe couldn't detect a face

**Solutions:**
```python
# Lower detection confidence
detector = LandmarkDetector(min_detection_confidence=0.3)

# Try different image
# Ensure face is clearly visible
# Check image quality and lighting
```

### Error: "MediaPipe not installed"

**Problem:** Missing MediaPipe package

**Solution:**
```bash
pip install mediapipe
```

### Error: "Image not found"

**Problem:** Invalid image path

**Solution:**
```python
# Check path exists
from pathlib import Path
image_path = Path("face.jpg")
if not image_path.exists():
    print("Image not found!")
```

### Poor Detection Quality

**Solutions:**
1. Use `refine_landmarks=True` (default)
2. Ensure good lighting in image
3. Use front-facing faces (not profile)
4. Check image resolution (at least 256x256 recommended)
5. Ensure face is not heavily occluded

### Multiple Faces - Wrong One Selected

**Problem:** Detector picks wrong face when multiple present

**Solutions:**
```python
# The detector automatically picks the largest face
# If you need a specific face, you can:
# 1. Crop image to single face first
# 2. Use face detection to identify correct face
# 3. Process each face separately
```

---

## Expected Output

When running `test_landmarks.py`, you should see:

```
============================================================
Testing landmark detection on: sample_face.jpg
============================================================

Running landmark detection...
✓ Landmark detection completed!
  Processing time: 0.15 seconds
  Image size: 512x512
  Faces detected: 1
  Total landmarks: 468

Feature Groups:
  Feature              Landmarks      BBox
  -----------------------------------------------------------------
  left_eye             16             (120,150)-(180,200)
  right_eye            16             (280,150)-(340,200)
  left_eyebrow         10             (110,120)-(190,140)
  right_eyebrow        10             (270,120)-(350,140)
  nose_tip             10             (240,200)-(280,240)
  nose_base            4              (250,240)-(270,250)
  mouth_outer          29             (200,280)-(320,320)
  mouth_inner          10              (220,300)-(300,310)
  left_ear             8              (80,180)-(120,240)
  right_ear            8              (360,180)-(400,240)
  face_contour         36             (50,100)-(450,400)
  jawline              36             (50,100)-(450,400)

Sample Landmark Coordinates (first 5):
  Index      X (px)          Y (px)          Z (px)
  -------------------------------------------------------
  0          234.50          127.30          -12.45
  1          235.20          128.10          -11.89
  2          236.10          129.20          -10.23
  3          237.50          130.50          -9.67
  4          239.20          131.80          -8.90

✓ Saved visualization: output/landmarks/landmarks_sample_face.png
✓ Saved coordinates (NumPy): output/landmarks/landmarks_coords_sample_face.npy
✓ Saved coordinates (CSV): output/landmarks/landmarks_coords_sample_face.csv
✓ Saved feature group coordinates: output/landmarks/feature_groups/

Landmark Statistics:
  X range: [50.00, 450.00]
  Y range: [100.00, 400.00]
  Z range: [-50.00, 25.00]
  Mean position: (250.00, 250.00)

============================================================
✓ All outputs saved successfully!
============================================================
```

---

## File Structure

After running tests, output directory structure:

```
output/landmarks/
├── landmarks_sample_face.png              # Visualization
├── landmarks_coords_sample_face.npy       # NumPy array
├── landmarks_coords_sample_face.csv       # CSV file
└── feature_groups/
    ├── left_eye_sample_face.npy
    ├── right_eye_sample_face.npy
    ├── left_eyebrow_sample_face.npy
    ├── right_eyebrow_sample_face.npy
    ├── nose_tip_sample_face.npy
    ├── nose_base_sample_face.npy
    ├── mouth_outer_sample_face.npy
    ├── mouth_inner_sample_face.npy
    ├── left_ear_sample_face.npy
    ├── right_ear_sample_face.npy
    └── face_contour_sample_face.npy
```

---

## Integration with Segmentation

You can combine landmark detection with face segmentation:

```python
from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector

segmenter = FaceSegmenter()
detector = LandmarkDetector()

image_path = "face.jpg"

# Get both results
seg_result = segmenter.segment(image_path)
landmark_result = detector.detect(image_path)

# Use landmarks to refine segmentation masks
# Or use segmentation to validate landmark positions
```

---

## Next Steps

After successful landmark detection:

1. ✅ **Verify outputs:** Check that landmarks are correctly positioned
2. ✅ **Test with different images:** Try various face angles and expressions
3. ✅ **Proceed to Step 5:** Create occlusion mask generator
4. ✅ **Integrate with pipeline:** Use landmarks in full reconstruction pipeline

---

## Support

If you encounter issues:

1. Check this documentation
2. Run verification: `python src/test_landmarks.py --verify`
3. Check MediaPipe documentation: https://google.github.io/mediapipe/
4. Review error messages carefully
5. Ensure all dependencies are installed

---

**Last Updated:** 2025
**Module Version:** 1.0.0
**Compatible with:** Python 3.10+, MediaPipe 0.10+

