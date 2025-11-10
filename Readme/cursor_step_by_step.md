# Cursor Implementation Guide: Forensic Facial Reconstruction System
## Step-by-Step Setup for Beginners (Windows → Linux SSH)

---

## STEP 1: Environment Setup & Dependencies
### Status: ⬜ NOT STARTED

### What to do:
1. Open Cursor IDE on your Windows machine
2. Use Cursor's "Remote - SSH" extension to connect to your Linux system
3. Create a new folder for this project

### Cursor Prompt #1 - Initial Setup:

```
I'm a beginner setting up a facial feature segmentation and forensic reconstruction 
system on Linux (Ubuntu 20.04 or 22.04). I'll be using Cursor IDE via SSH from Windows.

TASK: Help me set up the complete Python environment from scratch.

Requirements:
- Python 3.9 or higher
- CUDA 11.8+ support (if GPU available)
- Virtual environment
- All required dependencies for:
  * Face segmentation (BiSeNet)
  * Landmark detection (MediaPipe)
  * Image inpainting (Diffusers, Stable Diffusion)
  * Text encoding (OpenAI CLIP)
  * Face recognition (ArcFace)
  * Speech recognition (Google Cloud)
  * FastAPI for API server
  * Database search (FAISS)

Please provide:
1. Step-by-step bash commands to run in terminal
2. How to verify each step completed successfully
3. Common errors on Linux and how to fix them
4. Directory structure I should create

Assume I know how to:
- Use SSH to connect to Linux
- Run bash commands
- Use git clone
- Nothing else - explain everything

Start with the simplest steps first.
```

### After You Complete:
Copy the output from Cursor and save it as `setup_guide.sh`

Run it piece by piece:
```bash
# Connect via SSH from Windows PowerShell:
ssh username@your_linux_ip

# Go to your project folder
cd /path/to/project

# Run Cursor's generated setup commands one by one
```

**Verification Checklist:**
- [ ] Virtual environment created and activated
- [ ] All pip packages installed without errors
- [ ] Python version shows 3.9+
- [ ] CUDA detected (if GPU available)
- [ ] Can import torch, cv2, mediapipe, diffusers

**Move to STEP 2 only after verification complete**

---

## STEP 2: Download Pre-trained Models
### Status: ⬜ NOT STARTED

### Cursor Prompt #2 - Model Download:

```
I have completed environment setup. Now I need to download pre-trained models 
for facial segmentation and reconstruction.

TASK: Create Python script to download and verify all pre-trained models.

Models needed:
1. BiSeNet for face parsing (19 semantic segments)
2. MediaPipe Face Mesh (already included, just verify)
3. Stable Diffusion v1.5 inpainting weights
4. CLIP ViT-B/32 weights
5. ArcFace R50 for face embeddings
6. OpenCV pre-trained cascades

Please provide:
1. Python script (download_models.py) that:
   - Creates a 'models/' directory
   - Downloads each model with progress bar
   - Verifies SHA256 checksums
   - Shows disk space required before downloading
   - Tests that each model loads successfully

2. Instructions for:
   - How much disk space needed (rough estimate)
   - Where models are stored
   - How to verify downloads
   - What to do if download fails
   - Alternative URLs if primary fails

3. Error handling for:
   - Network timeout
   - Corrupted downloads
   - Out of disk space
   - Missing dependencies

Assume I'm using Linux and have ~100GB free disk space.
```

### After You Complete:

Run in your Linux terminal:
```bash
cd /path/to/project
python download_models.py
```

**Verification Checklist:**
- [ ] All models downloaded successfully
- [ ] Checksums verified
- [ ] Total disk space used shown
- [ ] Can import each model without errors
- [ ] No "ModuleNotFoundError" or "FileNotFoundError"

**Move to STEP 3 only after all models load**

---

## STEP 3: Build Facial Segmentation Module
### Status: ⬜ NOT STARTED

### Cursor Prompt #3 - Segmentation:

```
I have downloaded all pre-trained models. Now I need to build the facial 
segmentation module that splits face images into 19 components.

TASK: Create a complete FaceParser class for semantic segmentation.

Requirements:
1. Create 'src/face_parser.py' with FaceParser class that:
   - Loads BiSeNet model
   - Takes any JPG/PNG image path as input
   - Outputs 19 semantic masks (skin, mustache, eyes, etc.)
   - Creates colored segmentation visualization
   - Shows confidence score for each segment
   - Handles errors gracefully (image not found, no face, etc.)

2. Create 'src/test_segmentation.py' to:
   - Load a sample face image
   - Run segmentation
   - Display results
   - Save output images
   - Print statistics

3. Provide:
   - Clear docstrings for every function
   - Type hints for all inputs/outputs
   - Example usage
   - Troubleshooting tips
   - Performance metrics (speed, memory usage)

The 19 segments are:
background, skin, left_eyebrow, right_eyebrow, left_eye, right_eye, 
nose, mouth, upper_lip, lower_lip, hair, hat, earring, necklace, 
neck, cloth, face, left_ear, right_ear

Test with a simple face image. Make code modular and well-commented 
for a beginner.
```

### After You Complete:

Test in your Linux terminal:
```bash
cd /path/to/project
python src/test_segmentation.py --image sample_face.jpg
```

**Verification Checklist:**
- [ ] Segmentation runs without errors
- [ ] 19 masks generated
- [ ] Output images saved with colors
- [ ] Process takes <1 second
- [ ] Can manually inspect segmented images

**Important Files Created:**
- `src/face_parser.py` - Main segmentation class
- `src/test_segmentation.py` - Test script
- `output/segmentation_colored.jpg` - Colored output
- `output/masks/` - Individual mask files

**Move to STEP 4 only after segmentation works**

---

## STEP 4: Add Facial Landmark Detection
### Status: ⬜ NOT STARTED

### Cursor Prompt #4 - Landmarks:

```
I have working facial segmentation. Now I need to add facial landmark detection
to identify precise facial features (eyes, nose, mouth position, etc.).

TASK: Create LandmarkDetector class using MediaPipe.

Requirements:
1. Create 'src/landmark_detector.py' with:
   - LandmarkDetector class
   - Detects 468 3D facial landmarks (x, y, z coordinates)
   - Identifies key facial features:
     * Eyes (left & right)
     * Eyebrows (left & right)
     * Nose (tip and base)
     * Mouth (corners and center)
     * Ears (both)
     * Jaw outline
     * Face contour
   - Handles multiple face detection (pick largest face)
   - Returns landmarks in pixel coordinates
   - Provides bounding boxes for each feature

2. Create visualization function that:
   - Draws all 468 points on image
   - Draws connected mesh lines
   - Highlights specific features (different colors)
   - Adds text labels for major features

3. Create 'src/test_landmarks.py' to:
   - Load image
   - Detect landmarks
   - Show landmarks overlaid on image
   - Print coordinates in console
   - Save annotated image

4. Performance requirements:
   - Real-time performance (>30 FPS on CPU)
   - Handles different face angles/rotations
   - Doesn't fail on partial faces
   - Clear error messages if no face found

Include:
- Docstrings and type hints
- Example usage
- Error handling
- Coordinate system explanation
```

### After You Complete:

Test in terminal:
```bash
python src/test_landmarks.py --image sample_face.jpg --output output/landmarks.jpg
```

**Verification Checklist:**
- [ ] Detects 468 landmarks
- [ ] Landmarks visible on output image
- [ ] Coordinates printed in console
- [ ] Runs in <500ms per image
- [ ] Handles different face angles

**Move to STEP 5 only after landmarks detected**

---

## STEP 5: Create Occlusion Mask Generator
### Status: ⬜ NOT STARTED

### Cursor Prompt #5 - Occlusion Masks:

```
I have facial landmarks and segmentation. Now I need to create masks 
to simulate facial feature occlusion (hiding features to test reconstruction).

TASK: Create MaskGenerator class to create occlusion masks.

Requirements:
1. Create 'src/mask_generator.py' with MaskGenerator class:
   - Takes segmentation masks + landmarks as input
   - Can occlude specific features:
     * Mustache/Beard (lower face)
     * Eyebrows (left/right)
     * Eyes (left/right or both)
     * Eyes with glasses effect
     * Scar/injury (custom polygon)
     * Hair (top portion)
     * Entire upper/lower face
   - Generates binary mask (0=keep original, 255=inpaint this region)
   - Adds feathering to mask edges (smooth transition)
   - Handles overlapping regions properly

2. Create 'src/test_masking.py' to:
   - Load image
   - Generate different occlusion masks
   - Show before/after
   - Save mask visualizations
   - Explain what each mask does

3. Provide:
   - Adjustable margin/padding around features
   - Morphological operations (dilate/erode) for smoothing
   - Validation that mask is reasonable
   - Examples of different occlusion types

Use segmentation info to identify regions, not hardcoded coordinates.
Make it work for different face sizes and angles.
```

### After You Complete:

Test in terminal:
```bash
python src/test_masking.py --image sample_face.jpg --feature mustache
```

**Verification Checklist:**
- [ ] Different occlusion masks created
- [ ] Masks visualized correctly
- [ ] Mask edges smooth (not sharp)
- [ ] Multiple occlusion types work
- [ ] Output images saved

**Move to STEP 6 only after masking works**

---

## STEP 6: Implement Face Inpainting
### Status: ⬜ NOT STARTED

### Cursor Prompt #6 - Inpainting Module:

```
I have occlusion masks. Now I need to implement the core inpainting module
that reconstructs occluded facial features using Stable Diffusion.

TASK: Create FaceInpainter class for realistic feature reconstruction.

Requirements:
1. Create 'src/face_inpainter.py' with FaceInpainter class:
   - Uses Stable Diffusion v1.5 inpainting model
   - Takes original image + occlusion mask as input
   - Reconstructs occluded region realistically
   - Supports text prompts for guided generation:
     * Example: "realistic mustache, thick, black, detailed"
   - Adjustable inference steps (20-50 steps)
   - Adjustable guidance scale (7.5-15)
   - Supports GPU acceleration (CUDA)
   - Falls back to CPU gracefully if no GPU

2. Create 'src/test_inpainting.py' to:
   - Load image and occlusion mask
   - Run inpainting with default settings
   - Run inpainting with custom text prompts
   - Show before/after/reconstructed
   - Measure processing time
   - Save results

3. Features:
   - Memory optimization for large images (use fp16 precision)
   - Progress bar showing diffusion steps
   - Seed control for reproducibility
   - Batch processing support (multiple features at once)
   - Error handling for OOM (out of memory)

4. Testing requirements:
   - Works on CPU (slow but works)
   - Works on GPU (fast)
   - Handles 512x512 and 1024x1024 images
   - Clear feedback on what's happening

Provide:
- Explanation of each parameter
- Why certain prompts work better
- How to adjust quality vs speed tradeoff
- Typical processing times
- Memory requirements
```

### After You Complete:

Test in terminal (WARNING: First run downloads large model ~7GB):
```bash
# CPU version (slow, for testing):
python src/test_inpainting.py --image sample_face.jpg --mask output/masks/mustache.jpg --device cpu --steps 20

# GPU version (fast):
python src/test_inpainting.py --image sample_face.jpg --mask output/masks/mustache.jpg --device cuda --steps 50
```

**Verification Checklist:**
- [ ] Inpainting runs (even if slow on CPU)
- [ ] Reconstructed feature looks realistic
- [ ] Different prompts produce different results
- [ ] Processing time reasonable (<30 seconds on GPU)
- [ ] Output images saved
- [ ] No CUDA out of memory errors (if using GPU)

**Troubleshooting:**
If you get "CUDA out of memory": Use `--device cpu` or reduce image size

**Move to STEP 7 only after inpainting produces results**

---

## STEP 7: Add Text-to-Face Generation
### Status: ⬜ NOT STARTED

### Cursor Prompt #7 - Text to Face:

```
I have working inpainting. Now I need to add text-to-face generation 
so I can generate faces from witness descriptions.

TASK: Create TextToFaceGenerator class using CLIP-guided generation.

Requirements:
1. Create 'src/text_to_face.py' with TextToFaceGenerator class:
   - Uses CLIP text encoder to understand descriptions
   - Uses Stable Diffusion pipeline for generation
   - Takes text description as input:
     * Example: "Adult male, 45 years old, thick mustache, large ears, 
       dark complexion, angry expression"
   - Generates photorealistic face matching description
   - Supports negative prompts:
     * Example: "blurry, low quality, cartoon, sketch"
   - Adjustable inference steps and guidance scale
   - GPU/CPU support with fallback

2. Create 'src/test_text_to_face.py' to:
   - Load example descriptions
   - Generate faces for each description
   - Show side-by-side comparisons
   - Save all generated images
   - Print generation parameters used
   - Measure time taken

3. Provide:
   - Tips for better descriptions (what keywords work)
   - Bad description examples (what to avoid)
   - How to structure forensic descriptions
   - Different inference settings explained

4. Create 'data/sample_descriptions.txt' with examples:
   - Demographic descriptions
   - Ethnicity-specific features
   - Scar/distinctive mark descriptions
   - Expression descriptions
   - Indian regional features

Test with 5-10 different descriptions to verify quality.
```

### After You Complete:

Test in terminal:
```bash
python src/test_text_to_face.py --description "Adult male, 40-45 years old, thick black mustache, large ears"
```

**Verification Checklist:**
- [ ] Generates faces from text descriptions
- [ ] Faces look photorealistic
- [ ] Different descriptions produce different faces
- [ ] Processing time <20 seconds on GPU
- [ ] Generated images saved with description as filename
- [ ] Can run multiple generations in sequence

**Move to STEP 8 only after text-to-face works**

---

## STEP 8: Implement Multilingual Voice Input
### Status: ⬜ NOT STARTED

### Cursor Prompt #8 - Voice Recognition:

```
I have text-to-face generation. Now I need to add multilingual voice 
input support so witnesses can describe suspects in Hindi/Punjabi/etc.

TASK: Create VoiceDescriptionProcessor class for speech-to-text.

Requirements:
1. Create 'src/voice_processor.py' with:
   - VoiceDescriptionProcessor class
   - Support multiple languages:
     * Hindi (hi-IN)
     * Punjabi (pa-IN)
     * Bengali (bn-IN)
     * Tamil (ta-IN)
     * English (en-IN)
   - Two options for speech recognition:
     Option A: OpenAI Whisper (local, free, no API key)
     Option B: Google Cloud Speech-to-Text (more accurate, needs API key)
   - Load audio from .wav or .mp3 files
   - Transcribe to text
   - Return confidence score
   - Handle background noise gracefully

2. Create 'src/test_voice.py' to:
   - Load sample audio file (provide small test file)
   - Transcribe from Hindi/Punjabi/English
   - Show transcript
   - Show confidence score
   - Save results to text file

3. For beginners:
   - Use Whisper (easier, no API setup)
   - Provide sample audio files to test with
   - Step-by-step Whisper installation
   - Fallback if dependencies missing

4. Integration with text-to-face:
   - Pipeline: Voice → Transcription → Text → Face Generation

Include:
- Installation instructions for Whisper
- How to test without audio files (text input)
- Common issues and fixes
- Language detection (auto-detect input language)
```

### After You Complete:

Test in terminal:
```bash
# Download Whisper first:
pip install openai-whisper

# Test voice transcription:
python src/test_voice.py --audio sample_hindi_audio.wav --language hi-IN

# Or test with text directly (no audio needed):
python src/test_voice.py --text "बड़े कान, मोटा मूंछ" --language hi-IN
```

**Verification Checklist:**
- [ ] Whisper installed successfully
- [ ] Can transcribe English audio
- [ ] Can transcribe Hindi audio (if you have sample)
- [ ] Returns confidence scores
- [ ] Handles different audio qualities
- [ ] No crashes on bad audio

**Move to STEP 9 only after voice processing works**

---

## STEP 9: Create Forensic Description Parser
### Status: ⬜ NOT STARTED

### Cursor Prompt #9 - Description Parser:

```
I have voice transcription. Now I need to parse witness descriptions 
and extract forensic attributes for better face generation.

TASK: Create ForensicDescriptionParser to structure witness input.

Requirements:
1. Create 'src/description_parser.py' with ForensicDescriptionParser class:
   - Parses both English and Hindi descriptions
   - Extracts attributes:
     * Age/Age range
     * Gender
     * Ethnicity/Complexion
     * Hair (color, length, style)
     * Facial hair (mustache, beard, density)
     * Eyes (color, size, expression)
     * Nose (shape, size)
     * Scars/Marks (location, type)
     * Distinctive features
     * Expression (angry, neutral, etc.)
     * Build (thin, average, heavy)
   - Handles Hindi words mapping to English:
     * "मोटा" (mota) → "thick"
     * "बड़े" (bade) → "large"
     * "काले" (kale) → "black"
     * etc.
   - Returns structured dictionary of attributes
   - Handles vague descriptions gracefully
   - Shows confidence for each extracted attribute

2. Create 'src/test_description_parser.py' to:
   - Parse multiple sample descriptions (English and Hindi)
   - Show extracted attributes
   - Show confidence scores
   - Reconstruct description in structured format

3. Provide:
   - Hindi-English forensic word mapping (dictionary)
   - Pattern matching for age ranges
   - Ethnicity classification logic
   - Examples of good vs bad descriptions

4. Create 'data/forensic_lexicon.json' with:
   - Hindi words → English equivalents
   - Attribute categories
   - Common synonyms
   - Indian-specific features

Include:
- Regular expressions for pattern matching
- How to handle ambiguous words
- Fallback behavior for unknown words
- Examples and test cases
```

### After You Complete:

Test in terminal:
```bash
python src/test_description_parser.py \
  --description "Adult male, 40-45 years, thick mustache, large ears"

# Test with Hindi (if you have Hindi test data):
python src/test_description_parser.py \
  --description "बड़े कान, मोटा मूंछ, 40-45 साल" \
  --language hindi
```

**Verification Checklist:**
- [ ] Parses English descriptions correctly
- [ ] Extracts age, gender, hair, etc.
- [ ] Shows confidence for each attribute
- [ ] Handles Hindi words (at least a few)
- [ ] Handles ambiguous descriptions
- [ ] Returns structured output

**Important Files Created:**
- `src/description_parser.py` - Main parser
- `data/forensic_lexicon.json` - Hindi-English mapping
- `src/test_description_parser.py` - Test script

**Move to STEP 10 only after parser works**

---

## STEP 10: Create Face Recognition & Database Search
### Status: ⬜ NOT STARTED

### Cursor Prompt #10 - Database Matching:

```
I have description parsing. Now I need to create face recognition and 
database search to match reconstructed faces against a mugshot database.

TASK: Create ForensicDatabaseMatcher for suspect identification.

Requirements:
1. Create 'src/database_matcher.py' with ForensicDatabaseMatcher class:
   - Loads ArcFace model for face embeddings
   - Creates 512-dimensional face embeddings
   - Indexes embeddings using FAISS for fast search
   - Searches database of faces (start with 100-1000 faces for testing)
   - Returns top-K matching suspects with scores
   - Supports metadata (suspect_id, name, mugshot_date, etc.)

2. Create 'src/test_database_search.py' to:
   - Generate sample mugshot database (create fake metadata)
   - Create FAISS index
   - Generate test face embeddings
   - Search and show results
   - Display top-10 matches with scores

3. Create 'data/sample_mugshots.json' with:
   - 10-50 sample suspects
   - Fields: suspect_id, name, age, features, mugshot_path
   - Can be extended later with real data

4. Features:
   - Load/save FAISS index to disk
   - Soft biometric filtering (age range, gender)
   - Confidence scoring (0-1)
   - Metadata display with results
   - Handle empty database gracefully

5. For beginners:
   - Use pre-generated face embeddings (no need to process images first)
   - FAISS installation and setup explained
   - How to verify index creation
   - Troubleshooting guide

Include:
- Why ArcFace works for face matching
- How FAISS speeds up search
- Similarity score interpretation
- Handling false positives
```

### After You Complete:

Test in terminal:
```bash
python src/test_database_search.py \
  --query_image sample_face.jpg \
  --database_size 100 \
  --top_k 10
```

**Verification Checklist:**
- [ ] FAISS index created successfully
- [ ] Can search 100+ faces quickly (<1 second)
- [ ] Returns top-10 matches with scores
- [ ] Metadata displayed with results
- [ ] No errors on first run

**Important Files Created:**
- `src/database_matcher.py` - Main matcher class
- `data/sample_mugshots.json` - Sample database
- `output/faiss_index.bin` - Indexed database
- `src/test_database_search.py` - Test script

**Move to STEP 11 only after database search works**

---

## STEP 11: Build Complete End-to-End Pipeline
### Status: ⬜ NOT STARTED

### Cursor Prompt #11 - Complete Pipeline:

```
I have all individual components. Now I need to build an end-to-end 
pipeline that integrates everything together.

TASK: Create main.py that chains all components.

Requirements:
1. Create 'src/pipeline.py' with ForensicReconstructionPipeline class:
   - Takes witness input (text description or voice)
   - Step 1: Parse description → structured attributes
   - Step 2: Generate face from description
   - Step 3: Extract face embeddings
   - Step 4: Search database for matches
   - Step 5: Display top matches with confidence
   - Shows timing for each step
   - Logs all operations
   - Handles errors gracefully

2. Create 'main.py' entry point:
   - Command-line interface
   - Arguments: --input (text/voice file), --database (path to index), --top_k (10)
   - Shows progress bar
   - Pretty-prints results
   - Example commands:
     * python main.py --text "Adult male, 40-45, thick mustache"
     * python main.py --voice description.wav --language hi-IN
     * python main.py --image suspect_photo.jpg

3. Create 'src/utils.py' with:
   - Logging setup
   - Config management
   - Progress bar helpers
   - Result formatting
   - Error handling
   - Time measurement utilities

4. Create 'config.yaml' with:
   - Model paths
   - FAISS database path
   - Default parameters
   - Language settings
   - Output directories

5. Provide:
   - Step-by-step execution flow
   - Timing breakdown
   - Success/failure indicators
   - Next steps after identification

Include:
- Clear console output
- Save results to output directory
- Timestamped reports
- Reproducibility (save all inputs/outputs)
```

### After You Complete:

Test the complete pipeline:
```bash
# Test 1: From text description
python main.py --text "Adult male, 40-45, thick mustache" --top_k 5

# Test 2: From voice file (if available)
python main.py --voice description.wav --language en-IN --top_k 10

# Verify output saved:
ls -la output/
```

**Verification Checklist:**
- [ ] Complete pipeline runs without errors
- [ ] Each step shows progress
- [ ] Results displayed clearly
- [ ] Timing breakdown shown
- [ ] Output files saved with timestamp
- [ ] Can run multiple times without conflicts

**Expected Output:**
```
=== FORENSIC FACE RECONSTRUCTION PIPELINE ===

[Step 1] Parsing description... ✓ (0.2s)
  - Age: 40-45
  - Hair: black, thick
  - Mustache: thick, black
  - Expression: neutral

[Step 2] Generating face... ✓ (15.3s)
  - Created realistic face image
  - Saved: output/generated_face.jpg

[Step 3] Searching database... ✓ (0.8s)
  - Matched against 1000 suspects
  
[Results] Top 5 Suspects:
  1. Suspect #234 - Match: 92% ✓
  2. Suspect #567 - Match: 87%
  3. Suspect #123 - Match: 81%
  ...

Total time: 16.3 seconds
```

**Move to STEP 12 only after complete pipeline works**

---

## STEP 12: Create Web Interface (Optional but Recommended)
### Status: ⬜ NOT STARTED

### Cursor Prompt #12 - Web Interface:

```
I have working end-to-end pipeline. Now I'll create a simple web interface
so police officers can easily use the system without command line.

TASK: Create simple FastAPI web interface.

Requirements:
1. Create 'app.py' with FastAPI server:
   - Route: POST /generate_face
     Input: text description
     Output: generated face image
   - Route: POST /transcribe_voice
     Input: audio file + language
     Output: transcript + generated face
   - Route: POST /search_database
     Input: face image
     Output: top-10 matching suspects
   - Route: GET /status
     Output: system status, models loaded, etc.

2. Create 'templates/index.html' with:
   - Text input for description
   - File upload for voice/image
   - Language selector (Hindi/Punjabi/English/etc.)
   - Display generated face
   - Display top matches
   - Refresh/reset buttons

3. Features:
   - Real-time progress updates
   - Display generated images inline
   - Responsive design (works on mobile)
   - Error messages
   - Request/response logging

4. Installation & running:
   - pip install fastapi uvicorn python-multipart
   - uvicorn app:app --host 0.0.0.0 --port 8000
   - Access at: http://your-linux-ip:8000

Include:
- Basic HTML/CSS (nothing fancy, just functional)
- How to deploy on Linux server
- How to access from Windows browser
- Security considerations (authentication)
```

### After You Complete:

Start the web server:
```bash
# Install dependencies:
pip install fastapi uvicorn python-multipart aiofiles

# Start server:
uvicorn app:app --host 0.0.0.0 --port 8000

# From Windows browser, access:
http://your-linux-ip:8000
```

**Verification Checklist:**
- [ ] FastAPI server starts without errors
- [ ] Can access web interface from Windows
- [ ] Text input generates face in browser
- [ ] Voice upload works (if you have audio)
- [ ] Results display nicely
- [ ] No 500 errors

**Move to STEP 13 only after web interface works**

---

## STEP 13: Testing & Optimization
### Status: ⬜ NOT STARTED

### Cursor Prompt #13 - Testing:

```
I have complete working system. Now I need comprehensive testing 
and performance optimization.

TASK: Create comprehensive testing suite.

Requirements:
1. Create 'tests/' directory with:
   - test_segmentation.py - Unit tests for face parsing
   - test_landmarks.py - Unit tests for landmark detection
   - test_inpainting.py - Integration tests for inpainting
   - test_text_to_face.py - Integration tests for generation
   - test_description_parser.py - Tests for attribute extraction
   - test_database_search.py - Tests for matching
   - test_end_to_end.py - Full pipeline tests

2. Create 'benchmark.py' to measure:
   - Processing time for each component
   - Memory usage
   - GPU utilization
   - Database search speed vs. database size
   - Throughput (faces per second)
   - Bottlenecks identification

3. Create 'data/test_cases.json' with:
   - 10-20 test descriptions (English and Hindi)
   - Expected characteristics
   - Manual verification results
   - Edge cases (unusual descriptions, etc.)

4. Performance optimization:
   - Model quantization (faster inference)
   - Batch processing (multiple faces at once)
   - Caching (avoid reloading models)
   - Parallel processing (use multiple GPUs if available)

5. Provide:
   - How to run tests: pytest tests/
   - How to run benchmarks: python benchmark.py
   - Performance targets
   - Optimization recommendations
   - Scaling guidelines
```

### After You Complete:

Run tests and benchmarks:
```bash
# Install pytest:
pip install pytest

# Run all tests:
pytest tests/ -v

# Run benchmarks:
python benchmark.py

# Run with profiling:
python -m cProfile -s cumulative benchmark.py
```

**Verification Checklist:**
- [ ] All unit tests pass
- [ ] End-to-end tests complete successfully
- [ ] Benchmarks show acceptable performance
- [ ] No memory leaks
- [ ] Can identify bottlenecks
- [ ] Performance report generated

**Move to STEP 14 only after testing complete**

---

## STEP 14: Documentation & Deployment
### Status: ⬜ NOT STARTED

### Cursor Prompt #14 - Documentation:

```
I have tested and optimized system. Now I need proper documentation 
and deployment setup.

TASK: Create comprehensive documentation and deployment guide.

Requirements:
1. Create detailed documentation:
   - README.md: Project overview, quick start
   - INSTALL.md: Step-by-step installation guide
   - USAGE.md: How to use all features
   - API.md: API documentation (for developers)
   - TROUBLESHOOTING.md: Common issues and fixes
   - ARCHITECTURE.md: System design and flow

2. Create 'requirements.txt' with:
   - All Python dependencies
   - Specific versions
   - Installation instructions
   - Optional dependencies noted

3. Create Docker support (optional but recommended):
   - Dockerfile: Container for deployment
   - docker-compose.yml: Multi-container setup
   - .dockerignore: Files to exclude
   - Instructions for building and running

4. Create deployment guide:
   - How to deploy on cloud (AWS/GCP/Azure)
   - How to deploy on Linux server
   - Environment variables to set
   - Security configuration
   - Monitoring and logging setup

5. Create quick reference guide for users:
   - Common workflows
   - Example commands
   - Keyboard shortcuts (if applicable)
   - FAQs

Include:
- Professional formatting
- Clear step-by-step instructions
- Screenshots (where helpful)
- Troubleshooting sections
- Contact/support information
```

### After You Complete:

Generate documentation:
```bash
# Create requirements.txt:
pip freeze > requirements.txt

# Verify documentation files:
ls -la docs/
ls -la *.md

# Test README instructions on fresh system
```

**Verification Checklist:**
- [ ] All .md documentation files created
- [ ] requirements.txt accurate and tested
- [ ] Installation instructions work
- [ ] API documentation complete
- [ ] Deployment guide tested
- [ ] README gives good project overview

---

## FINAL VERIFICATION CHECKLIST

Before you're done, verify:

- [ ] **Step 1**: Environment setup - all dependencies installed
- [ ] **Step 2**: Models downloaded - all models loading
- [ ] **Step 3**: Segmentation working - 19 masks generated
- [ ] **Step 4**: Landmarks detected - 468 points on face
- [ ] **Step 5**: Occlusion masks created - visualized correctly
- [ ] **Step 6**: Inpainting working - realistic reconstruction
- [ ] **Step 7**: Text-to-face generating - realistic faces from text
- [ ] **Step 8**: Voice recognition working - transcription accurate
- [ ] **Step 9**: Description parser - attributes extracted
- [ ] **Step 10**: Database search - top matches found
- [ ] **Step 11**: End-to-end pipeline - complete flow working
- [ ] **Step 12**: Web interface - accessible from browser
- [ ] **Step 13**: Tests passing - all tests successful
- [ ] **Step 14**: Documentation complete - clear and thorough

---

## Project Structure (Final)

```
forensic_reconstruction/
├── main.py                 # Entry point
├── app.py                  # FastAPI web server
├── config.yaml             # Configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # Project overview
├── INSTALL.md              # Installation guide
├── USAGE.md                # Usage guide
├── API.md                  # API documentation
├── ARCHITECTURE.md         # System design
├── TROUBLESHOOTING.md      # Common issues
│
├── src/
│   ├── __init__.py
│   ├── face_parser.py           # Step 3
│   ├── landmark_detector.py      # Step 4
│   ├── mask_generator.py         # Step 5
│   ├── face_inpainter.py         # Step 6
│   ├── text_to_face.py           # Step 7
│   ├── voice_processor.py        # Step 8
│   ├── description_parser.py     # Step 9
│   ├── database_matcher.py       # Step 10
│   ├── pipeline.py               # Step 11
│   └── utils.py                  # Helper functions
│
├── tests/
│   ├── test_segmentation.py
│   ├── test_landmarks.py
│   ├── test_inpainting.py
│   ├── test_text_to_face.py
│   ├── test_description_parser.py
│   ├── test_database_search.py
│   └── test_end_to_end.py
│
├── templates/
│   └── index.html           # Web interface
│
├── data/
│   ├── forensic_lexicon.json       # Hindi-English mapping
│   ├── sample_descriptions.txt     # Example descriptions
│   ├── sample_mugshots.json        # Sample database
│   └── test_cases.json             # Test cases
│
├── models/                   # Downloaded models (large, ~100GB)
│   ├── bisenet/
│   ├── stable_diffusion/
│   ├── clip/
│   ├── arcface/
│   └── ...
│
├── output/
│   ├── segmented/           # Segmentation results
│   ├── landmarks/           # Landmark detections
│   ├── masks/               # Occlusion masks
│   ├── inpainted/           # Inpainting results
│   ├── generated_faces/     # Generated faces
│   ├── database_matches/    # Matching results
│   └── reports/             # Analysis reports
│
└── logs/
    ├── system.log           # General logs
    ├── errors.log           # Error logs
    └── performance.log      # Performance metrics
```

---

## Key Points for Beginners

1. **Do one step at a time**: Don't move to next step until previous is 100% working
2. **Use Cursor's AI**: Every prompt is meant to be given to Cursor - don't do it manually
3. **Test each component**: Run test files after each step
4. **Save outputs**: Keep generated images/masks for troubleshooting
5. **Ask Cursor for help**: If something fails, ask Cursor "Why did this fail?" with error message
6. **Linux SSH tips**:
   - Use `screen` or `tmux` for long-running processes
   - Use `ssh -X` if you need GUI (forward X11)
   - Check disk space with `df -h`
   - Monitor GPU with `nvidia-smi` (if NVIDIA GPU)
7. **Windows tips**:
   - Use PowerShell or Windows Terminal for SSH
   - File paths: Use `/` not `\` when in SSH
   - Transfer files with `scp` or `sftp`

---

## Next Actions

1. Open Cursor IDE
2. Connect to Linux server via Remote SSH
3. Create project folder: `mkdir forensic_reconstruction && cd forensic_reconstruction`
4. Start with **STEP 1** prompt above
5. Copy-paste the prompt into Cursor
6. Follow Cursor's instructions exactly
7. Only move to next step when current is complete
8. Let me know when you finish each step!