# Forensic Reconstruction Setup Guide

Complete step-by-step guide for setting up the facial feature segmentation and forensic reconstruction system on Ubuntu Linux.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
4. [Directory Structure](#directory-structure)
5. [Verification](#verification)
6. [Common Errors and Fixes](#common-errors-and-fixes)
7. [Manual Installation (Alternative)](#manual-installation-alternative)

---

## Prerequisites

Before starting, ensure you have:
- Ubuntu 20.04 or 22.04
- SSH access to your Linux machine
- At least 20GB free disk space
- Internet connection
- (Optional) NVIDIA GPU with CUDA support

---

## Quick Start

If you want to set up everything automatically:

```bash
# 1. Navigate to project directory
cd /home/teaching/G14/forensic_reconstruction

# 2. Make setup script executable
chmod +x setup.sh

# 3. Run setup script
./setup.sh

# 4. Verify installation
chmod +x verify_installation.sh
./verify_installation.sh
```

**That's it!** The script will handle everything automatically.

---

## Detailed Step-by-Step Instructions

Follow these steps if you prefer to understand each step or if the automated script fails.

### Step 1: Check Python Version

**Command:**
```bash
python3 --version
```

**Expected Output:**
```
Python 3.9.x or higher
```

**If Python 3.9+ is not installed:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
```

**Verification:**
- Run `python3 --version` again
- Should show Python 3.9 or higher

---

### Step 2: Check pip (Python Package Manager)

**Command:**
```bash
pip3 --version
```

**Expected Output:**
```
pip 23.x.x or higher
```

**If pip is not installed:**
```bash
sudo apt install python3-pip
```

**Upgrade pip:**
```bash
python3 -m pip install --upgrade pip
```

**Verification:**
- Run `pip3 --version`
- Should show a recent version

---

### Step 3: Check for GPU/CUDA (Optional but Recommended)

**Command:**
```bash
nvidia-smi
```

**Expected Output (if GPU available):**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 11.x   |
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
+-----------------------------------------------------------------------------+
```

**If no GPU:**
- That's okay! The system will work with CPU (slower but functional)
- Skip GPU-related steps

**If GPU detected:**
- Note your CUDA version (e.g., 11.8, 12.1)
- We'll use CUDA 11.8 for PyTorch installation

---

### Step 4: Create Project Directory Structure

**Commands:**
```bash
# You're already in the project directory, but let's create subdirectories
cd /home/teaching/G14/forensic_reconstruction

# Create necessary directories
mkdir -p data/input
mkdir -p data/output
mkdir -p models
mkdir -p logs
mkdir -p scripts
```

**Verification:**
```bash
ls -la
# Should show: data, models, logs, scripts directories
```

---

### Step 5: Create Virtual Environment

**What is a virtual environment?**
- Isolated Python environment for this project
- Prevents conflicts with other Python projects
- Best practice for Python development

**Command:**
```bash
python3 -m venv venv
```

**What this does:**
- Creates a `venv` directory with isolated Python environment
- Takes 1-2 minutes

**Verification:**
```bash
ls -la venv/
# Should show: bin, lib, include, pyvenv.cfg
```

---

### Step 6: Activate Virtual Environment

**Command:**
```bash
source venv/bin/activate
```

**What this does:**
- Activates the virtual environment
- Your terminal prompt should now show `(venv)` at the beginning

**Expected Prompt:**
```
(venv) user@machine:~/forensic_reconstruction$
```

**Verification:**
```bash
which python
# Should show: /home/teaching/G14/forensic_reconstruction/venv/bin/python
```

**Important:** You must activate the virtual environment every time you open a new terminal session!

---

### Step 7: Install PyTorch

PyTorch is the deep learning framework. Installation depends on whether you have a GPU.

#### Option A: With GPU (CUDA 11.8)

**Command:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Takes:** 5-10 minutes (large download ~2GB)

#### Option B: CPU Only

**Command:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Takes:** 3-5 minutes (smaller download ~500MB)

**Verification:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**Expected Output:**
```
PyTorch version: 2.0.0 or higher
```

**Test CUDA (if GPU available):**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
CUDA available: True
```

---

### Step 8: Install FAISS (Vector Database)

FAISS is used for fast similarity search.

#### Option A: With GPU

**Command:**
```bash
pip install faiss-gpu
```

#### Option B: CPU Only

**Command:**
```bash
pip install faiss-cpu
```

**Verification:**
```bash
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
```

---

### Step 9: Install Other Requirements

**Command:**
```bash
pip install -r requirements.txt
``
**What this does:**
- Installs all packages listed in `requirements.txt`
- Takes 10-20 minutes (many packages to download)

**Verification:**
```bash
pip list
# Should show all installed packages
```

---

### Step 10: Install spaCy Language Model

**Command:**
```bash
python -m spacy download en_core_web_sm
```

**What this does:**
- Downloads English language model for spaCy
- Takes 1-2 minutes (~50MB)

**Verification:**
```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded')"
```

---

### Step 11: Verify Complete Installation

Run the verification script:

```bash
chmod +x verify_installation.sh
./verify_installation.sh
```

Or test manually:

```bash
# Test each component
python -c "import torch; print('PyTorch OK')"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import diffusers; print('Diffusers OK')"
python -c "import clip; print('CLIP OK')"
python -c "import faiss; print('FAISS OK')"
python -c "import fastapi; print('FastAPI OK')"
```

---

## Directory Structure

After setup, your project should look like this:

```
forensic_reconstruction/
├── venv/                    # Virtual environment (created by setup)
│   ├── bin/                 # Python executables
│   ├── lib/                 # Installed packages
│   └── ...
├── data/                    # Data directories
│   ├── input/               # Input images/videos
│   └── output/              # Processed results
├── models/                  # Saved model files
├── logs/                    # Log files
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
├── setup.sh                 # Automated setup script
├── verify_installation.sh   # Verification script
└── README_SETUP.md          # This file
```

---

## Verification

### Quick Verification

Run the verification script:
```bash
source venv/bin/activate  # Activate venv first!
./verify_installation.sh
```

### Manual Verification

Test each component:

```bash
# Activate virtual environment
source venv/bin/activate

# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# Test MediaPipe (face landmarks)
python -c "import mediapipe as mp; mp_face = mp.solutions.face_mesh; print('MediaPipe OK')"

# Test OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test Diffusers (Stable Diffusion)
python -c "from diffusers import StableDiffusionPipeline; print('Diffusers OK')"

# Test CLIP
python -c "import clip; print('CLIP OK')"

# Test FAISS
python -c "import faiss; print('FAISS:', faiss.__version__)"

# Test FastAPI
python -c "from fastapi import FastAPI; print('FastAPI OK')"
```

---

## Common Errors and Fixes

### Error 1: "python3: command not found"

**Problem:** Python 3 is not installed.

**Fix:**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

---

### Error 2: "pip: command not found"

**Problem:** pip is not installed.

**Fix:**
```bash
sudo apt install python3-pip
python3 -m pip install --upgrade pip
```

---

### Error 3: "Permission denied" when installing packages

**Problem:** Trying to install globally instead of in virtual environment.

**Fix:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Your prompt should show (venv)
# Then install packages
pip install package_name
```

---

### Error 4: "CUDA out of memory" or "CUDA not available"

**Problem:** PyTorch can't access GPU.

**Fixes:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```
   If this fails, install NVIDIA driver:
   ```bash
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

2. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA in Python:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

### Error 5: "No module named 'X'"

**Problem:** Package not installed or virtual environment not activated.

**Fix:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install missing package
pip install X

# Or reinstall all requirements
pip install -r requirements.txt
```

---

### Error 6: "ERROR: Could not find a version that satisfies the requirement"

**Problem:** Package version conflict or incompatible Python version.

**Fixes:**

1. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

2. **Install without version pinning:**
   ```bash
   pip install package_name  # Instead of package_name==1.0.0
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

---

### Error 7: "spaCy model 'en_core_web_sm' not found"

**Problem:** spaCy model not downloaded.

**Fix:**
```bash
python -m spacy download en_core_web_sm
```

---

### Error 8: "Out of disk space"

**Problem:** Not enough space for packages.

**Fix:**
```bash
# Check disk space
df -h

# Clean pip cache
pip cache purge

# Remove old virtual environment and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

### Error 9: "SSL: CERTIFICATE_VERIFY_FAILED"

**Problem:** SSL certificate issues.

**Fix:**
```bash
# Update certificates
sudo apt update
sudo apt install ca-certificates

# Or use pip with trusted hosts (temporary workaround)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

---

### Error 10: Slow download speeds

**Problem:** Slow internet or PyPI mirror issues.

**Fix:**
```bash
# Use pip with timeout
pip install --default-timeout=100 package_name

# Or use alternative index
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

---

## Manual Installation (Alternative)

If the automated script doesn't work, follow these manual commands:

```bash
# 1. Navigate to project
cd /home/teaching/G14/forensic_reconstruction

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch (choose one)
# GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install FAISS (choose one)
# GPU:
pip install faiss-gpu
# CPU:
pip install faiss-cpu

# 7. Install other requirements
pip install -r requirements.txt

# 8. Install spaCy model
python -m spacy download en_core_web_sm

# 9. Verify
python -c "import torch; print('Setup complete!')"
```

---

## Next Steps

After successful setup:

1. **Always activate virtual environment before working:**
   ```bash
   source venv/bin/activate
   ```

2. **Test a simple import:**
   ```bash
   python -c "import torch, mediapipe, cv2; print('All core packages imported successfully!')"
   ```

3. **Start developing your application!**

---

## Getting Help

If you encounter issues:

1. Check the [Common Errors](#common-errors-and-fixes) section
2. Verify virtual environment is activated: `echo $VIRTUAL_ENV`
3. Check Python version: `python --version`
4. Check installed packages: `pip list`
5. Review error messages carefully - they often contain helpful hints

---

## Summary

**Essential commands to remember:**

```bash
# Activate virtual environment (do this first, every time!)
source venv/bin/activate

# Install a package
pip install package_name

# List installed packages
pip list

# Deactivate virtual environment
deactivate
```

**Remember:** Always activate the virtual environment (`source venv/bin/activate`) before running Python scripts or installing packages!

