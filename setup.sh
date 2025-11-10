#!/bin/bash

# Forensic Reconstruction Environment Setup Script
# This script sets up Python 3.9+, virtual environment, and all dependencies
# Run this script from the project root directory

set -e  # Exit on any error

echo "=========================================="
echo "Forensic Reconstruction Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
        print_info "Install Python 3.9+ with: sudo apt update && sudo apt install python3.9 python3.9-venv python3-pip"
        exit 1
    fi
else
    print_error "Python3 not found. Install with: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Step 2: Check pip
echo ""
echo "Step 2: Checking pip..."
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_info "Installing pip3..."
    sudo apt update
    sudo apt install -y python3-pip
    print_success "pip3 installed"
fi

# Step 3: Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
python3 -m pip install --upgrade pip
print_success "pip upgraded"

# Step 4: Check for GPU/CUDA
echo ""
echo "Step 4: Checking for GPU/CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    print_info "CUDA Version: $CUDA_VERSION"
    USE_GPU=true
else
    print_info "No NVIDIA GPU detected. Will use CPU-only versions."
    USE_GPU=false
fi

# Step 5: Create virtual environment
echo ""
echo "Step 5: Creating virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment 'venv' already exists. Skipping creation."
else
    python3 -m venv venv
    print_success "Virtual environment created in 'venv' directory"
fi

# Step 6: Activate virtual environment
echo ""
echo "Step 6: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 7: Install PyTorch (with or without CUDA)
echo ""
echo "Step 7: Installing PyTorch..."
if [ "$USE_GPU" = true ]; then
    print_info "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_success "PyTorch with CUDA installed"
else
    print_info "Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch (CPU) installed"
fi

# Step 8: Install FAISS (with or without GPU)
echo ""
echo "Step 8: Installing FAISS..."
if [ "$USE_GPU" = true ]; then
    print_info "Installing FAISS with GPU support..."
    pip install faiss-gpu
    print_success "FAISS-GPU installed"
else
    print_info "Installing FAISS (CPU-only)..."
    pip install faiss-cpu
    print_success "FAISS-CPU installed"
fi

# Step 9: Install other requirements
echo ""
echo "Step 9: Installing other requirements from requirements.txt..."
pip install -r requirements.txt
print_success "All requirements installed"

# Step 10: Install spaCy language model
echo ""
echo "Step 10: Installing spaCy English model..."
python -m spacy download en_core_web_sm
print_success "spaCy model installed"

# Step 11: Verify installation
echo ""
echo "Step 11: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
if [ "$USE_GPU" = true ]; then
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        print_success "CUDA is available in PyTorch"
    else
        print_error "CUDA not available in PyTorch (may need to reinstall)"
    fi
fi

echo ""
echo "=========================================="
print_success "Setup completed successfully!"
echo "=========================================="
echo ""
print_info "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
print_info "To deactivate, run:"
echo "  deactivate"
echo ""

