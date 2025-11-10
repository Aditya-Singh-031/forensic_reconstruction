#!/bin/bash

# Verification script to test all installed components
# Run this after setup.sh to verify everything works

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

echo "=========================================="
echo "Verification Script"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Virtual environment not activated!"
    print_info "Run: source venv/bin/activate"
    exit 1
else
    print_success "Virtual environment is active: $VIRTUAL_ENV"
fi

echo ""
echo "Testing Python packages..."
echo ""

# Test PyTorch
echo -n "Testing PyTorch... "
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    print_success "PyTorch OK"
else
    print_error "PyTorch failed"
    exit 1
fi

# Test CUDA (if available)
if command -v nvidia-smi &> /dev/null; then
    echo -n "Testing CUDA in PyTorch... "
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        print_success "CUDA available"
        python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    else
        print_error "CUDA not available in PyTorch"
    fi
fi

# Test MediaPipe
echo -n "Testing MediaPipe... "
if python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')" 2>/dev/null; then
    print_success "MediaPipe OK"
else
    print_error "MediaPipe failed"
    exit 1
fi

# Test OpenCV
echo -n "Testing OpenCV... "
if python -c "import cv2; print(f'OpenCV {cv2.__version__}')" 2>/dev/null; then
    print_success "OpenCV OK"
else
    print_error "OpenCV failed"
    exit 1
fi

# Test Diffusers
echo -n "Testing Diffusers... "
if python -c "import diffusers; print(f'Diffusers {diffusers.__version__}')" 2>/dev/null; then
    print_success "Diffusers OK"
else
    print_error "Diffusers failed"
    exit 1
fi

# Test CLIP
echo -n "Testing CLIP... "
if python -c "import open_clip; print('✓ CLIP (open-clip-torch)')" 2>/dev/null; then
    print_success "CLIP OK"
else
    print_error "CLIP failed"
    exit 1
fi

# Test FAISS
echo -n "Testing FAISS... "
if python -c "import faiss; print(f'FAISS {faiss.__version__}')" 2>/dev/null; then 
    print_success "FAISS OK"
else
    print_error "FAISS failed"
    exit 1
fi

# Test FastAPI
echo -n "Testing FastAPI... "
if python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')" 2>/dev/null; then
    print_success "FastAPI OK"
else
    print_error "FastAPI failed"
    exit 1
fi

# Test Transformers
echo -n "Testing Transformers... "
if python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
    print_success "Transformers OK"
else
    print_error "Transformers failed"
    exit 1
fi

# Test spaCy
echo -n "Testing spaCy... "
if python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')" 2>/dev/null; then
    print_success "spaCy OK"
else
    print_error "spaCy failed (may need to run: python -m spacy download en_core_web_sm)"
    exit 1
fi

# Test Google Cloud Speech (just import, no API key needed)
echo -n "Testing Google Cloud Speech... "
if python -c "from google.cloud import speech; print('Google Cloud Speech OK')" 2>/dev/null; then
    print_success "Google Cloud Speech OK"
else
    print_error "Google Cloud Speech failed"
    exit 1
fi

echo ""
echo "=========================================="
print_success "All tests passed!"
echo "=========================================="

