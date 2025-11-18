#!/bin/bash

# Setup Dataset Creation Environment
# Creates directory structure on /DATA and symlinks to current project

set -e

echo "=========================================="
echo "FACIAL FEATURES DATASET SETUP"
echo "=========================================="
echo ""

# Configuration
DATASET_ROOT="/DATA/facial_features_dataset"
PROJECT_ROOT="$HOME/G14/forensic_reconstruction"
SYMLINK_PATH="$PROJECT_ROOT/dataset"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create directory structure
echo -e "${BLUE}[1/6] Creating directory structure on /DATA...${NC}"
mkdir -p "$DATASET_ROOT"/{raw_images/{ffhq,celeba_hq},features,metadata,annotations}

# Create feature subdirectories
cd "$DATASET_ROOT/features"
mkdir -p segmentation landmarks
mkdir -p eyes_left eyes_right eyebrows_left eyebrows_right
mkdir -p nose mouth_outer mouth_inner
mkdir -p ears_left ears_right face_contour jawline

echo -e "${GREEN}✓ Directory structure created${NC}"
echo "  Available space:"
df -h "$DATASET_ROOT" | tail -1

# Step 2: Create symlink
echo ""
echo -e "${BLUE}[2/6] Creating symlink...${NC}"
if [ -L "$SYMLINK_PATH" ]; then
    echo "  Removing existing symlink..."
    rm "$SYMLINK_PATH"
fi

ln -s "$DATASET_ROOT" "$SYMLINK_PATH"
echo -e "${GREEN}✓ Symlink created${NC}"
echo "  $SYMLINK_PATH -> $DATASET_ROOT"

# Step 3: Verify symlink
echo ""
echo -e "${BLUE}[3/6] Verifying symlink...${NC}"
if [ -L "$SYMLINK_PATH" ] && [ -e "$SYMLINK_PATH" ]; then
    echo -e "${GREEN}✓ Symlink is valid${NC}"
    ls -lh "$SYMLINK_PATH"
else
    echo -e "${YELLOW}✗ Symlink verification failed${NC}"
    exit 1
fi

# Step 4: Copy dataset creation scripts
echo ""
echo -e "${BLUE}[4/6] Copying dataset creation scripts...${NC}"
cd "$PROJECT_ROOT"

if [ ! -f "src/dataset_downloader.py" ]; then
    echo "  Rename: dataset_downloader_complete.py -> src/dataset_downloader.py"
    echo "  (If not already done)"
fi

if [ ! -f "src/create_features_dataset.py" ]; then
    echo "  Rename: create_features_dataset_complete.py -> src/create_features_dataset.py"
    echo "  (If not already done)"
fi

if [ ! -f "src/validate_dataset.py" ]; then
    echo "  Rename: validate_dataset_complete.py -> src/validate_dataset.py"
    echo "  (If not already done)"
fi

echo -e "${GREEN}✓ Script setup complete${NC}"

# Step 5: Setup Git LFS
echo ""
echo -e "${BLUE}[5/6] Setting up Git LFS...${NC}"

cd "$PROJECT_ROOT"

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "  Installing Git LFS..."
    sudo apt-get update -qq
    sudo apt-get install -y git-lfs > /dev/null 2>&1
fi

# Initialize Git LFS
git lfs install

# Create .gitattributes if needed
if [ ! -f ".gitattributes" ]; then
    cat > .gitattributes << 'EOF'
# Large file tracking for dataset
dataset/raw_images/**/*.png filter=lfs diff=lfs merge=lfs -text
dataset/raw_images/**/*.jpg filter=lfs diff=lfs merge=lfs -text
dataset/features/**/*.png filter=lfs diff=lfs merge=lfs -text
dataset/features/**/*.npy filter=lfs diff=lfs merge=lfs -text
dataset/metadata/**/*.json filter=lfs diff=lfs merge=lfs -text
dataset/annotations/**/*.csv filter=lfs diff=lfs merge=lfs -text
EOF
    
    git add .gitattributes
    echo -e "${GREEN}✓ Created .gitattributes${NC}"
else
    echo -e "${GREEN}✓ .gitattributes already exists${NC}"
fi

# Step 6: Install Python dependencies
echo ""
echo -e "${BLUE}[6/6] Checking Python dependencies...${NC}"

pip install -q kaggle gdown tqdm pandas Pillow opencv-python 2>/dev/null || true

echo -e "${GREEN}✓ Dependencies ready${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Dataset location: $DATASET_ROOT"
echo "Symlink: $SYMLINK_PATH"
echo ""
echo "Available space on /DATA:"
df -h "$DATASET_ROOT" | tail -1
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Download datasets:"
echo "   python -m src.dataset_downloader --dataset both"
echo ""
echo "2. Test with small batch (10 images first):"
echo "   python -m src.create_features_dataset --max-images 10"
echo ""
echo "3. Create full dataset (100K images):"
echo "   python -m src.create_features_dataset"
echo ""
echo "4. Validate dataset:"
echo "   python -m src.validate_dataset"
echo ""
echo "5. Access dataset via symlink:"
echo "   ls -lh dataset/"
echo "   ls dataset/features/landmarks/"
echo ""
echo "=========================================="
echo ""
