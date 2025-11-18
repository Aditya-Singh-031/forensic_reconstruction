#!/bin/bash

# Clean Dataset - Remove all generated files
# Keep raw_images and raw_images/* (FFHQ, CelebA-HQ)

echo "=========================================="
echo "CLEANING DATASET"
echo "=========================================="
echo ""

DATASET_ROOT="/DATA/facial_features_dataset"

echo "This will DELETE:"
echo "  - All extracted features"
echo "  - All metadata files"
echo "  - All processing logs"
echo ""
echo "It will KEEP:"
echo "  - Raw images (FFHQ, CelebA-HQ)"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Deleting generated files..."

# Remove all feature directories content
rm -rf "$DATASET_ROOT/features"/*
echo "✓ Removed: features/*"

# Remove metadata
rm -rf "$DATASET_ROOT/metadata"/*
echo "✓ Removed: metadata/*"

# Remove annotations
rm -rf "$DATASET_ROOT/annotations"/*
echo "✓ Removed: annotations/*"

echo ""
echo "Recreating empty directories..."

# Recreate structure
mkdir -p "$DATASET_ROOT/features"/{segmentation,landmarks,eyes_left,eyes_right,eyebrows_left,eyebrows_right,nose,mouth_outer,mouth_inner,ears_left,ears_right,face_contour,jawline}
mkdir -p "$DATASET_ROOT/metadata"
mkdir -p "$DATASET_ROOT/annotations"

echo "✓ Recreated: features subdirectories"
echo "✓ Recreated: metadata"
echo "✓ Recreated: annotations"

echo ""
echo "Verifying raw images..."
ffhq_count=$(find "$DATASET_ROOT/raw_images/ffhq" -type f 2>/dev/null | wc -l)
celeba_count=$(find "$DATASET_ROOT/raw_images/celeba_hq" -type f 2>/dev/null | wc -l)

echo "  FFHQ: $ffhq_count images"
echo "  CelebA-HQ: $celeba_count images"

echo ""
echo "=========================================="
echo "CLEANUP COMPLETE!"
echo "=========================================="
echo ""
echo "Ready to process dataset again!"
echo ""
echo "Next step:"
echo "  python -m src.create_features_dataset --max-images 10"
echo ""
