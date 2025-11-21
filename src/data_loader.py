"""
Data Loader - Phase 3
PyTorch Dataset and DataLoader for training
Handles corruption, augmentation, and batching
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorruptedFaceDataset(Dataset):
    """PyTorch Dataset for corrupted face reconstruction."""
    
    def __init__(self,
                 feature_index: Dict,
                 image_names: List[str],
                 corruption_level: int = 2,
                 image_size: Tuple[int, int] = (256, 256),
                 augment: bool = True):
        """Initialize dataset.
        
        Args:
            feature_index: Feature index dictionary
            image_names: List of image names for this split
            corruption_level: Corruption level (1-3)
            image_size: Target image size (width, height)
            augment: Apply data augmentation
        """
        self.feature_index = feature_index
        self.image_names = image_names
        self.corruption_level = corruption_level
        self.image_size = image_size
        self.augment = augment
        
        # Feature types that can be corrupted
        self.corruptible_features = [
            "eyes_left",
            "eyes_right",
            "eyebrows_left",
            "eyebrows_right",
            "nose",
            "mouth_outer",
            "mouth_inner"
        ]
        
        # Transform for normalization
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.augment_transform = None
        
        logger.info(f"Initialized dataset with {len(self.image_names)} images")
        logger.info(f"  Corruption level: {self.corruption_level}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Augmentation: {self.augment}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_names)
    
    def load_and_resize(self, path: str) -> Image.Image:
        """Load image and resize to target size.
        
        Args:
            path: Image path
            
        Returns:
            img: Resized PIL Image
        """
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(self.image_size, Image.BICUBIC)
        return img
    
    def get_features_to_corrupt(self) -> List[str]:
        """Determine which features to corrupt based on level.
        
        Returns:
            features: List of feature types to corrupt
        """
        if self.corruption_level == 1:
            num_features = random.randint(2, 3)
        elif self.corruption_level == 2:
            num_features = random.randint(4, 5)
        elif self.corruption_level == 3:
            num_features = random.randint(6, 7)
        else:
            num_features = 4
        
        features = random.sample(
            self.corruptible_features,
            min(num_features, len(self.corruptible_features))
        )
        return features
    
    def create_corrupted_face(self, image_name: str) -> Tuple[Image.Image, List[str]]:
        """Create corrupted face by compositing random features.
        
        Args:
            image_name: Base image name
            
        Returns:
            corrupted: Corrupted face image
            features: List of corrupted feature names
        """
        # Load base face_contour
        base_path = self.feature_index[image_name]["features"]["face_contour"]
        base_img = self.load_and_resize(base_path)
        
        # Determine features to corrupt
        features_to_corrupt = self.get_features_to_corrupt()
        
        # For simplified version: return base image
        # Full version would composite random features here
        # This requires careful position alignment which we'll implement next
        
        # TODO: Implement actual feature compositing with position info
        corrupted = base_img.copy()
        
        return corrupted, features_to_corrupt
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary with tensors
                - corrupted: Corrupted face [3, H, W]
                - target: Ground truth face [3, H, W]
                - mask: Corruption mask [1, H, W]
        """
        image_name = self.image_names[idx]
        data = self.feature_index[image_name]
        
        # Create corrupted face
        corrupted_img, features = self.create_corrupted_face(image_name)
        
        # Load target (jawline)
        target_path = data["features"]["jawline"]
        target_img = self.load_and_resize(target_path)
        
        # Apply augmentation if enabled
        if self.augment and self.augment_transform is not None:
            corrupted_img = self.augment_transform(corrupted_img)
        
        # Convert to tensors
        corrupted_tensor = transforms.ToTensor()(corrupted_img)
        target_tensor = transforms.ToTensor()(target_img)
        
        # Normalize to [-1, 1]
        corrupted_tensor = self.normalize(corrupted_tensor)
        target_tensor = self.normalize(target_tensor)
        
        # Create corruption mask (simplified: all ones for now)
        mask = torch.ones(1, self.image_size[1], self.image_size[0])
        
        return {
            'corrupted': corrupted_tensor,
            'target': target_tensor,
            'mask': mask,
            'image_name': image_name,
            'corrupted_features': features
        }


def create_dataloaders(
    dataset_dir: str = "/DATA/facial_features_dataset",
    batch_size: int = 8,
    num_workers: int = 4,
    corruption_level: int = 2,
    image_size: Tuple[int, int] = (256, 256)
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders.
    
    Args:
        dataset_dir: Dataset root directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        corruption_level: Corruption level (1-3)
        image_size: Target image size
        
    Returns:
        loaders: Dictionary of DataLoaders
    """
    dataset_path = Path(dataset_dir)
    metadata_dir = dataset_path / "metadata"
    
    # Load feature index
    index_path = metadata_dir / "feature_index.json"
    with open(index_path) as f:
        feature_index = json.load(f)
    
    # Load splits
    splits_path = metadata_dir / "train_val_test_split.json"
    with open(splits_path) as f:
        splits = json.load(f)
    
    logger.info(f"Creating dataloaders:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Corruption level: {corruption_level}")
    
    # Create datasets
    train_dataset = CorruptedFaceDataset(
        feature_index,
        splits['train'],
        corruption_level=corruption_level,
        image_size=image_size,
        augment=True
    )
    
    val_dataset = CorruptedFaceDataset(
        feature_index,
        splits['val'],
        corruption_level=corruption_level,
        image_size=image_size,
        augment=False
    )
    
    test_dataset = CorruptedFaceDataset(
        feature_index,
        splits['test'],
        corruption_level=corruption_level,
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def test_dataloader():
    """Test dataloader with sample batch."""
    logger.info("Testing dataloader...")
    
    # Create loaders
    loaders = create_dataloaders(batch_size=2, num_workers=0)
    
    # Get sample batch
    train_loader = loaders['train']
    batch = next(iter(train_loader))
    
    logger.info("\nSample batch:")
    logger.info(f"  Corrupted shape: {batch['corrupted'].shape}")
    logger.info(f"  Target shape: {batch['target'].shape}")
    logger.info(f"  Mask shape: {batch['mask'].shape}")
    logger.info(f"  Image names: {batch['image_name']}")
    logger.info(f"  Corrupted features: {batch['corrupted_features']}")
    
    logger.info("\nDataloader test passed! ✓")


if __name__ == "__main__":
    test_dataloader()
