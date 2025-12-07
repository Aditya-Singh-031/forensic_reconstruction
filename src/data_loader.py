"""
Corrupted Face Dataset Loader for Forensic Reconstruction Training.
Implements random feature corruption with exact position compositing.
Robust handling for training stability.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as T

# Allow loading truncated images (common issue with large datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Exclude 'mouth_inner' to prevent artifacts
CORRUPTIBLE_FEATURES = [
    "eyes_left", "eyes_right", 
    "eyebrows_left", "eyebrows_right",
    "nose", "mouth_outer"
]

class CorruptedFaceDataset(Dataset):
    """
    Dataset that generates corrupted faces by randomly compositing features.
    """
    
    def __init__(
        self,
        feature_index_path: str,
        split_path: str,
        split_name: str = "train",
        corruption_level: int = 2,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.split_name = split_name
        self.corruption_level = corruption_level
        self.image_size = image_size
        self.augment = augment and (split_name == "train")
        
        # Load feature index
        with open(feature_index_path) as f:
            self.feature_index = json.load(f)
        
        # Load split
        with open(split_path) as f:
            splits = json.load(f)
        
        # Filter image_names to ensure they exist in index
        valid_names = [n for n in splits[split_name] if n in self.feature_index]
        self.image_names = valid_names
        
        # Build feature pools for sampling (Optimization: Pre-build list)
        self.feature_pools = {ft: [] for ft in CORRUPTIBLE_FEATURES}
        for img_name, entry in self.feature_index.items():
            for ft in CORRUPTIBLE_FEATURES:
                if ft in entry.get("features", {}) and ft in entry.get("bboxes", {}):
                    self.feature_pools[ft].append(img_name)
        
        logger.info(f"Loaded {split_name.upper()} set: {len(self.image_names)} images")
        
        # Transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if self.augment:
            self.augment_transform = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.RandomGrayscale(p=0.1),
            ])

    def _get_corruption_count(self) -> int:
        """Determine how many features to corrupt."""
        available = len(CORRUPTIBLE_FEATURES)
        if self.corruption_level == 1:
            return random.randint(1, 2)
        elif self.corruption_level == 2:
            return random.randint(3, 4)
        else:
            return random.randint(5, available)

    def _load_rgba(self, path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            return img
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one training sample with robust error handling.
        If loading fails, try the next index to avoid crashing training.
        """
        attempts = 0
        while attempts < 3:  # Retry up to 3 times
            try:
                curr_idx = (idx + attempts) % len(self.image_names)
                return self._process_item(curr_idx)
            except Exception as e:
                attempts += 1
                # logger.warning(f"Failed to load idx {curr_idx}: {e}")
        
        # Fallback: Return simple zero tensors if all retries fail
        return self._get_empty_sample()

    def _process_item(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[idx]
        entry = self.feature_index[image_name]
        
        # 1. Load Base & Target
        face_path = entry["features"]["face_contour"]
        jawline_path = entry["features"]["jawline"]
        
        face_img = self._load_rgba(face_path)
        target_img = self._load_rgba(jawline_path)
        
        if face_img is None or target_img is None:
            raise ValueError("Missing base images")

        # 2. Corrupt
        corrupted = face_img.copy()
        
        # Mask: 1 channel, size of image
        w, h = corrupted.size
        corruption_mask = np.zeros((h, w), dtype=np.float32)
        
        k = self._get_corruption_count()
        feats_to_corrupt = random.sample(CORRUPTIBLE_FEATURES, k)
        
        for ft in feats_to_corrupt:
            # Validate target has slot
            if ft not in entry["features"] or ft not in entry["bboxes"]: continue
            
            # Get placement bbox
            bbox = entry["bboxes"][ft]
            x_min, y_min = int(bbox["x_min"]), int(bbox["y_min"])
            target_w = int(bbox["x_max"] - x_min)
            target_h = int(bbox["y_max"] - y_min)
            
            if target_w <= 0 or target_h <= 0: continue
            
            # Sample donor
            if not self.feature_pools[ft]: continue
            donor_name = random.choice(self.feature_pools[ft])
            donor_entry = self.feature_index[donor_name]
            
            if ft not in donor_entry["features"]: continue
            
            donor_img = self._load_rgba(donor_entry["features"][ft])
            if donor_img is None: continue
            
            # Resize donor
            donor_resized = donor_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            alpha_mask = donor_resized.split()[3]
            
            # Paste
            corrupted.paste(donor_resized, (x_min, y_min), alpha_mask)
            
            # Update binary mask (Robust Slicing)
            y_end = min(y_min + target_h, h)
            x_end = min(x_min + target_w, w)
            
            # Crop mask to match the safe slice
            mask_arr = np.array(alpha_mask, dtype=np.float32) / 255.0
            visible_h = y_end - y_min
            visible_w = x_end - x_min
            
            if visible_h > 0 and visible_w > 0:
                mask_crop = mask_arr[:visible_h, :visible_w]
                corruption_mask[y_min:y_end, x_min:x_end] = np.maximum(
                    corruption_mask[y_min:y_end, x_min:x_end],
                    mask_crop
                )

        # 3. Post-Process
        # Convert to RGB (white background handling)
        def to_rgb(img_rgba):
            bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
            bg.paste(img_rgba, (0, 0), img_rgba)
            return bg

        corrupted_rgb = to_rgb(corrupted)
        target_rgb = to_rgb(target_img)
        
        # Resize
        if self.image_size != w:
            corrupted_rgb = corrupted_rgb.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            target_rgb = target_rgb.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            # Resize mask (Nearest Neighbor to keep binary-ish)
            mask_img = Image.fromarray(corruption_mask)
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            corruption_mask = np.array(mask_img)

        # Augment (Simultaneously)
        if self.augment:
            # We seed the RNG to apply the EXACT SAME transform to input & target
            # This is critical so they stay pixel-aligned
            seed = np.random.randint(2147483647)
            
            random.seed(seed)
            torch.manual_seed(seed)
            corrupted_rgb = self.augment_transform(corrupted_rgb)
            
            random.seed(seed)
            torch.manual_seed(seed)
            target_rgb = self.augment_transform(target_rgb)

        # To Tensor
        inp = self.normalize(self.to_tensor(corrupted_rgb))
        tgt = self.normalize(self.to_tensor(target_rgb))
        msk = torch.from_numpy(corruption_mask).unsqueeze(0)  # [1, H, W]

        return {
            "corrupted": inp,
            "target": tgt,
            "mask": msk,
            "name": image_name
        }

    def _get_empty_sample(self):
        """Emergency fallback to prevent crashing."""
        z = torch.zeros((3, self.image_size, self.image_size))
        m = torch.zeros((1, self.image_size, self.image_size))
        return {"corrupted": z, "target": z, "mask": m, "name": "error"}

def create_dataloaders(
    feature_index_path: str,
    split_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512,
):
    """Factory function."""
    train_ds = CorruptedFaceDataset(feature_index_path, split_path, "train", corruption_level=2, image_size=image_size)
    val_ds = CorruptedFaceDataset(feature_index_path, split_path, "val", corruption_level=2, image_size=image_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# TEST BLOCK
if __name__ == "__main__":
    # Quick Test
    import matplotlib.pyplot as plt
    
    BASE = Path("/home/teaching/G14/forensic_reconstruction/dataset")
    IDX = BASE / "metadata" / "feature_index.json"
    SPLIT = BASE / "metadata" / "splits.json"
    
    if IDX.exists():
        ds = CorruptedFaceDataset(str(IDX), str(SPLIT), split_name="train", image_size=512)
        sample = ds[0]
        
        print(f"Sample: {sample['name']}")
        print(f"Corrupted shape: {sample['corrupted'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        
        # Un-normalize for display
        def denorm(t): return (t * 0.5 + 0.5).permute(1, 2, 0).numpy()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(denorm(sample["corrupted"]))
        ax[0].set_title("Input (Corrupted)")
        ax[1].imshow(denorm(sample["target"]))
        ax[1].set_title("Target (Original)")
        ax[2].imshow(sample["mask"].squeeze(), cmap="gray")
        ax[2].set_title("Corruption Mask")
        plt.savefig("dataloader_check.png")
        print("Saved dataloader_check.png")