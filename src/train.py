"""
Main Training Script for Forensic Face Reconstruction.
Combines Data Loader, Model, and Losses with Curriculum Learning.
"""

import os
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image, make_grid
import numpy as np

# Absolute imports
from src.data_loader import create_dataloaders
from src.model import create_model
from src.losses import ForensicReconstructionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# METRICS
# ============================================================
def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 2.0 # Range [-1, 1]
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()

def compute_ssim(pred, target):
    """Simplified SSIM monitoring."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p, mu_t = pred.mean(), target.mean()
    var_p, var_t = pred.var(), target.var()
    cov = torch.mean((pred - mu_p) * (target - mu_t))
    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2))
    return ssim.item()

# ============================================================
# TRAINER CLASS
# ============================================================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 1
        
        # Directories
        self.output_dir = Path(args.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "visualizations"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Data Loaders
        self.train_loader, self.val_loader = create_dataloaders(
            feature_index_path=args.index_path,
            split_path=args.split_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=512
        )
        
        # Set initial corruption level manually (Level 1 = Easy)
        self.train_loader.dataset.corruption_level = 1
        self.val_loader.dataset.corruption_level = 1
        
        # 2. Model
        self.model = create_model('unet_attention', device=self.device)
        
        # 3. Loss & Optimizer
        self.criterion = ForensicReconstructionLoss(
            device=self.device,
            w_pixel=1.0, 
            w_perceptual=0.8, 
            w_identity=0.1,
            hole_weight=6.0
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=1e-4
        )
        
        # Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.scaler = GradScaler() # Mixed Precision
        self.best_psnr = 0.0
        
        if args.resume:
            self.load_checkpoint(args.resume)
            
        logger.info(f"Ready to train on {self.device}")

    def update_curriculum(self, epoch):
        """Update dataset difficulty based on epoch."""
        # Curriculum Schedule:
        # Epoch 1-20: Level 1 (Easy)
        # Epoch 21-50: Level 2 (Medium)
        # Epoch 51+: Level 3 (Hard)
        new_level = 1
        if epoch > 20: new_level = 2
        if epoch > 50: new_level = 3
        
        # Check against the dataset's current level
        if self.train_loader.dataset.corruption_level != new_level:
            logger.info(f"\n🎓 CURRICULUM UPGRADE: Switching to Level {new_level}\n")
            self.train_loader.dataset.corruption_level = new_level
            self.val_loader.dataset.corruption_level = new_level

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.update_curriculum(epoch)
        
        for i, batch in enumerate(self.train_loader):
            corrupted = batch['corrupted'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                reconstructed = self.model(corrupted)
                losses = self.criterion(reconstructed, target, mask)
                loss = losses['total']
            
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if i % self.args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{i}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} (Pix:{losses['pixel']:.3f} Per:{losses['perceptual']:.3f})"
                )
                
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                corrupted = batch['corrupted'].to(self.device)
                target = batch['target'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                with autocast():
                    reconstructed = self.model(corrupted)
                    losses = self.criterion(reconstructed, target, mask)
                
                val_loss += losses['total'].item()
                val_psnr += compute_psnr(reconstructed, target)
                
                # Save Visualization (First batch only)
                if i == 0:
                    self.save_visuals(corrupted, reconstructed, target, epoch)
                    
        n = len(self.val_loader)
        return val_loss / n, val_psnr / n

    def save_visuals(self, corrupted, reconstructed, target, epoch):
        """Save grid: [Input (Corrupted) | Output (Reconstructed) | Target (Original)]"""
        def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
        
        n = min(corrupted.size(0), 4)
        imgs = torch.cat([
            denorm(corrupted[:n]), 
            denorm(reconstructed[:n]), 
            denorm(target[:n])
        ], dim=0)
        
        grid = make_grid(imgs, nrow=n, padding=2)
        save_path = self.vis_dir / f"epoch_{epoch:03d}.png"
        save_image(grid, save_path)
        logger.info(f"Saved visualization: {save_path}")

    def save_checkpoint(self, epoch, psnr, is_best=False):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_psnr': self.best_psnr,
        }
        torch.save(state, self.ckpt_dir / "latest.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pth")
            logger.info(f"⭐ New Best Model! PSNR: {psnr:.2f} dB")

    def load_checkpoint(self, path):
        logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_psnr = ckpt.get('best_psnr', 0.0)

    def run(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            start_t = time.time()
            
            # Train & Validate
            t_loss = self.train_epoch(epoch)
            v_loss, v_psnr = self.validate(epoch)
            self.scheduler.step()
            
            # Logging
            time_ep = time.time() - start_t
            logger.info(f"Epoch {epoch} ({time_ep:.1f}s) | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | PSNR: {v_psnr:.2f}")
            
            # Save
            is_best = v_psnr > self.best_psnr
            if is_best: self.best_psnr = v_psnr
            self.save_checkpoint(epoch, v_psnr, is_best)

def main():
    parser = argparse.ArgumentParser()
    # Default Paths
    base_dir = Path("/home/teaching/G14/forensic_reconstruction")
    parser.add_argument("--index-path", default=str(base_dir / "dataset/metadata/feature_index.json"))
    parser.add_argument("--split-path", default=str(base_dir / "dataset/metadata/splits.json"))
    parser.add_argument("--output-dir", default=str(base_dir / "output/training_run_1"))
    
    # Params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    Trainer(args).run()

if __name__ == "__main__":
    main()