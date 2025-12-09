"""
Evaluation Script for Forensic Face Reconstruction.
Calculates PSNR, SSIM, and LPIPS on the held-out TEST set.
CPU-Optimized Version.
"""

import torch
import logging
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

# Absolute imports
from src.data_loader import CorruptedFaceDataset
from src.model import create_model
from src.losses import LPIPSLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio for [-1, 1] images."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 2.0 
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()

def compute_ssim(pred, target):
    """Simplified SSIM (Mean/Var based) for monitoring."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p, mu_t = pred.mean(), target.mean()
    var_p, var_t = pred.var(), target.var()
    cov = torch.mean((pred - mu_p) * (target - mu_t))
    return ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2))

def evaluate():
    # 1. Force CPU
    device = 'cpu'
    
    # Optimization: Use all available cores
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    logger.info(f"Running on CPU with {num_cores} threads to avoid GPU OOM.")
    
    # Paths
    base_dir = Path("/home/teaching/G14/forensic_reconstruction")
    checkpoint_path = base_dir / "output/training_run_1/checkpoints/best_model.pth"
    index_path = base_dir / "dataset/metadata/feature_index.json"
    split_path = base_dir / "dataset/metadata/splits.json"
    
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # 2. Dataset
    test_dataset = CorruptedFaceDataset(
        feature_index_path=str(index_path),
        split_path=str(split_path),
        split_name="test",
        corruption_level=3,
        image_size=512,
        augment=False
    )
    
    # Use fewer workers for DataLoader on CPU to prevent thread contention
    loader_workers = max(1, min(4, num_cores // 4))
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=loader_workers,
        pin_memory=False # No pinning needed for CPU
    )
    
    logger.info(f"Loaded TEST set: {len(test_dataset)} images")
    
    # 3. Load Model
    model = create_model('unet_attention', device=device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Use valid key
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt['model_state_dict']
        model.load_state_dict(state_dict)
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")
    except FileNotFoundError:
        logger.error("Checkpoint not found! Run training first.")
        return
    except KeyError:
        logger.error(f"Checkpoint key error. Available keys: {ckpt.keys()}")
        return

    model.eval()

    # 4. Load LPIPS
    try:
        lpips_metric = LPIPSLoss(device=device)
    except Exception as e:
        logger.warning(f"LPIPS failed to load: {e}. Skipping LPIPS.")
        lpips_metric = None
    
    # 5. Run Evaluation
    metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            corrupted = batch['corrupted'].to(device)
            target = batch['target'].to(device)
            
            # No autocast on CPU for stability
            reconstructed = model(corrupted)
            
            # Calculate metrics
            for i in range(len(corrupted)):
                p = reconstructed[i]
                t = target[i]
                
                metrics['psnr'].append(compute_psnr(p, t))
                metrics['ssim'].append(compute_ssim(p, t).item())
                
                if lpips_metric:
                    l_val = lpips_metric(p.unsqueeze(0), t.unsqueeze(0)).item()
                    metrics['lpips'].append(l_val)

    # 6. Aggregate
    avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
    avg_ssim = sum(metrics['ssim']) / len(metrics['ssim'])
    avg_lpips = sum(metrics['lpips']) / len(metrics['lpips']) if metrics['lpips'] else 0.0
    
    print("\n" + "="*40)
    print("FINAL TEST RESULTS (Corruption Level 3)")
    print("="*40)
    print(f"PSNR  : {avg_psnr:.4f} dB  (Higher is better)")
    print(f"SSIM  : {avg_ssim:.4f}     (Higher is better)")
    print(f"LPIPS : {avg_lpips:.4f}    (Lower is better)")
    print("="*40 + "\n")
    
    # Save CSV
    try:
        df = pd.DataFrame(metrics)
        output_path = base_dir / "output/training_run_1/test_metrics.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed metrics to {output_path}")
    except:
        pass

if __name__ == "__main__":
    evaluate()