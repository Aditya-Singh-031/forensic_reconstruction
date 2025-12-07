"""
Fine-tune Stable Diffusion Inpainting for Forensic Face Reconstruction.
Uses pre-trained SD model and fine-tunes on corrupted face dataset.
"""

import os
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel

from src.data_loader import create_dataloaders
from src.losses import ForensicReconstructionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# METRICS
# ============================================================
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 2.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()


# ============================================================
# SD FINE-TUNING TRAINER
# ============================================================
class SDFineTuner:
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
        
        logger.info("="*60)
        logger.info("LOADING STABLE DIFFUSION INPAINTING MODEL")
        logger.info("="*60)
        
        # Load Pre-trained SD Inpainting Components
        self.load_sd_model(args.sd_model_path)
        
        # Data Loaders
        self.train_loader, self.val_loader = create_dataloaders(
            feature_index_path=args.index_path,
            split_path=args.split_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=512
        )
        
        # Losses
        self.criterion = ForensicReconstructionLoss(
            device=self.device,
            w_pixel=1.0,
            w_perceptual=0.8,
            w_identity=0.1,
            hole_weight=6.0
        )
        
        # Optimizer (only UNet parameters)
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        self.best_psnr = 0.0
        
        if args.resume:
            self.load_checkpoint(args.resume)
        
        logger.info(f"✓ Ready to fine-tune on {self.device}")
        logger.info("="*60)
    
    def load_sd_model(self, model_path):
        """Load pre-trained Stable Diffusion Inpainting model."""
        logger.info(f"Loading from: {model_path}")
        
        # Load full pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Extract components
        self.vae = self.pipeline.vae.to(self.device)
        self.unet = self.pipeline.unet.to(self.device)
        self.text_encoder = self.pipeline.text_encoder.to(self.device)
        self.tokenizer = self.pipeline.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Freeze VAE and Text Encoder (only train UNet)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        
        # Set to training mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        
        logger.info(f"✓ VAE loaded (frozen)")
        logger.info(f"✓ Text Encoder loaded (frozen)")
        logger.info(f"✓ UNet loaded (trainable)")
        logger.info(f"  UNet parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    def encode_prompt(self, batch_size):
        """Encode empty prompt for unconditional generation."""
        # For inpainting, we use empty prompt or "high quality face"
        prompt = ["high quality face reconstruction"] * batch_size
        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def update_curriculum(self, epoch):
        """Update corruption level based on epoch."""
        new_level = 1
        if epoch > 15: new_level = 2
        if epoch > 35: new_level = 3
        
        if self.train_loader.dataset.corruption_level != new_level:
            logger.info(f"\n🎓 CURRICULUM: Switching to Level {new_level}\n")
            self.train_loader.dataset.corruption_level = new_level
            self.val_loader.dataset.corruption_level = new_level
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.unet.train()
        self.update_curriculum(epoch)
        
        total_loss = 0
        total_diffusion_loss = 0
        total_recon_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            corrupted = batch['corrupted'].to(self.device)  # [B, 3, 512, 512] in [-1, 1]
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)  # [B, 1, 512, 512]
            
            batch_size = corrupted.shape[0]
            
            # 1. Encode images to latent space
            with torch.no_grad():
                # Convert to [0, 1] for VAE
                corrupted_01 = (corrupted + 1.0) / 2.0
                target_01 = (target + 1.0) / 2.0
                
                latents_corrupted = self.vae.encode(corrupted_01).latent_dist.sample()
                latents_target = self.vae.encode(target_01).latent_dist.sample()
                latents_corrupted *= self.vae.config.scaling_factor
                latents_target *= self.vae.config.scaling_factor
            
            # 2. Get text embeddings
            text_embeddings = self.encode_prompt(batch_size)
            
            # 3. Add noise to target latents (diffusion training)
            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents_target, noise, timesteps)
            
            # 4. Prepare UNet input (masked inpainting conditioning)
            # Resize mask to latent size
            mask_latent = F.interpolate(
                mask, 
                size=(latents_target.shape[-2], latents_target.shape[-1]),
                mode='nearest'
            )
            
            # Concatenate: [noisy_latents, mask, corrupted_latents]
            # SD Inpainting expects 9 channels: 4 (latent) + 1 (mask) + 4 (masked_latent)
            latent_model_input = torch.cat([noisy_latents, mask_latent, latents_corrupted], dim=1)
            
            # 5. Predict noise
            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # 6. Diffusion loss (MSE between predicted and actual noise)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            
            # 7. Decode prediction for perceptual/identity loss
            with torch.no_grad():
                # Denoise to get predicted clean latents
                predicted_latents = self.noise_scheduler.step(
                    noise_pred, timesteps[0], noisy_latents
                ).pred_original_sample
                
                # Decode to image space
                predicted_images = self.vae.decode(predicted_latents / self.vae.config.scaling_factor).sample
                
                # Convert back to [-1, 1]
                predicted_images = predicted_images * 2.0 - 1.0
            
            # 8. Reconstruction loss (perceptual + identity)
            recon_losses = self.criterion(predicted_images, target, mask)
            
            # 9. Combined loss
            loss = diffusion_loss + 0.5 * recon_losses['total']
            
            # 10. Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_recon_loss += recon_losses['total'].item()
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'diff': f"{diffusion_loss.item():.4f}",
                'recon': f"{recon_losses['total'].item():.4f}"
            })
            
            if i % self.args.log_interval == 0 and i > 0:
                logger.info(
                    f"[{i}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Diffusion: {diffusion_loss.item():.4f} | "
                    f"Recon: {recon_losses['total'].item():.4f}"
                )
        
        n = len(self.train_loader)
        return total_loss / n, total_diffusion_loss / n, total_recon_loss / n
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate and generate samples."""
        self.unet.eval()
        
        val_loss = 0
        val_psnr = 0
        
        for i, batch in enumerate(tqdm(self.val_loader, desc=f"Val {epoch}")):
            corrupted = batch['corrupted'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Convert to [0, 1] for pipeline
            corrupted_01 = (corrupted + 1.0) / 2.0
            mask_01 = mask
            
            # Generate using full pipeline (inference)
            output = self.pipeline(
                prompt=["high quality face reconstruction"] * corrupted.shape[0],
                image=corrupted_01,
                mask_image=mask_01,
                num_inference_steps=20,  # Fast inference
                guidance_scale=7.5
            ).images
            
            # Convert PIL to tensor
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            output_tensors = torch.stack([to_tensor(img) for img in output]).to(self.device)
            output_tensors = output_tensors * 2.0 - 1.0  # [0,1] -> [-1,1]
            
            # Compute losses
            losses = self.criterion(output_tensors, target, mask)
            val_loss += losses['total'].item()
            
            # PSNR
            for j in range(output_tensors.shape[0]):
                val_psnr += compute_psnr(output_tensors[j], target[j])
            
            # Save first batch visualization
            if i == 0:
                self.save_visuals(corrupted, output_tensors, target, epoch)
            
            # Only validate on subset (faster)
            if i >= 50:
                break
        
        n = min(50, len(self.val_loader))
        n_samples = n * self.args.batch_size
        
        return val_loss / n, val_psnr / n_samples
    
    def save_visuals(self, corrupted, reconstructed, target, epoch):
        """Save visualization grid."""
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
        logger.info(f"✓ Saved: {save_path}")
    
    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save UNet checkpoint."""
        state = {
            'epoch': epoch,
            'unet': self.unet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
        }
        
        torch.save(state, self.ckpt_dir / "latest.pth")
        
        if is_best:
            torch.save(state, self.ckpt_dir / "best_unet.pth")
            # Also save full pipeline
            self.pipeline.save_pretrained(self.ckpt_dir / "best_pipeline")
            logger.info(f"⭐ NEW BEST! PSNR: {psnr:.2f} dB")
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt['unet'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_psnr = ckpt.get('best_psnr', 0.0)
    
    def run(self):
        """Main training loop."""
        logger.info("\n" + "="*60)
        logger.info("STARTING FINE-TUNING")
        logger.info("="*60 + "\n")
        
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            start_t = time.time()
            
            # Train
            t_loss, t_diff, t_recon = self.train_epoch(epoch)
            
            # Validate
            v_loss, v_psnr = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Log
            epoch_time = time.time() - start_t
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch} ({epoch_time:.1f}s)")
            logger.info(f"{'='*60}")
            logger.info(f"  Train: Loss={t_loss:.4f} (Diff={t_diff:.4f}, Recon={t_recon:.4f})")
            logger.info(f"  Val:   Loss={v_loss:.4f}, PSNR={v_psnr:.2f} dB")
            logger.info(f"  LR:    {self.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"{'='*60}\n")
            
            # Save
            is_best = v_psnr > self.best_psnr
            if is_best:
                self.best_psnr = v_psnr
            self.save_checkpoint(epoch, v_psnr, is_best)
        
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETE!")
        logger.info(f"Best PSNR: {self.best_psnr:.2f} dB")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    base_dir = Path("/home/teaching/G14/forensic_reconstruction")
    parser.add_argument("--sd-model-path", default=str(base_dir / "models/stable_diffusion_inpainting"))
    parser.add_argument("--index-path", default=str(base_dir / "dataset/metadata/feature_index.json"))
    parser.add_argument("--split-path", default=str(base_dir / "dataset/metadata/splits.json"))
    parser.add_argument("--output-dir", default=str(base_dir / "output/sd_finetuning"))
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)  # SD needs more memory
    parser.add_argument("--lr", type=float, default=5e-6)  # Lower LR for fine-tuning
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = SDFineTuner(args)
    trainer.run()


if __name__ == "__main__":
    main()
