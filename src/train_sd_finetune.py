"""
Fine-tune Stable Diffusion Inpainting for Forensic Face Reconstruction.
Fixed dtype handling for FP16 models.
"""

import os
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
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


def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 2.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()


class SDFineTuner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 1
        
        # Set dtype
        self.dtype = torch.float16 if args.use_fp16 else torch.float32
        
        # Directories
        self.output_dir = Path(args.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "visualizations"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("LOADING STABLE DIFFUSION INPAINTING MODEL")
        logger.info("="*60)
        
        # Load Pre-trained SD
        self.load_sd_model(args.sd_model_path)
        
        # Data Loaders
        self.train_loader, self.val_loader = create_dataloaders(
            feature_index_path=args.index_path,
            split_path=args.split_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=512
        )
        
        # Losses (FP32 for stability)
        self.criterion = ForensicReconstructionLoss(
            device=self.device,
            w_pixel=1.0,
            w_perceptual=0.5,  # Lower weight for training stability
            w_identity=0.05,   # Lower weight for training stability
            hole_weight=6.0
        )
        
        # Optimizer - only UNet
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-7
        )
        
        self.best_psnr = 0.0
        
        if args.resume:
            self.load_checkpoint(args.resume)
        
        logger.info(f"✓ Ready to fine-tune on {self.device} ({self.dtype})")
        logger.info("="*60)
    
    def load_sd_model(self, model_path):
        """Load pre-trained Stable Diffusion."""
        logger.info(f"Loading from: {model_path}")
        
        # Load pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        self.vae = self.pipeline.vae.to(self.device, dtype=self.dtype)
        self.unet = self.pipeline.unet.to(self.device, dtype=self.dtype)
        self.text_encoder = self.pipeline.text_encoder.to(self.device, dtype=self.dtype)
        self.tokenizer = self.pipeline.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Freeze VAE and Text Encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        
        logger.info(f"✓ VAE loaded (frozen, {self.dtype})")
        logger.info(f"✓ Text Encoder loaded (frozen)")
        logger.info(f"✓ UNet loaded (trainable)")
        logger.info(f"  UNet parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    def encode_prompt(self, batch_size):
        """Encode text prompt."""
        prompt = ["high quality face reconstruction"] * batch_size
        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def update_curriculum(self, epoch):
        """Update corruption level."""
        new_level = 1
        if epoch > 10: new_level = 2
        if epoch > 25: new_level = 3
        
        if self.train_loader.dataset.corruption_level != new_level:
            logger.info(f"\n🎓 CURRICULUM: Level {new_level}\n")
            self.train_loader.dataset.corruption_level = new_level
            self.val_loader.dataset.corruption_level = new_level
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.unet.train()
        self.update_curriculum(epoch)
        
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            # Move to device and convert to model dtype
            corrupted = batch['corrupted'].to(self.device, dtype=self.dtype)
            target = batch['target'].to(self.device, dtype=self.dtype)
            mask = batch['mask'].to(self.device, dtype=self.dtype)
            
            batch_size = corrupted.shape[0]
            
            # Convert [-1, 1] to [0, 1] for VAE
            corrupted_01 = (corrupted + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0
            
            # Encode to latent space
            with torch.no_grad():
                latents_corrupted = self.vae.encode(corrupted_01).latent_dist.sample()
                latents_target = self.vae.encode(target_01).latent_dist.sample()
                latents_corrupted *= self.vae.config.scaling_factor
                latents_target *= self.vae.config.scaling_factor
            
            # Get text embeddings
            text_embeddings = self.encode_prompt(batch_size)
            
            # Add noise to target latents
            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents_target, noise, timesteps)
            
            # Prepare inpainting input
            mask_latent = F.interpolate(
                mask,
                size=(latents_target.shape[-2], latents_target.shape[-1]),
                mode='nearest'
            )
            
            # UNet input: [noisy_latents, mask, corrupted_latents]
            latent_model_input = torch.cat([noisy_latents, mask_latent, latents_corrupted], dim=1)
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Diffusion loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if i % self.args.log_interval == 0 and i > 0:
                logger.info(f"[{i}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate."""
        self.unet.eval()
        
        val_loss = 0
        val_psnr = 0
        n_samples = 0
        
        # Only validate on subset for speed
        max_batches = 20
        
        for i, batch in enumerate(tqdm(self.val_loader, desc=f"Val {epoch}")):
            if i >= max_batches:
                break
            
            corrupted = batch['corrupted'].to(self.device, dtype=self.dtype)
            target = batch['target'].to(self.device, dtype=self.dtype)
            mask = batch['mask'].to(self.device, dtype=self.dtype)
            
            batch_size = corrupted.shape[0]
            
            # Convert for VAE
            corrupted_01 = (corrupted + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0
            
            # Encode
            latents_corrupted = self.vae.encode(corrupted_01).latent_dist.sample()
            latents_target = self.vae.encode(target_01).latent_dist.sample()
            latents_corrupted *= self.vae.config.scaling_factor
            latents_target *= self.vae.config.scaling_factor
            
            # Get text
            text_embeddings = self.encode_prompt(batch_size)
            
            # Add noise
            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents_target, noise, timesteps)
            
            # Prepare input
            mask_latent = F.interpolate(
                mask,
                size=(latents_target.shape[-2], latents_target.shape[-1]),
                mode='nearest'
            )
            
            latent_model_input = torch.cat([noisy_latents, mask_latent, latents_corrupted], dim=1)
            
            # Predict
            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()
            
            # Decode for PSNR (only first timestep for speed)
            if timesteps[0] < 100:  # Only decode low noise samples
                pred_latents = noisy_latents - noise_pred
                pred_images = self.vae.decode(pred_latents / self.vae.config.scaling_factor).sample
                pred_images = pred_images * 2.0 - 1.0
                
                for j in range(batch_size):
                    psnr = compute_psnr(pred_images[j], target[j])
                    val_psnr += psnr
                    n_samples += 1
            
            # Save visualization (first batch)
            if i == 0:
                # Generate full inference samples
                self.save_visuals_inference(corrupted[:4], target[:4], mask[:4], epoch)
        
        avg_loss = val_loss / min(max_batches, len(self.val_loader))
        avg_psnr = val_psnr / max(n_samples, 1)
        
        return avg_loss, avg_psnr
    
    def save_visuals_inference(self, corrupted, target, mask, epoch):
        """Save visualization using full inference pipeline."""
        self.unet.eval()
        
        # Convert to [0, 1]
        corrupted_01 = (corrupted + 1.0) / 2.0
        mask_01 = mask
        
        # Convert to PIL for pipeline
        import torchvision.transforms as T
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        
        results = []
        
        for i in range(corrupted.shape[0]):
            img_pil = to_pil(corrupted_01[i].cpu())
            mask_pil = to_pil(mask_01[i].cpu())
            
            # Generate
            output = self.pipeline(
                prompt="high quality face reconstruction",
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            results.append(to_tensor(output))
        
        # Stack and save
        results = torch.stack(results).to(self.device)
        results = results * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
        
        imgs = torch.cat([
            denorm(corrupted),
            denorm(results),
            denorm(target)
        ], dim=0)
        
        grid = make_grid(imgs, nrow=4, padding=2)
        save_path = self.vis_dir / f"epoch_{epoch:03d}.png"
        save_image(grid, save_path)
        logger.info(f"✓ Saved: {save_path}")
        
        self.unet.train()
    
    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save checkpoint."""
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
            self.pipeline.save_pretrained(self.ckpt_dir / "best_pipeline")
            logger.info(f"⭐ BEST! PSNR: {psnr:.2f} dB")
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        logger.info(f"Resuming: {path}")
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
            t_loss = self.train_epoch(epoch)
            
            # Validate
            v_loss, v_psnr = self.validate(epoch)
            
            # Scheduler
            self.scheduler.step()
            
            # Log
            epoch_time = time.time() - start_t
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch} ({epoch_time:.1f}s)")
            logger.info(f"{'='*60}")
            logger.info(f"  Train Loss: {t_loss:.4f}")
            logger.info(f"  Val Loss:   {v_loss:.4f}")
            logger.info(f"  Val PSNR:   {v_psnr:.2f} dB")
            logger.info(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
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
    
    base_dir = Path("/home/teaching/G14/forensic_reconstruction")
    parser.add_argument("--sd-model-path", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--index-path", default=str(base_dir / "dataset/metadata/feature_index.json"))
    parser.add_argument("--split-path", default=str(base_dir / "dataset/metadata/splits.json"))
    parser.add_argument("--output-dir", default=str(base_dir / "output/sd_finetuning"))
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = SDFineTuner(args)
    trainer.run()


if __name__ == "__main__":
    main()
