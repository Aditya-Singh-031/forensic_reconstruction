"""
Fine-tune Stable Diffusion Inpainting (runwayml/stable-diffusion-inpainting)
for forensic face reconstruction.

Key stability fixes:
- Use full FP32 for training (no mixed precision) to avoid NaN.
- Lower default LR to 1e-6.
- Gradient clipping + NaN cleaning on grads.
- Only train UNet; VAE + text encoder frozen.
- Simple diffusion MSE loss only (no extra perceptual/identity loss).
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

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler

from src.data_loader import create_dataloaders
from src.losses import ForensicReconstructionLoss  # still used for val PSNR/vis, but not backprop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 2.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------
class SDFineTuner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 1

        # FP32 training for stability
        self.dtype = torch.float32

        # Dirs
        self.output_dir = Path(args.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "visualizations"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("LOADING STABLE DIFFUSION INPAINTING MODEL")
        logger.info("=" * 60)

        self.load_sd_model(args.sd_model_path)

        # Data
        self.train_loader, self.val_loader = create_dataloaders(
            feature_index_path=args.index_path,
            split_path=args.split_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=512,
        )

        # Loss for evaluation only (no backprop through this)
        self.eval_criterion = ForensicReconstructionLoss(
            device=self.device,
            w_pixel=1.0,
            w_perceptual=0.5,
            w_identity=0.05,
            hole_weight=6.0,
        )

        # Optimizer: only UNet
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # Simple scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-7
        )

        self.best_psnr = 0.0

        if args.resume:
            self.load_checkpoint(args.resume)

        logger.info(f"✓ Ready to fine-tune on {self.device} ({self.dtype})")
        logger.info("=" * 60)

    # -----------------------------------------------------
    # Model loading
    # -----------------------------------------------------
    def load_sd_model(self, model_path: str):
        logger.info(f"Loading from: {model_path}")

        # Load in FP32 to avoid NaNs; we explicitly set dtype later.
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Components
        self.vae = self.pipeline.vae.to(self.device, dtype=self.dtype)
        self.unet = self.pipeline.unet.to(self.device, dtype=self.dtype)
        self.text_encoder = self.pipeline.text_encoder.to(self.device, dtype=self.dtype)
        self.tokenizer = self.pipeline.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        # Freeze everything except UNet
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()

        logger.info(f"✓ VAE loaded (frozen, {self.dtype})")
        logger.info("✓ Text Encoder loaded (frozen)")
        logger.info("✓ UNet loaded (trainable)")
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        logger.info(f"  UNet parameters: {trainable_params / 1e6:.2f}M")

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def encode_prompt(self, batch_size: int):
        prompt = ["high quality face reconstruction"] * batch_size

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return text_embeddings

    def update_curriculum(self, epoch: int):
        level = 1
        if epoch > 10:
            level = 2
        if epoch > 25:
            level = 3

        if self.train_loader.dataset.corruption_level != level:
            logger.info(f"\n🎓 CURRICULUM: Level {level}\n")
            self.train_loader.dataset.corruption_level = level
            self.val_loader.dataset.corruption_level = level

    # -----------------------------------------------------
    # Training epoch
    # -----------------------------------------------------
    def train_epoch(self, epoch: int):
        self.unet.train()
        self.update_curriculum(epoch)

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            corrupted = batch["corrupted"].to(self.device, dtype=self.dtype)
            target = batch["target"].to(self.device, dtype=self.dtype)
            mask = batch["mask"].to(self.device, dtype=self.dtype)

            bsz = corrupted.shape[0]

            # [-1,1] -> [0,1]
            corrupted_01 = (corrupted + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0

            with torch.no_grad():
                latents_corrupted = self.vae.encode(corrupted_01).latent_dist.sample()
                latents_target = self.vae.encode(target_01).latent_dist.sample()
                latents_corrupted *= self.vae.config.scaling_factor
                latents_target *= self.vae.config.scaling_factor

            text_embeddings = self.encode_prompt(bsz)

            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=self.device,
            ).long()

            noisy_latents = self.noise_scheduler.add_noise(latents_target, noise, timesteps)

            # mask to latent size
            mask_latent = F.interpolate(
                mask,
                size=(latents_target.shape[-2], latents_target.shape[-1]),
                mode="nearest",
            )

            latent_model_input = torch.cat(
                [noisy_latents, mask_latent, latents_corrupted], dim=1
            )

            # Forward
            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = F.mse_loss(noise_pred, noise)

            # Backward with safety
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Clean NaN/inf grads just in case
            for p in self.unet.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1e5, neginf=-1e5)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step % self.args.log_interval == 0 and step > 0:
                logger.info(f"[{step}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int):
        self.unet.eval()

        val_loss = 0.0
        val_psnr = 0.0
        n_samples = 0

        max_batches = 20  # keep validation fast

        for i, batch in enumerate(tqdm(self.val_loader, desc=f"Val {epoch}")):
            if i >= max_batches:
                break

            corrupted = batch["corrupted"].to(self.device, dtype=self.dtype)
            target = batch["target"].to(self.device, dtype=self.dtype)
            mask = batch["mask"].to(self.device, dtype=self.dtype)

            bsz = corrupted.shape[0]

            corrupted_01 = (corrupted + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0

            latents_corrupted = self.vae.encode(corrupted_01).latent_dist.sample()
            latents_target = self.vae.encode(target_01).latent_dist.sample()
            latents_corrupted *= self.vae.config.scaling_factor
            latents_target *= self.vae.config.scaling_factor

            text_embeddings = self.encode_prompt(bsz)

            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=self.device,
            ).long()

            noisy_latents = self.noise_scheduler.add_noise(latents_target, noise, timesteps)

            mask_latent = F.interpolate(
                mask,
                size=(latents_target.shape[-2], latents_target.shape[-1]),
                mode="nearest",
            )
            latent_model_input = torch.cat(
                [noisy_latents, mask_latent, latents_corrupted], dim=1
            )

            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()

            # very rough PSNR approx: single-step denoise
            pred_latents = noisy_latents - noise_pred
            pred_images = self.vae.decode(
                pred_latents / self.vae.config.scaling_factor
            ).sample
            pred_images = pred_images * 2.0 - 1.0

            for j in range(bsz):
                val_psnr += compute_psnr(pred_images[j], target[j])
                n_samples += 1

            if i == 0:
                self.save_visuals_inference(corrupted[:4], target[:4], mask[:4], epoch)

        avg_loss = val_loss / max(1, min(max_batches, len(self.val_loader)))
        avg_psnr = val_psnr / max(1, n_samples)
        return avg_loss, avg_psnr

    # -----------------------------------------------------
    # Visualization (full pipeline)
    # -----------------------------------------------------
    @torch.no_grad()
    def save_visuals_inference(self, corrupted, target, mask, epoch: int):
        self.unet.eval()

        corrupted_01 = (corrupted + 1.0) / 2.0
        mask_01 = mask

        import torchvision.transforms as T

        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()

        out_tensors = []
        for i in range(corrupted.shape[0]):
            img_pil = to_pil(corrupted_01[i].cpu())
            mask_pil = to_pil(mask_01[i].cpu())

            out_img = self.pipeline(
                prompt="high quality face reconstruction",
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            out_tensors.append(to_tensor(out_img))

        out_tensors = torch.stack(out_tensors).to(self.device)
        out_tensors = out_tensors * 2.0 - 1.0

        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)

        n = corrupted.shape[0]
        grid = torch.cat(
            [denorm(corrupted[:n]), denorm(out_tensors[:n]), denorm(target[:n])], dim=0
        )
        grid = make_grid(grid, nrow=n, padding=2)

        save_path = self.vis_dir / f"epoch_{epoch:03d}.png"
        save_image(grid, save_path)
        logger.info(f"✓ Saved visualization: {save_path}")

    # -----------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------
    def save_checkpoint(self, epoch: int, psnr: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "unet": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }

        torch.save(state, self.ckpt_dir / "latest.pth")

        if is_best:
            torch.save(state, self.ckpt_dir / "best_unet.pth")
            self.pipeline.save_pretrained(self.ckpt_dir / "best_pipeline")
            logger.info(f"⭐ New BEST! PSNR: {psnr:.2f} dB saved.")

    def load_checkpoint(self, path: str):
        logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt["unet"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_psnr = ckpt.get("best_psnr", 0.0)

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info("STARTING SD FINE-TUNING (FP32, SAFE)")
        logger.info("=" * 60 + "\n")

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_psnr = self.validate(epoch)
            self.scheduler.step()

            dt = time.time() - t0
            logger.info("\n" + "=" * 60)
            logger.info(f"EPOCH {epoch} finished in {dt:.1f}s")
            logger.info(f"  Train Loss : {train_loss:.6f}")
            logger.info(f"  Val Loss   : {val_loss:.6f}")
            logger.info(f"  Val PSNR   : {val_psnr:.2f} dB")
            logger.info(f"  LR         : {self.optimizer.param_groups[0]['lr']:.8f}")
            logger.info("=" * 60 + "\n")

            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            self.save_checkpoint(epoch, val_psnr, is_best)

        logger.info("\n" + "=" * 60)
        logger.info("FINE-TUNING COMPLETE")
        logger.info(f"Best PSNR: {self.best_psnr:.2f} dB")
        logger.info("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    base_dir = Path("/home/teaching/G14/forensic_reconstruction")

    parser.add_argument("--sd-model-path", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument(
        "--index-path",
        default=str(base_dir / "dataset/metadata/feature_index.json"),
    )
    parser.add_argument(
        "--split-path",
        default=str(base_dir / "dataset/metadata/splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "output/sd_finetuning"),
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    trainer = SDFineTuner(args)
    trainer.run()


if __name__ == "__main__":
    main()
