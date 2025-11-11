"""
Text-to-Face Generation using Stable Diffusion (text-to-image)

This module provides a `TextToFaceGenerator` class that uses a Stable Diffusion
text-to-image pipeline to produce photorealistic faces from verbal descriptions.

Features:
  - Text prompt input with optional negative prompt
  - Adjustable inference steps (20-50) and guidance scale (7.5-15)
  - GPU/CPU support with automatic FP16/FP32 selection
  - Seed control for reproducibility
  - Memory optimizations (attention/vae slicing)
  - Batch generation (multiple images per description)

Example:
    from text_to_face import TextToFaceGenerator

    gen = TextToFaceGenerator(device='cuda')
    img = gen.generate(
        description="Adult male, 45 years old, thick mustache, large ears, dark complexion",
        num_inference_steps=30,
        guidance_scale=8.0,
        seed=123
    )
    img.save("generated_face.png")

Author: Forensic Reconstruction System
Date: 2025
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError(
        "diffusers library not installed. Install with: pip install diffusers transformers accelerate"
    )

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToFaceGenerator:
    """
    Generate faces from text descriptions using Stable Diffusion.
    
    Attributes:
        pipeline: StableDiffusionPipeline instance
        device: torch.device
        dtype: torch dtype (float16 on GPU, else float32)
    """

    DEFAULT_NEGATIVE = (
        "blurry, low quality, cartoon, sketch, painting, distorted, deformed,"
        " extra limbs, disfigured, watermark, text"
    )

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_half_precision: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
    ) -> None:
        """
        Initialize the text-to-image generator.
        
        Args:
            model_name: Diffusers model ID (text-to-image SD1.5)
            device: 'cuda', 'cpu', or None for auto
            use_half_precision: Use fp16 on GPU
            cache_dir: Optional cache directory
            enable_attention_slicing: Reduce memory usage
            enable_vae_slicing: Reduce memory usage
        """
        start = time.time()
        logger.info("Initializing TextToFaceGenerator...")

        # Device and precision
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if use_half_precision and self.device.type == "cuda":
            self.dtype = torch.float16
            logger.info("Using half precision (FP16)")
        else:
            self.dtype = torch.float32
            logger.info("Using full precision (FP32)")

        # Load pipeline
        logger.info(f"Loading Stable Diffusion pipeline: {model_name}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.pipeline = self.pipeline.to(self.device)

        # Memory optimizations
        if enable_attention_slicing:
            try:
                self.pipeline.enable_attention_slicing()
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")
        if enable_vae_slicing:
            try:
                self.pipeline.enable_vae_slicing()
            except Exception as e:
                logger.warning(f"Could not enable VAE slicing: {e}")

        # Show progress bars by default; toggle per call
        self.pipeline.set_progress_bar_config(disable=False)

        logger.info(f"✓ TextToFaceGenerator initialized in {time.time()-start:.2f}s")

    def generate(
        self,
        description: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
        show_progress: bool = True,
        return_numpy: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """
        Generate a single face image from a description.
        
        Args:
            description: Text description of target face
            negative_prompt: Things to avoid (defaults to sensible negatives)
            num_inference_steps: 20-50 typical
            guidance_scale: 7.5-15 typical
            seed: Random seed for reproducibility
            height: Generated image height
            width: Generated image width
            show_progress: Enable diffusion progress bars
            return_numpy: Return numpy array instead of PIL
        
        Returns:
            PIL Image or numpy array
        """
        if not description or not isinstance(description, str):
            raise ValueError("description must be a non-empty string")

        negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE
        num_inference_steps = max(1, min(50, int(num_inference_steps)))
        guidance_scale = float(guidance_scale)

        # Seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Progress
        self.pipeline.set_progress_bar_config(disable=not show_progress)

        try:
            result = self.pipeline(
                prompt=description,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU out of memory during generation. Try reducing steps/size, or use CPU."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Text-to-image generation failed: {e}") from e

        if return_numpy:
            return np.array(result)
        return result

    def generate_batch(
        self,
        description: str,
        num_images: int = 4,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
        show_progress: bool = True,
    ) -> List[Image.Image]:
        """
        Generate multiple images from the same description.
        
        Args are similar to `generate`, plus:
            num_images: number of images to generate
        
        Returns:
            List of PIL images
        """
        if num_images <= 0:
            raise ValueError("num_images must be >= 1")

        negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE
        num_inference_steps = max(1, min(50, int(num_inference_steps)))

        # Seed strategy: use base seed and increment per image (if provided)
        base_seed = int(seed) if seed is not None else None

        images: List[Image.Image] = []
        self.pipeline.set_progress_bar_config(disable=not show_progress)

        for i in range(num_images):
            local_seed = base_seed + i if base_seed is not None else None
            img = self.generate(
                description=description,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=local_seed,
                height=height,
                width=width,
                show_progress=show_progress,
                return_numpy=False,
            )
            images.append(img)
        return images


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Sanitize text to be safe as filename."""
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in text)
    safe = "_".join(safe.strip().split())
    return safe[:max_len] if len(safe) > max_len else safe


# Example run
if __name__ == "__main__":
    print("TextToFaceGenerator module loaded. Use via src/test_text_to_face.py CLI.")
