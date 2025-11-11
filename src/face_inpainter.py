"""
Face Inpainting Module using Stable Diffusion

This module provides a `FaceInpainter` class that uses Stable Diffusion v1.5
inpainting to realistically reconstruct occluded facial features. It's designed
for forensic reconstruction and data augmentation tasks.

Features:
  - Stable Diffusion v1.5 inpainting model (runwayml/stable-diffusion-inpainting)
  - Text-guided generation with customizable prompts
  - Adjustable inference steps (20-50) and guidance scale (7.5-15)
  - GPU/CPU support with automatic FP16/FP32 precision
  - Memory-efficient processing
  - Progress bars for diffusion steps
  - Seed control for reproducible results
  - Batch processing support

Inputs:
  - Original image: PIL Image or numpy array (RGB, any size)
  - Occlusion mask: numpy array (H, W) where 255 = inpaint, 0 = keep
  - Text prompt (optional): Description of what to generate in masked region

Outputs:
  - Inpainted image: PIL Image (same size as input)

Example usage:
    from face_inpainter import FaceInpainter
    
    inpainter = FaceInpainter(device='cuda')
    result = inpainter.inpaint(
        image=image,
        mask=mask,
        prompt="realistic thick black mustache, detailed",
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )

Author: Forensic Reconstruction System
Date: 2025
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import cv2

try:
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import load_image
except ImportError:
    raise ImportError(
        "diffusers library not installed. Install with: pip install diffusers transformers accelerate"
    )

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    tqdm = None

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceInpainter:
    """
    Face inpainting class using Stable Diffusion v1.5 inpainting model.
    
    This class handles loading the Stable Diffusion model, preprocessing
    images and masks, running inference with text prompts, and post-processing
    results. It supports both GPU and CPU execution with automatic precision
    selection (FP16 for GPU, FP32 for CPU).
    
    Attributes:
        pipeline: StableDiffusionInpaintPipeline instance
        device: Computing device (cuda or cpu)
        dtype: Data type (torch.float16 or torch.float32)
        model_loaded: Whether the model has been loaded
    """
    
    # Default prompt templates for different face features
    DEFAULT_PROMPTS = {
        'eyes': "highly detailed realistic eyes, natural lighting, professional photography",
        'eyebrows': "realistic eyebrows, natural hair texture, detailed",
        'mouth': "realistic mouth, natural lips, detailed",
        'mustache': "realistic thick black mustache, detailed facial hair",
        'nose': "realistic nose, natural skin texture, detailed",
        'hair': "realistic hair, natural texture, detailed",
        'upper_face': "highly detailed upper face features, realistic skin, professional",
        'lower_face': "highly detailed lower face features, realistic skin, professional",
        'default': "highly detailed facial feature, realistic, professional photography, natural lighting"
    }
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[str] = None,
        use_half_precision: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
    ):
        """
        Initialize the FaceInpainter.
        
        Args:
            model_name: Hugging Face model identifier for Stable Diffusion inpainting
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_half_precision: Use FP16 for faster inference (GPU only, recommended)
            cache_dir: Custom cache directory for model files
            enable_attention_slicing: Enable attention slicing to reduce memory (recommended)
            enable_vae_slicing: Enable VAE slicing to reduce memory (recommended)
        
        Raises:
            RuntimeError: If model loading fails
            ImportError: If required libraries are missing
        """
        logger.info("Initializing FaceInpainter...")
        start_time = time.time()
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Determine precision
        if use_half_precision and self.device.type == "cuda":
            self.dtype = torch.float16
            logger.info("Using half precision (FP16) for faster inference")
        else:
            self.dtype = torch.float32
            logger.info("Using full precision (FP32)")
        
        # Load model
        try:
            logger.info(f"Loading Stable Diffusion inpainting model: {model_name}")
            logger.info("  This may take a few minutes on first run (downloads ~7GB)")
            
            # Load pipeline
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if enable_attention_slicing:
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("  ✓ Attention slicing enabled (reduces memory)")
                except Exception as e:
                    logger.warning(f"  Could not enable attention slicing: {e}")
            
            if enable_vae_slicing:
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("  ✓ VAE slicing enabled (reduces memory)")
                except Exception as e:
                    logger.warning(f"  Could not enable VAE slicing: {e}")
            
            # Enable progress bar (will show during inference)
            self.pipeline.set_progress_bar_config(disable=False)
            
            self.model_loaded = True
            
            elapsed = time.time() - start_time
            logger.info(f"✓ FaceInpainter initialized in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            self.model_loaded = False
            raise RuntimeError(f"Model loading failed: {e}")
    
    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: np.ndarray,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        strength: float = 1.0,
        show_progress: bool = True,
    ) -> Image.Image:
        """
        Inpaint masked regions in an image using Stable Diffusion.
        
        Args:
            image: Input image (PIL Image or numpy array, RGB, any size)
            mask: Occlusion mask (numpy array, HxW, uint8 where 255 = inpaint, 0 = keep)
            prompt: Text description of what to generate (uses default if None)
            negative_prompt: What to avoid generating (e.g., "blurry, distorted")
            num_inference_steps: Number of diffusion steps (20-50, higher = better quality)
            guidance_scale: How closely to follow prompt (7.5-15, higher = more adherence)
            seed: Random seed for reproducibility (None = random)
            strength: How much to change the image (0.0-1.0, 1.0 = full inpainting)
            show_progress: Show progress bar during inference
        
        Returns:
            PIL Image of inpainted result (same size as input)
        
        Raises:
            RuntimeError: If model not loaded or inference fails
            ValueError: If inputs are invalid
            torch.cuda.OutOfMemoryError: If GPU runs out of memory (try CPU or reduce image size)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Initialize FaceInpainter first.")
        
        # Validate and preprocess inputs
        image_pil, mask_pil = self._preprocess_inputs(image, mask)
        
        # Use default prompt if not provided
        if prompt is None:
            prompt = self.DEFAULT_PROMPTS['default']
        
        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = "blurry, distorted, low quality, artifacts, deformed"
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Validate parameters
        num_inference_steps = max(1, min(50, int(num_inference_steps)))
        guidance_scale = max(1.0, min(20.0, float(guidance_scale)))
        strength = max(0.0, min(1.0, float(strength)))
        
        logger.info(f"Running inpainting...")
        logger.info(f"  Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"  Prompt: {prompt}")
        logger.info(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale:.1f}, Strength: {strength:.2f}")
        if seed is not None:
            logger.info(f"  Seed: {seed}")
        
        start_time = time.time()
        
        try:
            # Enable/disable progress bar
            self.pipeline.set_progress_bar_config(disable=not show_progress)
            
            # Run inference
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            ).images[0]
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Inpainting completed in {elapsed:.2f} seconds")
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = (
                "GPU out of memory. Try:\n"
                "  - Using CPU: FaceInpainter(device='cpu')\n"
                "  - Reducing image size\n"
                "  - Reducing num_inference_steps\n"
                "  - Enabling attention/VAE slicing (already enabled by default)"
            )
            logger.error(f"✗ {error_msg}")
            raise RuntimeError(f"Out of memory: {e}\n{error_msg}")
        
        except Exception as e:
            logger.error(f"✗ Inpainting failed: {e}")
            raise RuntimeError(f"Inpainting failed: {e}")
    
    def inpaint_batch(
        self,
        image: Union[Image.Image, np.ndarray],
        masks: Dict[str, np.ndarray],
        prompts: Optional[Dict[str, str]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Image.Image]:
        """
        Inpaint multiple masks on the same image (batch processing).
        
        Args:
            image: Input image (same for all masks)
            masks: Dictionary mapping mask name -> mask array
            prompts: Optional dictionary mapping mask name -> prompt (uses default if None)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            seed: Random seed (incremented for each mask for variety)
            show_progress: Show progress bars
        
        Returns:
            Dictionary mapping mask name -> inpainted image
        """
        results = {}
        prompts = prompts or {}
        
        logger.info(f"Processing batch of {len(masks)} masks...")
        
        current_seed = seed
        for i, (mask_name, mask) in enumerate(masks.items(), 1):
            logger.info(f"\n[{i}/{len(masks)}] Processing mask: {mask_name}")
            
            prompt = prompts.get(mask_name, None)
            
            result = self.inpaint(
                image=image,
                mask=mask,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                show_progress=show_progress,
            )
            
            results[mask_name] = result
            
            # Increment seed for next iteration (if provided)
            if current_seed is not None:
                current_seed += 1
        
        logger.info(f"\n✓ Batch processing completed: {len(results)} results")
        return results
    
    def _preprocess_inputs(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess image and mask for Stable Diffusion.
        
        Stable Diffusion expects:
        - Image: PIL Image, RGB, any size (will be resized internally to 512x512)
        - Mask: PIL Image, grayscale, same size as image, white (255) = inpaint
        
        Args:
            image: Input image (PIL or numpy)
            mask: Occlusion mask (numpy, HxW, uint8)
        
        Returns:
            Tuple of (image_pil, mask_pil)
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(image, mode='RGB')
            else:
                raise ValueError(f"Image must be RGB (H, W, 3), got shape {image.shape}")
        elif isinstance(image, Image.Image):
            image_pil = image.convert('RGB')
        else:
            raise ValueError(f"Image must be PIL Image or numpy array, got {type(image)}")
        
        # Validate and convert mask
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"Mask must be numpy array, got {type(mask)}")
        
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D (H, W), got shape {mask.shape}")
        
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            # Normalize to 0-255 if needed
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # Check size compatibility
        img_h, img_w = image_pil.size[1], image_pil.size[0]  # PIL uses (W, H)
        mask_h, mask_w = mask.shape
        
        if img_h != mask_h or img_w != mask_w:
            logger.warning(
                f"Image size ({img_w}x{img_h}) != mask size ({mask_w}x{mask_h}). "
                f"Resizing mask to match image."
            )
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to PIL Image (grayscale)
        mask_pil = Image.fromarray(mask, mode='L')
        
        return image_pil, mask_pil
    
    def get_default_prompt(self, feature_name: str) -> str:
        """
        Get default prompt for a face feature.
        
        Args:
            feature_name: Name of feature (e.g., 'eyes', 'mouth', 'mustache')
        
        Returns:
            Default prompt string
        """
        return self.DEFAULT_PROMPTS.get(feature_name.lower(), self.DEFAULT_PROMPTS['default'])


def create_inpainter(
    model_name: str = "runwayml/stable-diffusion-inpainting",
    device: Optional[str] = None,
    use_half_precision: bool = True,
) -> FaceInpainter:
    """
    Convenience function to create a FaceInpainter instance.
    
    Args:
        model_name: Stable Diffusion model identifier
        device: Device to use ('cuda', 'cpu', or None for auto)
        use_half_precision: Use FP16 for GPU (recommended)
    
    Returns:
        FaceInpainter instance
    """
    return FaceInpainter(
        model_name=model_name,
        device=device,
        use_half_precision=use_half_precision,
    )


# Example usage
if __name__ == "__main__":
    print("FaceInpainter module loaded successfully!")
    print("Use: inpainter = FaceInpainter()")
    print("     result = inpainter.inpaint(image, mask, prompt='realistic mustache')")
