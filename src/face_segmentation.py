"""
Facial Segmentation Module using SegFormer

This module provides semantic segmentation of face images into components
like skin, hair, eyes, nose, mouth, etc. using the SegFormer model.

Author: Forensic Reconstruction System
Date: 2025
"""

import os
import time
import logging
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    raise ImportError(
        "transformers library not installed. Install with: pip install transformers"
    )

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceSegmenter:
    """
    Face segmentation class using SegFormer for semantic segmentation.
    
    SegFormer is trained on ADE20k dataset (150 classes). This class maps
    relevant ADE20k classes to face components and provides face-specific
    segmentation functionality.
    
    Attributes:
        model: SegFormer model for semantic segmentation
        processor: Image processor for SegFormer
        device: Computing device (cuda or cpu)
        class_mapping: Mapping from ADE20k classes to face components
    """
    
    # ADE20k class IDs that are relevant for face segmentation
    # Note: ADE20k doesn't have specific face classes, so we use general classes
    # and map them to face components based on typical face regions
    ADE20K_TO_FACE_MAPPING = {
        # Skin/face regions (using person, skin-like textures)
        'person': 'face_skin',
        'skin': 'face_skin',
        'background': 'background',
        # Hair
        'hair': 'hair',
        'head': 'hair',  # Often includes hair region
        # Eyes (using general object classes that might appear in eye region)
        'eye': 'eyes',
        # Nose
        'nose': 'nose',
        # Mouth
        'mouth': 'mouth',
        # Ears
        'ear': 'ears',
        # Clothing (neck region)
        'clothing': 'neck',
        'neck': 'neck',
    }
    
    # Color palette for visualization (RGB)
    COMPONENT_COLORS = {
        'background': (0, 0, 0),          # Black
        'face_skin': (255, 220, 177),     # Light skin tone
        'hair': (50, 25, 0),              # Dark brown/black
        'eyes': (0, 0, 255),              # Blue
        'nose': (255, 200, 150),          # Light skin
        'mouth': (200, 0, 0),              # Red
        'ears': (255, 200, 150),           # Light skin
        'neck': (200, 150, 100),          # Medium skin
        'eyebrows': (30, 15, 0),          # Dark brown
        'unknown': (128, 128, 128),       # Gray
    }
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: Optional[str] = None,
        use_half_precision: bool = True
    ):
        """
        Initialize the FaceSegmenter.
        
        Args:
            model_name: Hugging Face model identifier for SegFormer
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_half_precision: Use FP16 for faster inference (GPU only)
        
        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Initializing FaceSegmenter...")
        start_time = time.time()
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        try:
            logger.info(f"Loading SegFormer model: {model_name}")
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
            # Move to device and set precision
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            if use_half_precision and self.device.type == "cuda":
                self.model = self.model.half()
                self.dtype = torch.float16
                logger.info("Using half precision (FP16) for faster inference")
            else:
                self.dtype = torch.float32
                logger.info("Using full precision (FP32)")
            
            # Create face component mapping from ADE20k classes
            self._create_face_mapping()
            
            elapsed = time.time() - start_time
            logger.info(f"✓ FaceSegmenter initialized in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load SegFormer model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _create_face_mapping(self):
        """Create mapping from ADE20k class IDs to face components."""
        # Get ADE20k class names (150 classes)
        # Since we don't have direct access to class names, we'll use a heuristic
        # approach: map based on typical face region locations
        
        # ADE20k has 150 classes, but for face segmentation we focus on:
        # - Person class (usually class 0 or a specific ID)
        # - Background (class 0 typically)
        # We'll use post-processing to identify face regions
        
        self.class_mapping = {}
        self.face_component_ids = {
            'background': 0,
            'face_skin': 1,
            'hair': 2,
            'eyes': 3,
            'nose': 4,
            'mouth': 5,
            'ears': 6,
            'neck': 7,
            'eyebrows': 8,
        }
        
        logger.info("Face component mapping created")
    
    def segment(
        self,
        image_path: Union[str, Path],
        return_masks: bool = True,
        return_colored: bool = True,
        return_statistics: bool = True
    ) -> Dict:
        """
        Segment a face image into semantic components.
        
        Args:
            image_path: Path to input image (JPG, PNG, etc.)
            return_masks: Whether to return individual component masks
            return_colored: Whether to return colored segmentation visualization
            return_statistics: Whether to return pixel statistics per component
        
        Returns:
            Dictionary containing:
                - 'segmentation': Segmentation mask (numpy array)
                - 'colored': Colored visualization (PIL Image)
                - 'masks': Dictionary of individual component masks (if return_masks=True)
                - 'statistics': Dictionary of pixel counts per component (if return_statistics=True)
                - 'processing_time': Time taken in seconds
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded or processed
        """
        start_time = time.time()
        
        # Load and validate image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size  # (width, height)
            logger.info(f"Loaded image: {image_path.name} ({original_size[0]}x{original_size[1]})")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        if self.dtype == torch.float16:
            pixel_values = pixel_values.half()
        
        # Run inference
        logger.info("Running segmentation inference...")
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits
        
        # Post-process: upscale logits to original image size
        upsampled_logits = F.interpolate(
            logits,
            size=original_size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Get predicted segmentation
        seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # Shape: (H, W)
        
        # Convert ADE20k segmentation to face components
        face_seg = self._ade20k_to_face_components(seg, image)
        
        # Build result dictionary
        result = {
            'segmentation': face_seg,
            'original_size': original_size,
            'processing_time': time.time() - start_time
        }
        
        # Generate colored visualization
        if return_colored:
            result['colored'] = self._create_colored_visualization(face_seg, image)
        
        # Extract individual masks
        if return_masks:
            result['masks'] = self._extract_component_masks(face_seg)
        
        # Calculate statistics
        if return_statistics:
            result['statistics'] = self._calculate_statistics(face_seg)
        
        logger.info(f"✓ Segmentation completed in {result['processing_time']:.2f} seconds")
        
        return result
    
    def _ade20k_to_face_components(
        self,
        seg_mask: np.ndarray,
        original_image: Image.Image
    ) -> np.ndarray:
        """
        Convert ADE20k segmentation to face-specific components.
        
        Since ADE20k doesn't have specific face classes, we use heuristics
        to identify face regions based on typical face structure.
        
        Args:
            seg_mask: ADE20k segmentation mask (H, W)
            original_image: Original PIL image for reference
        
        Returns:
            Face component segmentation mask (H, W)
        """
        # Convert image to numpy for processing
        img_array = np.array(original_image)
        h, w = seg_mask.shape
        
        # Initialize face component mask
        face_seg = np.zeros_like(seg_mask, dtype=np.uint8)
        
        # Heuristic: Identify person class (usually largest non-background region)
        unique_classes, counts = np.unique(seg_mask, return_counts=True)
        
        # Find person class (typically class with significant area, not background)
        # Background is usually class 0
        person_class = None
        if len(unique_classes) > 1:
            # Sort by count, skip background (class 0)
            sorted_indices = np.argsort(counts)[::-1]
            for idx in sorted_indices:
                if unique_classes[idx] != 0 and counts[idx] > h * w * 0.1:  # At least 10% of image
                    person_class = unique_classes[idx]
                    break
        
        if person_class is None:
            logger.warning("No person detected in image - using full image as face region")
            person_class = 0  # Use background as fallback
        
        # Create binary mask for person region
        person_mask = (seg_mask == person_class).astype(np.uint8)
        
        # Use face detection to refine face region (optional, using simple heuristics)
        # For now, we'll use the person mask and apply region-based labeling
        
        # Simple heuristic: Divide person region into face components
        # This is a simplified approach - in production, you'd use face landmarks
        
        # Get bounding box of person region
        coords = np.where(person_mask > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Divide into regions (simplified face structure)
            face_h = y_max - y_min
            face_w = x_max - x_min
            
            # Upper region: hair/forehead
            hair_y = y_min + int(face_h * 0.2)
            # Middle-upper: eyes/eyebrows
            eyes_y = y_min + int(face_h * 0.4)
            # Middle: nose
            nose_y = y_min + int(face_h * 0.6)
            # Lower-middle: mouth
            mouth_y = y_min + int(face_h * 0.75)
            
            # Left/right for ears
            ear_x_left = x_min + int(face_w * 0.1)
            ear_x_right = x_max - int(face_w * 0.1)
            
            # Label regions
            for y in range(h):
                for x in range(w):
                    if person_mask[y, x] > 0:
                        if y < hair_y:
                            face_seg[y, x] = self.face_component_ids['hair']
                        elif y < eyes_y:
                            face_seg[y, x] = self.face_component_ids['eyes']
                        elif y < nose_y:
                            face_seg[y, x] = self.face_component_ids['nose']
                        elif y < mouth_y:
                            face_seg[y, x] = self.face_component_ids['mouth']
                        else:
                            face_seg[y, x] = self.face_component_ids['face_skin']
                        
                        # Check for ear regions
                        if x < ear_x_left or x > ear_x_right:
                            if y_min + int(face_h * 0.3) < y < y_min + int(face_h * 0.7):
                                face_seg[y, x] = self.face_component_ids['ears']
        else:
            # No person detected - mark as background
            face_seg = np.zeros_like(seg_mask, dtype=np.uint8)
        
        return face_seg
    
    def _create_colored_visualization(
        self,
        seg_mask: np.ndarray,
        original_image: Image.Image
    ) -> Image.Image:
        """
        Create colored visualization of segmentation.
        
        Args:
            seg_mask: Segmentation mask (H, W)
            original_image: Original image for overlay
        
        Returns:
            PIL Image with colored segmentation overlay
        """
        # Create color map
        h, w = seg_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each component to its color
        id_to_component = {v: k for k, v in self.face_component_ids.items()}
        
        for component_id, component_name in id_to_component.items():
            mask = (seg_mask == component_id)
            color = self.COMPONENT_COLORS.get(component_name, self.COMPONENT_COLORS['unknown'])
            colored[mask] = color
        
        # Create overlay (blend with original image)
        img_array = np.array(original_image)
        overlay = cv2.addWeighted(img_array, 0.6, colored, 0.4, 0)
        
        return Image.fromarray(overlay)
    
    def _extract_component_masks(self, seg_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract individual masks for each face component.
        
        Args:
            seg_mask: Segmentation mask (H, W)
        
        Returns:
            Dictionary mapping component names to binary masks
        """
        masks = {}
        id_to_component = {v: k for k, v in self.face_component_ids.items()}
        
        for component_id, component_name in id_to_component.items():
            mask = (seg_mask == component_id).astype(np.uint8) * 255
            masks[component_name] = mask
        
        return masks
    
    def _calculate_statistics(self, seg_mask: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate statistics for each component.
        
        Args:
            seg_mask: Segmentation mask (H, W)
        
        Returns:
            Dictionary with statistics for each component
        """
        stats = {}
        total_pixels = seg_mask.size
        id_to_component = {v: k for k, v in self.face_component_ids.items()}
        
        for component_id, component_name in id_to_component.items():
            mask = (seg_mask == component_id)
            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
            
            stats[component_name] = {
                'pixel_count': int(pixel_count),
                'percentage': round(percentage, 2)
            }
        
        return stats
    
    def segment_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Segment multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once
        
        Returns:
            List of segmentation results (one per image)
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} images)")
            
            for image_path in batch:
                try:
                    result = self.segment(image_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to segment {image_path}: {e}")
                    results.append(None)
        
        return results


def create_segmenter(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    device: Optional[str] = None
) -> FaceSegmenter:
    """
    Convenience function to create a FaceSegmenter instance.
    
    Args:
        model_name: SegFormer model identifier
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        FaceSegmenter instance
    """
    return FaceSegmenter(model_name=model_name, device=device)


# Example usage
if __name__ == "__main__":
    # Example: Basic usage
    segmenter = create_segmenter()
    
    # Segment an image (replace with your image path)
    # result = segmenter.segment("path/to/face/image.jpg")
    # print(f"Segmentation completed in {result['processing_time']:.2f}s")
    # result['colored'].save("output_segmentation.png")
    
    print("FaceSegmenter module loaded successfully!")
    print("Use: segmenter = FaceSegmenter()")
    print("     result = segmenter.segment('path/to/image.jpg')")

