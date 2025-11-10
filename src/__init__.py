"""
Forensic Facial Reconstruction System - Source Package

This package contains the core modules for facial segmentation,
landmark detection, inpainting, and reconstruction.
"""

from .face_segmentation import FaceSegmenter, create_segmenter
from .landmark_detector import LandmarkDetector, create_detector

__all__ = [
    'FaceSegmenter', 
    'create_segmenter',
    'LandmarkDetector',
    'create_detector'
]

__version__ = '1.0.0'

