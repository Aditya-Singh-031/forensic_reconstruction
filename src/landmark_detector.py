"""
Facial Landmark Detection Module using MediaPipe Face Mesh

This module provides 468 3D facial landmark detection using MediaPipe Face Mesh.
It identifies key facial features like eyes, eyebrows, nose, mouth, ears, and jawline.

Author: Forensic Reconstruction System
Date: 2025
"""

import time
import logging
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings

import numpy as np
import cv2
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "MediaPipe not installed. Install with: pip install mediapipe"
    )

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkDetector:
    """
    Facial landmark detection class using MediaPipe Face Mesh.
    
    Detects 468 3D facial landmarks and identifies key feature groups:
    - Eyes (left, right)
    - Eyebrows (left, right)
    - Nose (tip, base)
    - Mouth (corners, center)
    - Ears (left, right)
    - Face contour/jawline
    
    Attributes:
        face_mesh: MediaPipe Face Mesh solution
        drawing_utils: MediaPipe drawing utilities
        mp_face_mesh: MediaPipe face mesh module
    """
    
    # MediaPipe Face Mesh landmark indices for key features
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    
    # Face contour (jawline) - outer boundary
    FACE_CONTOUR = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    # Left eye
    LEFT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    
    # Right eye
    RIGHT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]
    
    # Left eyebrow
    LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    
    # Right eyebrow
    RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    
    # Nose
    NOSE_TIP = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360, 279, 358, 327, 326, 2, 97, 240, 64, 98, 327]
    NOSE_BASE = [2, 97, 98, 327]
    
    # Mouth (outer)
    MOUTH_OUTER = [
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
    ]
    
    # Mouth (inner)
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
    
    # Left ear
    LEFT_EAR = [234, 127, 162, 21, 54, 103, 67, 109]
    
    # Right ear
    RIGHT_EAR = [454, 323, 361, 288, 397, 365, 379, 378]
    
    # Face oval (for general face region)
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the LandmarkDetector.
        
        Args:
            static_image_mode: If True, treats input as static images (not video)
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: If True, refines landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection [0.0, 1.0]
            min_tracking_confidence: Minimum confidence for landmark tracking [0.0, 1.0]
        
        Raises:
            RuntimeError: If MediaPipe initialization fails
        """
        logger.info("Initializing LandmarkDetector...")
        start_time = time.time()
        
        try:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            # Store configuration
            self.max_num_faces = max_num_faces
            self.refine_landmarks = refine_landmarks
            
            elapsed = time.time() - start_time
            logger.info(f"✓ LandmarkDetector initialized in {elapsed:.2f} seconds")
            logger.info(f"  - Max faces: {max_num_faces}")
            logger.info(f"  - Refined landmarks: {refine_landmarks}")
            logger.info(f"  - 468 landmarks per face")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            raise RuntimeError(f"MediaPipe initialization failed: {e}")
    
    def detect(
        self,
        image_path: Union[str, Path],
        return_visualization: bool = True,
        return_groups: bool = True,
        return_coordinates: bool = True
    ) -> Dict:
        """
        Detect facial landmarks in an image.
        
        Args:
            image_path: Path to input image (JPG, PNG, etc.)
            return_visualization: Whether to return annotated image
            return_groups: Whether to return landmark groups (eyes, nose, etc.)
            return_coordinates: Whether to return raw coordinates
        
        Returns:
            Dictionary containing:
                - 'landmarks': List of all 468 landmarks (x, y, z) in pixel coordinates
                - 'landmarks_normalized': List of normalized coordinates [0.0, 1.0]
                - 'groups': Dictionary of feature groups (eyes, nose, etc.)
                - 'visualization': Annotated image (PIL Image, if return_visualization=True)
                - 'num_faces': Number of faces detected
                - 'face_index': Index of selected face (if multiple detected)
                - 'processing_time': Time taken in seconds
                - 'image_size': Original image size (width, height)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded or no face detected
        """
        start_time = time.time()
        
        # Load and validate image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            image_size = image.size  # (width, height)
            logger.info(f"Loaded image: {image_path.name} ({image_size[0]}x{image_size[1]})")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Process with MediaPipe
        logger.info("Running landmark detection...")
        results = self.face_mesh.process(image_array)
        
        # Check if faces detected
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in image. Try a different image or lower min_detection_confidence.")
        
        num_faces = len(results.multi_face_landmarks)
        logger.info(f"Detected {num_faces} face(s)")
        
        # Select largest face if multiple detected
        face_index = self._select_largest_face(results.multi_face_landmarks, image_size)
        face_landmarks = results.multi_face_landmarks[face_index]
        
        if num_faces > 1:
            logger.info(f"Selected face {face_index + 1} (largest)")
        
        # Extract landmarks
        landmarks_normalized = []
        landmarks_pixel = []
        
        for landmark in face_landmarks.landmark:
            # Normalized coordinates [0.0, 1.0]
            x_norm = landmark.x
            y_norm = landmark.y
            z_norm = landmark.z
            
            # Convert to pixel coordinates
            x_pixel = int(x_norm * image_size[0])
            y_pixel = int(y_norm * image_size[1])
            # Z is depth, approximate pixel scale (MediaPipe uses relative depth)
            z_pixel = z_norm * image_size[0]  # Scale z relative to image width
            
            landmarks_normalized.append([x_norm, y_norm, z_norm])
            landmarks_pixel.append([x_pixel, y_pixel, z_pixel])
        
        # Build result dictionary
        result = {
            'landmarks_normalized': np.array(landmarks_normalized),
            'landmarks': np.array(landmarks_pixel),
            'num_faces': num_faces,
            'face_index': face_index,
            'processing_time': time.time() - start_time,
            'image_size': image_size
        }
        
        # Extract feature groups
        if return_groups:
            result['groups'] = self._extract_feature_groups(
                landmarks_pixel,
                landmarks_normalized
            )
        
        # Create visualization
        if return_visualization:
            result['visualization'] = self._create_visualization(
                image_array,
                face_landmarks,
                result.get('groups', {})
            )
        
        # Add coordinates if requested
        if return_coordinates:
            result['coordinates'] = {
                'pixel': result['landmarks'],
                'normalized': result['landmarks_normalized']
            }
        
        logger.info(f"✓ Landmark detection completed in {result['processing_time']:.2f} seconds")
        logger.info(f"  - Landmarks detected: {len(landmarks_pixel)}")
        
        return result
    
    def _select_largest_face(
        self,
        face_landmarks_list: List,
        image_size: Tuple[int, int]
    ) -> int:
        """
        Select the largest face from multiple detected faces.
        
        Args:
            face_landmarks_list: List of MediaPipe face landmark objects
            image_size: Image size (width, height)
        
        Returns:
            Index of the largest face
        """
        if len(face_landmarks_list) == 1:
            return 0
        
        # Calculate bounding box area for each face
        face_areas = []
        for face_landmarks in face_landmarks_list:
            # Get min/max coordinates
            x_coords = [lm.x * image_size[0] for lm in face_landmarks.landmark]
            y_coords = [lm.y * image_size[1] for lm in face_landmarks.landmark]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            area = width * height
            
            face_areas.append(area)
        
        # Return index of face with largest area
        return int(np.argmax(face_areas))
    
    def _extract_feature_groups(
        self,
        landmarks_pixel: List[List[float]],
        landmarks_normalized: List[List[float]]
    ) -> Dict[str, Dict]:
        """
        Extract landmark groups for key facial features.
        
        Args:
            landmarks_pixel: List of pixel coordinates
            landmarks_normalized: List of normalized coordinates
        
        Returns:
            Dictionary mapping feature names to their landmarks
        """
        groups = {}
        
        # Define feature groups with their landmark indices
        feature_definitions = {
            'left_eye': self.LEFT_EYE,
            'right_eye': self.RIGHT_EYE,
            'left_eyebrow': self.LEFT_EYEBROW,
            'right_eyebrow': self.RIGHT_EYEBROW,
            'nose_tip': self.NOSE_TIP[:10],  # Top 10 nose tip points
            'nose_base': self.NOSE_BASE,
            'mouth_outer': self.MOUTH_OUTER,
            'mouth_inner': self.MOUTH_INNER,
            'left_ear': self.LEFT_EAR,
            'right_ear': self.RIGHT_EAR,
            'face_contour': self.FACE_CONTOUR,
            'jawline': self.FACE_CONTOUR,  # Alias
        }
        
        for feature_name, indices in feature_definitions.items():
            # Get landmarks for this feature
            pixel_coords = np.array([landmarks_pixel[i] for i in indices if i < len(landmarks_pixel)])
            norm_coords = np.array([landmarks_normalized[i] for i in indices if i < len(landmarks_normalized)])
            
            # Calculate bounding box
            if len(pixel_coords) > 0:
                bbox = {
                    'x_min': int(np.min(pixel_coords[:, 0])),
                    'y_min': int(np.min(pixel_coords[:, 1])),
                    'x_max': int(np.max(pixel_coords[:, 0])),
                    'y_max': int(np.max(pixel_coords[:, 1])),
                    'width': int(np.max(pixel_coords[:, 0]) - np.min(pixel_coords[:, 0])),
                    'height': int(np.max(pixel_coords[:, 1]) - np.min(pixel_coords[:, 1]))
                }
                
                # Calculate center
                center = {
                    'x': int(np.mean(pixel_coords[:, 0])),
                    'y': int(np.mean(pixel_coords[:, 1]))
                }
            else:
                bbox = None
                center = None
            
            groups[feature_name] = {
                'indices': indices,
                'landmarks_pixel': pixel_coords.tolist(),
                'landmarks_normalized': norm_coords.tolist(),
                'bbox': bbox,
                'center': center,
                'count': len(pixel_coords)
            }
        
        return groups
    
    def _create_visualization(
        self,
        image_array: np.ndarray,
        face_landmarks,
        groups: Dict
    ) -> Image.Image:
        """
        Create visualization with landmarks and feature groups highlighted.
        
        Args:
            image_array: Original image as numpy array
            face_landmarks: MediaPipe face landmarks object
            groups: Dictionary of feature groups
        
        Returns:
            PIL Image with annotations
        """
        # Create a copy for drawing
        annotated_image = image_array.copy()
        
        # Draw face mesh (all landmarks)
        self.mp_drawing.draw_landmarks(
            annotated_image,
            face_landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None,
            self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Draw feature groups with different colors
        # Convert to BGR for OpenCV drawing
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Color scheme for different features
        colors = {
            'left_eye': (255, 0, 0),      # Blue
            'right_eye': (255, 0, 0),      # Blue
            'left_eyebrow': (0, 255, 0),   # Green
            'right_eyebrow': (0, 255, 0),  # Green
            'nose_tip': (0, 0, 255),       # Red
            'nose_base': (0, 0, 255),      # Red
            'mouth_outer': (255, 0, 255),  # Magenta
            'mouth_inner': (255, 0, 255),  # Magenta
            'left_ear': (255, 255, 0),      # Cyan
            'right_ear': (255, 255, 0),     # Cyan
            'face_contour': (0, 255, 255), # Yellow
        }
        
        # Draw bounding boxes and centers for each group
        for feature_name, group_data in groups.items():
            if group_data['bbox'] is None:
                continue
            
            bbox = group_data['bbox']
            center = group_data['center']
            color = colors.get(feature_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(
                annotated_bgr,
                (bbox['x_min'], bbox['y_min']),
                (bbox['x_max'], bbox['y_max']),
                color,
                2
            )
            
            # Draw center point
            if center:
                cv2.circle(annotated_bgr, (center['x'], center['y']), 5, color, -1)
            
            # Draw label
            label = feature_name.replace('_', ' ').title()
            cv2.putText(
                annotated_bgr,
                label,
                (bbox['x_min'], bbox['y_min'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Convert back to RGB
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(annotated_rgb)
    
    def get_landmark_coordinates(
        self,
        result: Dict,
        format: str = 'numpy'
    ) -> Union[np.ndarray, List, str]:
        """
        Get landmark coordinates in various formats.
        
        Args:
            result: Result dictionary from detect() method
            format: Output format ('numpy', 'list', 'csv')
        
        Returns:
            Landmarks in requested format
        """
        landmarks = result['landmarks']
        
        if format == 'numpy':
            return landmarks
        elif format == 'list':
            return landmarks.tolist()
        elif format == 'csv':
            import io
            output = io.StringIO()
            np.savetxt(output, landmarks, delimiter=',', fmt='%.2f')
            return output.getvalue()
        else:
            raise ValueError(f"Unknown format: {format}. Use 'numpy', 'list', or 'csv'.")


def create_detector(
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True
) -> LandmarkDetector:
    """
    Convenience function to create a LandmarkDetector instance.
    
    Args:
        static_image_mode: Treat input as static images
        max_num_faces: Maximum number of faces to detect
        refine_landmarks: Refine landmarks around eyes and lips
    
    Returns:
        LandmarkDetector instance
    """
    return LandmarkDetector(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks
    )


# Example usage
if __name__ == "__main__":
    # Example: Basic usage
    detector = create_detector()
    
    # Detect landmarks (replace with your image path)
    # result = detector.detect("path/to/face/image.jpg")
    # print(f"Detected {len(result['landmarks'])} landmarks")
    # result['visualization'].save("output_landmarks.png")
    
    print("LandmarkDetector module loaded successfully!")
    print("Use: detector = LandmarkDetector()")
    print("     result = detector.detect('path/to/image.jpg')")

