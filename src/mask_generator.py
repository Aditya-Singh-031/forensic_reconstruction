"""
Occlusion Mask Generator

This module provides a `MaskGenerator` class that uses facial segmentation masks
and optionally facial landmarks to generate binary occlusion masks for specific
face components. These masks are intended for use in occlusion/inpainting
experiments and data augmentation.

Features supported (individually or in combination):
  - eyes (left, right, or both)
  - eyebrows (left, right, or both)
  - mouth
  - mustache/below nose (area between nose base and upper lip)
  - nose
  - hair (top portion)
  - full upper face / lower face

Conventions:
  - Output mask is single-channel uint8 where 255 indicates "occlude/inpaint"
    and 0 indicates "keep".
  - Overlapping requested regions are unioned (logical OR).
  - Optional feathering smooths edges of the mask for anti-aliasing.

Inputs:
  - segmentation: numpy array (H, W) of face component IDs produced by
    `FaceSegmenter` in `src/face_segmentation.py`. Expected component IDs are
    available via `FaceSegmenter.face_component_ids`.
  - landmarks (optional): dictionary produced by `LandmarkDetector.detect(...)`
    in `src/landmark_detector.py`, specifically the `groups` field for feature
    polygons and bounding boxes.

Example CLI usage:
  python -m src.test_masking --image path/to/img.jpg --output_dir output/masks
  python -m src.test_masking --batch images/ --output_dir output/masks

Author: Forensic Reconstruction System
Date: 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegionSpec:
    """
    Configuration for a single occlusion region.
    
    Attributes:
        name: Logical name of the feature (e.g., 'left_eye', 'mouth', 'hair_top').
        margin_px: Extra pixel padding to dilate the region.
        side: Optionally specify a side for bilateral features: 'left' or 'right'.
    """
    name: str
    margin_px: int = 4
    side: Optional[str] = None  # 'left' | 'right' | None


def _safe_polygon_mask(shape: Tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """
    Create a filled polygon mask for given points.
    
    Args:
        shape: (H, W) output mask shape.
        pts: Nx2 array of integer pixel coordinates (x, y).
    
    Returns:
        Binary uint8 mask with 1s inside the polygon.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if pts is None or len(pts) == 0:
        return mask
    pts_i32 = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts_i32], 1)
    return mask


def _dilate(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Dilate binary mask by margin_px using an elliptical kernel."""
    if margin_px <= 0:
        return mask
    k = 2 * margin_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def _feather(mask255: np.ndarray, feather_px: int) -> np.ndarray:
    """
    Apply Gaussian blur to soften edges. Keeps dtype uint8, range [0, 255].
    """
    if feather_px <= 0:
        return mask255
    k = max(1, feather_px | 1)  # ensure odd kernel
    blurred = cv2.GaussianBlur(mask255, (k, k), sigmaX=0, sigmaY=0)
    return blurred


class MaskGenerator:
    """
    Generate occlusion masks for facial components from segmentation and
    optional landmarks.
    
    Typical usage:
        mg = MaskGenerator(segmentation=seg, component_ids=face_component_ids, landmarks=groups)
        mask_mouth = mg.generate(['mouth'], margin_px=6, feather_px=3)
        mask_eyes = mg.generate([RegionSpec('eyes', side='left'), RegionSpec('eyes', side='right')])
    
    Notes:
        - If landmarks are provided, precise polygons are used where possible.
        - If landmarks are not provided, segmentation components and geometric
          heuristics are used.
        - When both are available, landmarks take precedence for higher accuracy.
    """

    def __init__(
        self,
        segmentation: np.ndarray,
        component_ids: Dict[str, int],
        landmarks: Optional[Dict[str, Dict]] = None,
    ):
        """
        Args:
            segmentation: (H, W) integer mask of component IDs.
            component_ids: Mapping of component names to IDs (from FaceSegmenter.face_component_ids).
            landmarks: Optional landmark groups dict (from LandmarkDetector.detect()['groups']).
        
        Raises:
            ValueError: On invalid shapes or missing required components.
        """
        if segmentation is None or segmentation.ndim != 2:
            raise ValueError("segmentation must be a 2D array (H, W)")
        self.segmentation = segmentation
        self.h, self.w = segmentation.shape
        self.component_ids = component_ids or {}
        self.landmarks = landmarks or {}

        # Precompute binary masks for known segmentation components (0/1)
        self._component_binary: Dict[str, np.ndarray] = {}
        for name, cid in self.component_ids.items():
            self._component_binary[name] = (self.segmentation == cid).astype(np.uint8)

        # Derive a generic face mask if present, else non-background union
        if 'face_skin' in self._component_binary:
            self.face_mask = self._component_binary['face_skin'].copy()
        else:
            # best-effort: any non-zero assumed to relate to face region in our pipeline
            self.face_mask = (self.segmentation > 0).astype(np.uint8)

    # ------------------------ Public API ------------------------
    def generate(
        self,
        regions: Union[Iterable[Union[str, RegionSpec]], str],
        margin_px: int = 4,
        feather_px: int = 0,
        hair_top_ratio: float = 0.5,
        return_binary: bool = True,
    ) -> np.ndarray:
        """
        Generate an occlusion mask for one or more regions.
        
        Args:
            regions: A string ('mouth') or iterable of strings/RegionSpec.
            margin_px: Default dilation margin applied to each region.
            feather_px: Apply feathering (Gaussian blur kernel ~ px). 0 disables.
            hair_top_ratio: For 'hair' top portion, keep top ratio (0-1].
            return_binary: If True, threshold to {0,255}; else keep softened 0-255.
        
        Returns:
            Single-channel uint8 mask (H, W) with 255 marking occlusion.
        """
        if isinstance(regions, (str, bytes)):
            regions = [regions]  # type: ignore

        accum = np.zeros((self.h, self.w), dtype=np.uint8)

        for r in regions:
            if isinstance(r, RegionSpec):
                name = r.name
                side = r.side
                pad = r.margin_px if r.margin_px is not None else margin_px
            else:
                name = str(r)
                side = None
                pad = margin_px

            mask01 = self._region_mask01(name=name, side=side, margin_px=pad, hair_top_ratio=hair_top_ratio)
            accum = np.clip(accum + (mask01 * 255).astype(np.uint8), 0, 255)

        if feather_px > 0:
            accum = _feather(accum, feather_px)
            if return_binary:
                # Threshold to binary after feathering to keep edges slightly expanded
                _, accum = cv2.threshold(accum, 32, 255, cv2.THRESH_BINARY)
        else:
            # Ensure binary
            accum = (accum > 0).astype(np.uint8) * 255

        return accum

    def generate_batch(
        self,
        batch_specs: Dict[str, List[Union[str, RegionSpec]]],
        margin_px: int = 4,
        feather_px: int = 0,
        hair_top_ratio: float = 0.5,
        return_binary: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate multiple masks at once for automated pipelines.
        
        Args:
            batch_specs: Mapping from mask name to region spec list.
        
        Returns:
            Dict of mask name -> uint8 mask.
        """
        out: Dict[str, np.ndarray] = {}
        for key, spec in batch_specs.items():
            out[key] = self.generate(
                regions=spec,
                margin_px=margin_px,
                feather_px=feather_px,
                hair_top_ratio=hair_top_ratio,
                return_binary=return_binary,
            )
        return out

    # ------------------------ Region builders ------------------------
    def _region_mask01(
        self,
        name: str,
        side: Optional[str],
        margin_px: int,
        hair_top_ratio: float,
    ) -> np.ndarray:
        name_l = name.lower()
        if name_l in ('eyes', 'eye'):
            # both eyes or side-specific
            if side in ('left', 'right'):
                m = self._mask_eye(side=side, margin_px=margin_px)
            else:
                m = np.clip(self._mask_eye('left', margin_px) + self._mask_eye('right', margin_px), 0, 1)
            return m
        if name_l in ('eyebrows', 'eyebrow'):
            if side in ('left', 'right'):
                m = self._mask_eyebrow(side=side, margin_px=margin_px)
            else:
                m = np.clip(self._mask_eyebrow('left', margin_px) + self._mask_eyebrow('right', margin_px), 0, 1)
            return m
        if name_l == 'mouth':
            return self._mask_mouth(margin_px=margin_px)
        if name_l in ('mustache', 'below_nose', 'mustache_below_nose'):
            return self._mask_mustache(margin_px=margin_px)
        if name_l == 'nose':
            return self._mask_nose(margin_px=margin_px)
        if name_l in ('hair', 'hair_top'):
            return self._mask_hair_top(margin_px=margin_px, top_ratio=hair_top_ratio)
        if name_l in ('upper_face', 'upper'):
            return self._mask_upper_face(margin_px=margin_px)
        if name_l in ('lower_face', 'lower'):
            return self._mask_lower_face(margin_px=margin_px)

        raise ValueError(f"Unknown region: '{name}'. Supported: eyes, eyebrows, mouth, mustache, nose, hair, upper_face, lower_face")

    # ---- Concrete region implementations (return 0/1 masks) ----
    def _mask_from_landmark_group(self, group_name: str, margin_px: int) -> np.ndarray:
        """
        Build mask from a landmarks group polygon (convex hull). Falls back to
        segmentation if group not found or empty.
        """
        group = self.landmarks.get(group_name)
        if group and group.get('landmarks_pixel'):
            try:
                pts = np.array(group['landmarks_pixel'], dtype=np.float32)
                # Validate: non-empty
                if pts.size == 0 or len(pts) < 3:
                    logger.warning(f"[MaskGenerator] Landmark group '{group_name}' has insufficient points: shape={pts.shape}")
                    raise ValueError("insufficient_points")
                # Remove NaN/Inf rows
                if not np.isfinite(pts).all():
                    mask_finite = np.all(np.isfinite(pts), axis=1)
                    pts = pts[mask_finite]
                    logger.warning(f"[MaskGenerator] Removed non-finite points for '{group_name}', remaining={len(pts)}")
                    if len(pts) < 3:
                        raise ValueError("non_finite_after_filter")
                # Ensure shape suitable for convexHull: (N,1,2) float32
                if pts.ndim != 2 or pts.shape[1] < 2:
                    logger.warning(f"[MaskGenerator] Invalid pts shape for '{group_name}': {pts.shape}")
                    raise ValueError("invalid_shape")
                pts2 = pts[:, :2].astype(np.float32).reshape(-1, 1, 2)
                # Debug (optional): uncomment for verbose values
                # logger.debug(f"[MaskGenerator] '{group_name}' pts2 shape={pts2.shape}, sample={pts2[:5].reshape(-1,2)}")
                hull = cv2.convexHull(pts2)
                hull2 = hull.reshape(-1, 2)  # (M,2)
                mask = _safe_polygon_mask((self.h, self.w), hull2)
                mask = _dilate(mask, margin_px)
                return (mask > 0).astype(np.uint8)
            except Exception as e:
                logger.warning(f"[MaskGenerator] convexHull failed for '{group_name}': {e}. Falling back to segmentation bbox.")
        # Fallback: try segmentation component with a reasonable name mapping
        seg_name = self._landmark_to_seg_component(group_name)
        if seg_name in self._component_binary:
            seg = self._component_binary[seg_name]
            # If segmentation exists, try bbox-based polygon when hull failed
            if np.any(seg):
                ys, xs = np.where(seg > 0)
                x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
                rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                mask = _safe_polygon_mask((self.h, self.w), rect)
                return _dilate(mask, margin_px)
            return _dilate(seg, margin_px)
        return np.zeros((self.h, self.w), dtype=np.uint8)

    @staticmethod
    def _landmark_to_seg_component(group_name: str) -> str:
        mapping = {
            'left_eye': 'eyes',
            'right_eye': 'eyes',
            'mouth_outer': 'mouth',
            'mouth_inner': 'mouth',
            'nose_tip': 'nose',
            'nose_base': 'nose',
            'left_eyebrow': 'eyebrows',
            'right_eyebrow': 'eyebrows',
        }
        return mapping.get(group_name, group_name)

    def _mask_eye(self, side: str, margin_px: int) -> np.ndarray:
        side = side.lower()
        lm_group = 'left_eye' if side == 'left' else 'right_eye'
        m = self._mask_from_landmark_group(lm_group, margin_px)
        if np.any(m):
            return (m > 0).astype(np.uint8)
        # Fallback using segmentation 'eyes' split by centerline
        eyes = self._component_binary.get('eyes', np.zeros_like(self.face_mask))
        if not np.any(eyes):
            return np.zeros_like(self.face_mask)
        mid_x = self.w // 2
        if side == 'left':
            sel = np.zeros_like(eyes); sel[:, :mid_x] = 1
        else:
            sel = np.zeros_like(eyes); sel[:, mid_x:] = 1
        m = (eyes & sel).astype(np.uint8)
        return _dilate(m, margin_px)

    def _mask_eyebrow(self, side: str, margin_px: int) -> np.ndarray:
        side = side.lower()
        lm_group = 'left_eyebrow' if side == 'left' else 'right_eyebrow'
        m = self._mask_from_landmark_group(lm_group, margin_px)
        if np.any(m):
            return (m > 0).astype(np.uint8)
        # Approximate from eye region by shifting upward
        eye_m = self._mask_eye(side=side, margin_px=0)
        if not np.any(eye_m):
            # fallback to segmentation 'eyebrows' if exists
            brow_seg = self._component_binary.get('eyebrows', np.zeros_like(self.face_mask))
            return _dilate(brow_seg, margin_px)
        # Move eye mask up by ~10% of image height
        shift = max(2, int(0.05 * self.h))
        M = np.float32([[1, 0, 0], [0, 1, -shift]])
        shifted = cv2.warpAffine(eye_m, M, (self.w, self.h), flags=cv2.INTER_NEAREST, borderValue=0)
        return _dilate(shifted, margin_px + 2)

    def _mask_mouth(self, margin_px: int) -> np.ndarray:
        m = self._mask_from_landmark_group('mouth_outer', margin_px)
        if np.any(m):
            return (m > 0).astype(np.uint8)
        seg = self._component_binary.get('mouth', np.zeros_like(self.face_mask))
        return _dilate(seg, margin_px)

    def _mask_nose(self, margin_px: int) -> np.ndarray:
        # Prefer nose_tip hull, fallback to segmentation nose
        m_tip = self._mask_from_landmark_group('nose_tip', margin_px)
        if np.any(m_tip):
            return (m_tip > 0).astype(np.uint8)
        seg = self._component_binary.get('nose', np.zeros_like(self.face_mask))
        return _dilate(seg, margin_px)

    def _mask_mustache(self, margin_px: int) -> np.ndarray:
        """
        Region between nose base and upper lip. Uses landmarks if available,
        else estimates from nose/mouth segmentation bounding boxes.
        """
        # Landmarks-based
        nose_base = self.landmarks.get('nose_base', {})
        mouth_outer = self.landmarks.get('mouth_outer', {})
        if nose_base.get('bbox') and mouth_outer.get('bbox'):
            nb = nose_base['bbox']; mb = mouth_outer['bbox']
            y_top = int(nb['y_max'])
            y_bottom = int(mb['y_min'])
            if y_bottom > y_top:
                x_left = min(nb['x_min'], mb['x_min'])
                x_right = max(nb['x_max'], mb['x_max'])
                rect = np.array([
                    [x_left, y_top],
                    [x_right, y_top],
                    [x_right, y_bottom],
                    [x_left, y_bottom],
                ], dtype=np.float32)
                m = _safe_polygon_mask((self.h, self.w), rect)
                return _dilate(m, margin_px)
        # Fallback: use seg bboxes
        nose = self._component_binary.get('nose', np.zeros_like(self.face_mask))
        mouth = self._component_binary.get('mouth', np.zeros_like(self.face_mask))
        if np.any(nose) and np.any(mouth):
            y_idxs, x_idxs = np.where(nose > 0)
            nb = (x_idxs.min(), y_idxs.min(), x_idxs.max(), y_idxs.max())
            y_idxs, x_idxs = np.where(mouth > 0)
            mb = (x_idxs.min(), y_idxs.min(), x_idxs.max(), y_idxs.max())
            y_top = nb[3]; y_bottom = mb[1]
            if y_bottom > y_top:
                x_left = min(nb[0], mb[0]); x_right = max(nb[2], mb[2])
                rect = np.array([[x_left, y_top], [x_right, y_top], [x_right, y_bottom], [x_left, y_bottom]], dtype=np.float32)
                m = _safe_polygon_mask((self.h, self.w), rect)
                return _dilate(m, margin_px)
        return np.zeros((self.h, self.w), dtype=np.uint8)

    def _mask_hair_top(self, margin_px: int, top_ratio: float) -> np.ndarray:
        """
        Top portion of hair region. top_ratio in (0,1], e.g., 0.5 keeps upper half.
        """
        hair = self._component_binary.get('hair', np.zeros_like(self.face_mask))
        if not np.any(hair):
            return np.zeros_like(hair)
        y_idxs, x_idxs = np.where(hair > 0)
        y_min, y_max = int(y_idxs.min()), int(y_idxs.max())
        cutoff = int(y_min + (y_max - y_min) * max(0.0, min(1.0, top_ratio)))
        top_mask = np.zeros_like(hair)
        top_mask[y_min:cutoff + 1, :] = 1
        m = (hair & top_mask).astype(np.uint8)
        return _dilate(m, margin_px)

    def _mask_upper_face(self, margin_px: int) -> np.ndarray:
        """
        Upper face: above a midline guided by eyes or nose.
        """
        # Use eyes center if available
        y_ref: Optional[int] = None
        for key in ('left_eye', 'right_eye'):
            g = self.landmarks.get(key)
            if g and g.get('bbox'):
                if y_ref is None:
                    y_ref = g['bbox']['y_max']
                else:
                    y_ref = max(y_ref, g['bbox']['y_max'])
        if y_ref is None:
            # fallback: nose or 50% of face bbox
            nose = self.landmarks.get('nose_tip')
            if nose and nose.get('bbox'):
                y_ref = nose['bbox']['y_min']
        if y_ref is None:
            ys, xs = np.where(self.face_mask > 0)
            if len(ys) == 0:
                return np.zeros_like(self.face_mask)
            y_min, y_max = ys.min(), ys.max()
            y_ref = int((y_min + y_max) * 0.55)
        up = np.zeros_like(self.face_mask)
        up[:max(0, y_ref), :] = 1
        up = (up & self.face_mask).astype(np.uint8)
        return _dilate(up, margin_px)

    def _mask_lower_face(self, margin_px: int) -> np.ndarray:
        """
        Lower face: below a midline guided by mouth or nose.
        """
        y_ref: Optional[int] = None
        mouth = self.landmarks.get('mouth_outer')
        if mouth and mouth.get('bbox'):
            y_ref = mouth['bbox']['y_min']
        if y_ref is None:
            nose = self.landmarks.get('nose_tip')
            if nose and nose.get('bbox'):
                y_ref = int(nose['bbox']['y_max'])
        if y_ref is None:
            ys, xs = np.where(self.face_mask > 0)
            if len(ys) == 0:
                return np.zeros_like(self.face_mask)
            y_min, y_max = ys.min(), ys.max()
            y_ref = int((y_min + y_max) * 0.5)
        low = np.zeros_like(self.face_mask)
        low[min(self.h - 1, y_ref):, :] = 1
        low = (low & self.face_mask).astype(np.uint8)
        return _dilate(low, margin_px)


# Convenience mapping for common presets
PRESETS: Dict[str, List[Union[str, RegionSpec]]] = {
    'eyes_both': [RegionSpec('eyes', side='left'), RegionSpec('eyes', side='right')],
    'eyebrows_both': [RegionSpec('eyebrows', side='left'), RegionSpec('eyebrows', side='right')],
    'mouth': ['mouth'],
    'mustache': ['mustache'],
    'nose': ['nose'],
    'hair_top': ['hair'],
    'upper_face': ['upper_face'],
    'lower_face': ['lower_face'],
}


