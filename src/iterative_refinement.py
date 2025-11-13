"""
Iterative Face Refinement Engine
Allows users to interactively refine generated faces through feedback loops
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class IterativeRefinementEngine:
    """
    Manages iterative face refinement through user feedback.
    
    Workflow:
    1. Generate base face from description
    2. Show to user with refinement options
    3. User selects refinement (e.g., "make mustache thicker")
    4. System updates prompt and regenerates
    5. Compare old vs new
    6. User approves or requests more refinement
    """
    
    def __init__(self, generator, segmenter, landmark_detector, inpainter):
        """
        Initialize refinement engine with core components.
        
        Args:
            generator: TextToFaceGenerator instance
            segmenter: FaceSegmenter instance
            landmark_detector: LandmarkDetector instance
            inpainter: FaceInpainter instance
        """
        logger.info("Initializing IterativeRefinementEngine...")
        
        self.generator = generator
        self.segmenter = segmenter
        self.landmark_detector = landmark_detector
        self.inpainter = inpainter
        
        # Refinement history
        self.history = []
        self.current_face = None
        self.current_description = None
        self.refinement_count = 0
        
        # Refinement suggestions
        self._setup_refinement_options()
        
        logger.info("✓ IterativeRefinementEngine initialized")
    
    def _setup_refinement_options(self):
        """Setup available refinement options."""
        self.refinement_options = {
            'mustache': {
                'thicker': 'Add more density and volume to mustache',
                'thinner': 'Make mustache more subtle and thin',
                'darker': 'Darken mustache color',
                'lighter': 'Lighten mustache color',
                'different_style': 'Change to different mustache style (handlebar, chevron, etc.)',
                'remove': 'Remove mustache entirely'
            },
            'beard': {
                'thicker': 'Make beard fuller and denser',
                'thinner': 'Make beard more stubble-like',
                'longer': 'Grow beard longer',
                'shorter': 'Trim beard shorter',
                'different_style': 'Change beard style',
                'remove': 'Remove beard'
            },
            'eyes': {
                'larger': 'Make eyes larger and more prominent',
                'smaller': 'Make eyes smaller',
                'different_color': 'Change eye color',
                'more_detailed': 'Add more detail and intensity',
                'different_expression': 'Change eye expression (angry, kind, surprised)',
            },
            'hair': {
                'longer': 'Make hair longer',
                'shorter': 'Make hair shorter',
                'different_color': 'Change hair color',
                'different_style': 'Change hairstyle (curly, straight, wavy)',
                'more_volume': 'Add more volume to hair',
                'receding': 'Show receding hairline'
            },
            'skin': {
                'darker': 'Darken skin tone',
                'lighter': 'Lighten skin tone',
                'smoother': 'Make skin smoother and clearer',
                'more_detail': 'Add texture and imperfections',
                'add_scars': 'Add visible scars/marks',
                'add_wrinkles': 'Add age lines/wrinkles'
            },
            'face_shape': {
                'wider': 'Make face wider',
                'narrower': 'Make face narrower',
                'rounder': 'Make face rounder',
                'squarer': 'Make face square-shaped',
                'longer': 'Make face longer',
                'younger': 'Make face look younger',
                'older': 'Make face look older'
            },
            'nose': {
                'larger': 'Make nose larger and more prominent',
                'smaller': 'Make nose smaller',
                'broader': 'Make nose broader',
                'narrower': 'Make nose narrower',
                'more_detail': 'Add more nostril/tip definition'
            },
            'mouth': {
                'larger': 'Make mouth larger',
                'smaller': 'Make mouth smaller',
                'fuller_lips': 'Make lips fuller',
                'thinner_lips': 'Make lips thinner',
                'different_expression': 'Change mouth expression (smiling, frowning, neutral)',
            },
            'overall': {
                'more_realistic': 'Improve photorealism and detail',
                'better_lighting': 'Improve lighting and shadows',
                'higher_quality': 'Higher overall quality and resolution',
                'professional_look': 'Make look more professional/formal',
                'casual_look': 'Make look more casual/relaxed'
            }
        }
    
    def start_refinement_session(self, description: str, initial_prompt: Optional[str] = None,
                                 num_inference_steps: int = 30) -> Dict:
        """
        Start a new refinement session.
        
        Args:
            description: Base witness description
            initial_prompt: Optional custom prompt (if None, uses description)
            num_inference_steps: Steps for generation
            
        Returns:
            Dictionary with generated face and available refinements
        """
        logger.info(f"\n{'='*60}")
        logger.info("Starting Iterative Refinement Session")
        logger.info('='*60)
        logger.info(f"Description: {description}")
        
        self.current_description = description
        self.refinement_count = 0
        self.history = []
        
        # Generate initial face
        prompt = initial_prompt or description
        logger.info(f"Generating initial face with prompt: {prompt[:60]}...")
        
        base_face = self.generator.generate(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            seed=42
        )
        
        # Save initial face
        output_dir = Path('output/iterative_refinement')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        face_path = output_dir / 'iteration_000_base.png'
        if isinstance(base_face, Image.Image):
            base_face.save(str(face_path))
        else:
            Image.fromarray(base_face).save(str(face_path))
        
        self.current_face = base_face
        
        # Record in history
        self.history.append({
            'iteration': 0,
            'prompt': prompt,
            'image_path': str(face_path),
            'refinement_type': 'initial',
            'user_feedback': None
        })
        
        result = {
            'iteration': 0,
            'base_face_path': str(face_path),
            'current_prompt': prompt,
            'description': description,
            'available_refinements': self._get_refinement_suggestions(),
            'history': self.history
        }
        
        logger.info(f"✓ Initial face generated and saved: {face_path}")
        logger.info(f"\nAvailable refinement categories: {list(self.refinement_options.keys())}")
        
        return result
    
    def refine_feature(self, feature_category: str, refinement_type: str,
                      intensity: float = 1.0,
                      num_inference_steps: int = 30) -> Dict:
        """
        Apply a specific refinement to the current face.
        
        Args:
            feature_category: Category (mustache, eyes, hair, skin, etc.)
            refinement_type: Type of refinement within category
            intensity: Intensity of refinement (0.0-2.0, 1.0 = normal)
            num_inference_steps: Diffusion steps
            
        Returns:
            Dictionary with refined face and comparison
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Refinement {self.refinement_count + 1}")
        logger.info('='*60)
        
        # Validate refinement
        if feature_category not in self.refinement_options:
            logger.error(f"Unknown category: {feature_category}")
            return {'error': f'Unknown category: {feature_category}'}
        
        if refinement_type not in self.refinement_options[feature_category]:
            logger.error(f"Unknown refinement: {refinement_type}")
            return {'error': f'Unknown refinement for {feature_category}'}
        
        # Get refinement instruction
        instruction = self.refinement_options[feature_category][refinement_type]
        
        # Build updated prompt
        updated_prompt = self._build_refined_prompt(
            self.current_description,
            feature_category,
            refinement_type,
            intensity
        )
        
        logger.info(f"Feature: {feature_category}")
        logger.info(f"Refinement: {refinement_type}")
        logger.info(f"Intensity: {intensity:.1f}x")
        logger.info(f"Updated prompt: {updated_prompt[:80]}...")
        
        # Generate refined face
        refined_face = self.generator.generate(
            updated_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            seed=42 + self.refinement_count + 1  # Different seed
        )
        
        # Save refined face
        output_dir = Path('output/iterative_refinement')
        self.refinement_count += 1
        
        iteration_num = len(self.history)
        face_path = output_dir / f'iteration_{iteration_num:03d}_{feature_category}_{refinement_type}.png'
        
        if isinstance(refined_face, Image.Image):
            refined_face.save(str(face_path))
        else:
            Image.fromarray(refined_face).save(str(face_path))
        
        # Create comparison
        comparison_path = self._create_comparison(
            self.current_face,
            refined_face,
            f"{feature_category}: {refinement_type}",
            iteration_num
        )
        
        # Update current face
        self.current_face = refined_face
        
        # Record in history
        self.history.append({
            'iteration': iteration_num,
            'prompt': updated_prompt,
            'image_path': str(face_path),
            'comparison_path': str(comparison_path),
            'refinement_type': refinement_type,
            'feature_category': feature_category,
            'intensity': intensity,
            'instruction': instruction
        })
        
        logger.info(f"✓ Refined face saved: {face_path}")
        logger.info(f"✓ Comparison saved: {comparison_path}")
        
        return {
            'iteration': iteration_num,
            'refined_face_path': str(face_path),
            'comparison_path': str(comparison_path),
            'prompt': updated_prompt,
            'refinement_info': {
                'category': feature_category,
                'type': refinement_type,
                'instruction': instruction,
                'intensity': intensity
            },
            'history': self.history,
            'available_refinements': self._get_refinement_suggestions()
        }
    
    def batch_refine(self, refinements: List[Dict]) -> Dict:
        """
        Apply multiple refinements in sequence.
        
        Args:
            refinements: List of refinement dicts with keys:
                        {category, type, intensity (optional)}
                        
        Returns:
            Final result after all refinements
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Refinement: {len(refinements)} steps")
        logger.info('='*60)
        
        results = []
        for i, refinement in enumerate(refinements, 1):
            logger.info(f"\n[{i}/{len(refinements)}] Applying {refinement['category']}:{refinement['type']}")
            
            result = self.refine_feature(
                feature_category=refinement['category'],
                refinement_type=refinement['type'],
                intensity=refinement.get('intensity', 1.0)
            )
            
            results.append(result)
        
        logger.info(f"\n✓ Batch refinement complete ({len(refinements)} steps)")
        
        return {
            'total_refinements': len(refinements),
            'steps': results,
            'final_face_path': results[-1]['refined_face_path'] if results else None,
            'history': self.history
        }
    
    def _build_refined_prompt(self, base_description: str,
                             feature_category: str,
                             refinement_type: str,
                             intensity: float) -> str:
        """Build enhanced prompt based on refinement."""
        
        instruction = self.refinement_options[feature_category][refinement_type]
        
        # Intensity modifiers
        if intensity > 1.2:
            intensity_str = "very, extremely"
        elif intensity > 1.0:
            intensity_str = "more"
        elif intensity < 0.8:
            intensity_str = "slightly"
        else:
            intensity_str = ""
        
        # Build detailed prompt
        refined_prompt = (
            f"{base_description}. "
            f"Focus on {feature_category}: {intensity_str} {instruction}. "
            f"Highly detailed, photorealistic, professional photography, studio lighting, "
            f"high quality, sharp focus, cinematic"
        )
        
        return refined_prompt.strip()
    
    def _create_comparison(self, original, refined, title: str, iteration: int) -> Path:
        """Create side-by-side comparison of original vs refined."""
        
        # Convert to PIL if needed
        if isinstance(original, np.ndarray):
            original = Image.fromarray(original)
        if isinstance(refined, np.ndarray):
            refined = Image.fromarray(refined)
        
        # Resize to same size
        size = (256, 256)
        original_resized = original.resize(size)
        refined_resized = refined.resize(size)
        
        # Create side-by-side image
        comparison = Image.new('RGB', (size[0] * 2 + 20, size[1] + 60), color=(255, 255, 255))
        comparison.paste(original_resized, (10, 40))
        comparison.paste(refined_resized, (size[0] + 10, 40))
        
        # Add labels (basic, no font needed)
        output_dir = Path('output/iterative_refinement')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = output_dir / f'comparison_{iteration:03d}.png'
        comparison.save(str(comparison_path))
        
        return comparison_path
    
    def _get_refinement_suggestions(self) -> Dict:
        """Get list of available refinements with descriptions."""
        suggestions = {}
        for category, options in self.refinement_options.items():
            suggestions[category] = {
                type_: description
                for type_, description in options.items()
            }
        return suggestions
    
    def save_session(self, session_name: str) -> Path:
        """Save refinement session to file."""
        
        output_dir = Path('output/iterative_refinement')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = output_dir / f'{session_name}_session.json'
        
        session_data = {
            'name': session_name,
            'description': self.current_description,
            'total_refinements': len(self.history),
            'history': self.history
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"✓ Session saved: {session_file}")
        
        return session_file
    
    def get_session_summary(self) -> Dict:
        """Get summary of current refinement session."""
        
        return {
            'description': self.current_description,
            'total_iterations': len(self.history),
            'current_iteration': len(self.history) - 1,
            'refinement_steps': [
                {
                    'iteration': h['iteration'],
                    'type': h.get('refinement_type', 'initial'),
                    'category': h.get('feature_category', 'base'),
                    'image_path': h['image_path']
                }
                for h in self.history
            ],
            'current_face_path': str(Path('output/iterative_refinement') / f'iteration_{len(self.history)-1:03d}.png')
        }
