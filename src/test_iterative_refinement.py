"""
Test Iterative Face Refinement Engine
Interactive refinement demonstration
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.iterative_refinement import IterativeRefinementEngine
from src.text_to_face import TextToFaceGenerator
from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector
from src.face_inpainter import FaceInpainter


def demo_single_refinement():
    """Demo: Single refinement step."""
    
    print("\n" + "="*60)
    print("DEMO 1: Single Refinement")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    generator = TextToFaceGenerator()
    segmenter = FaceSegmenter()
    landmark_detector = LandmarkDetector()
    inpainter = FaceInpainter()
    
    # Create refinement engine
    refinement_engine = IterativeRefinementEngine(
        generator, segmenter, landmark_detector, inpainter
    )
    
    # Start session
    description = "Adult male, 40 years old, Indian, dark complexion"
    result = refinement_engine.start_refinement_session(description)
    
    print(f"\nBase face generated: {result['base_face_path']}")
    print(f"Current prompt: {result['current_prompt']}")
    
    # Apply first refinement
    print("\n" + "-"*60)
    print("Applying refinement: Make mustache thicker")
    print("-"*60)
    
    refined_result = refinement_engine.refine_feature(
        feature_category='mustache',
        refinement_type='thicker',
        intensity=1.2
    )
    
    print(f"\n✓ Refined face: {refined_result['refined_face_path']}")
    print(f"✓ Comparison: {refined_result['comparison_path']}")
    print(f"\nRefinement info:")
    print(f"  Category: {refined_result['refinement_info']['category']}")
    print(f"  Type: {refined_result['refinement_info']['type']}")
    print(f"  Instruction: {refined_result['refinement_info']['instruction']}")
    
    # Save session
    refinement_engine.save_session('demo_single_refinement')
    
    print("\n✓ Demo 1 complete!")


def demo_batch_refinement():
    """Demo: Multiple refinements in sequence."""
    
    print("\n" + "="*60)
    print("DEMO 2: Batch Refinement (Multiple Steps)")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    generator = TextToFaceGenerator()
    segmenter = FaceSegmenter()
    landmark_detector = LandmarkDetector()
    inpainter = FaceInpainter()
    
    # Create refinement engine
    refinement_engine = IterativeRefinementEngine(
        generator, segmenter, landmark_detector, inpainter
    )
    
    # Start session
    description = "Adult female, 25-30, fair skin, long black hair, Indian"
    result = refinement_engine.start_refinement_session(description)
    
    print(f"\nBase face generated: {result['base_face_path']}")
    
    # Define batch refinements
    refinements = [
        {'category': 'hair', 'type': 'longer', 'intensity': 1.3},
        {'category': 'eyes', 'type': 'larger', 'intensity': 1.1},
        {'category': 'skin', 'type': 'smoother', 'intensity': 1.0},
    ]
    
    print(f"\nApplying {len(refinements)} refinements in sequence...")
    
    batch_result = refinement_engine.batch_refine(refinements)
    
    print(f"\n✓ Batch refinement complete!")
    print(f"Total steps: {batch_result['total_refinements']}")
    print(f"Final face: {batch_result['final_face_path']}")
    
    # Show history
    print(f"\nRefinement history:")
    for i, step in enumerate(batch_result['steps'], 1):
        info = step['refinement_info']
        print(f"  [{i}] {info['category']}: {info['type']} (intensity: {info['intensity']:.1f}x)")
    
    # Save session
    refinement_engine.save_session('demo_batch_refinement')
    
    print("\n✓ Demo 2 complete!")


def demo_interactive_menu():
    """Demo: Interactive refinement menu."""
    
    print("\n" + "="*60)
    print("DEMO 3: Interactive Refinement Menu")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    generator = TextToFaceGenerator()
    segmenter = FaceSegmenter()
    landmark_detector = LandmarkDetector()
    inpainter = FaceInpainter()
    
    # Create refinement engine
    refinement_engine = IterativeRefinementEngine(
        generator, segmenter, landmark_detector, inpainter
    )
    
    # Start session
    description = "Adult male, 35-40, medium build, serious expression"
    result = refinement_engine.start_refinement_session(description)
    
    print(f"\nBase face generated: {result['base_face_path']}")
    print(f"\nAvailable refinement categories:")
    
    categories = list(result['available_refinements'].keys())
    for i, cat in enumerate(categories, 1):
        print(f"  [{i}] {cat.upper()}")
    
    print(f"\nAvailable refinements per category:")
    for cat, refinements in list(result['available_refinements'].items())[:3]:
        print(f"\n  {cat.upper()}:")
        for ref_type, description in list(refinements.items())[:3]:
            print(f"    - {ref_type}: {description}")
        if len(refinements) > 3:
            print(f"    ... and {len(refinements) - 3} more")
    
    # Simulate user selecting refinements
    print("\n" + "-"*60)
    print("Simulating user selections...")
    print("-"*60)
    
    user_selections = [
        {'category': 'facial_hair', 'type': 'beard', 'msg': 'Add thick beard'},
        {'category': 'face_shape', 'type': 'squarer', 'msg': 'Make face squarer'},
    ]
    
    # Apply first selection (if beard exists, use beard; else use available)
    try:
        refined1 = refinement_engine.refine_feature(
            feature_category='beard',
            refinement_type='thicker',
            intensity=1.2
        )
        print(f"\n✓ Applied: Make beard thicker")
        print(f"  Refined: {refined1['refined_face_path']}")
    except:
        print("Beard option not available, trying alternative...")
        refined1 = refinement_engine.refine_feature(
            feature_category='overall',
            refinement_type='more_realistic',
            intensity=1.1
        )
        print(f"✓ Applied: Improve realism")
    
    # Apply second selection
    try:
        refined2 = refinement_engine.refine_feature(
            feature_category='face_shape',
            refinement_type='squarer',
            intensity=1.1
        )
        print(f"\n✓ Applied: Make face squarer")
        print(f"  Refined: {refined2['refined_face_path']}")
    except:
        print("Face shape option not available, trying alternative...")
        refined2 = refinement_engine.refine_feature(
            feature_category='overall',
            refinement_type='higher_quality',
            intensity=1.0
        )
        print(f"✓ Applied: Improve quality")
    
    # Show session summary
    print("\n" + "-"*60)
    print("Session Summary")
    print("-"*60)
    
    summary = refinement_engine.get_session_summary()
    print(f"\nDescription: {summary['description']}")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"Refinement steps:")
    
    for step in summary['refinement_steps']:
        print(f"  [{step['iteration']}] {step['category']}: {step['type']}")
    
    # Save session
    refinement_engine.save_session('demo_interactive_menu')
    
    print("\n✓ Demo 3 complete!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ITERATIVE REFINEMENT ENGINE - DEMONSTRATION")
    print("="*60)
    
    print("\nAvailable demos:")
    print("  1. Single refinement step")
    print("  2. Batch refinement (multiple steps)")
    print("  3. Interactive refinement menu")
    print("  4. Run all demos")
    
    choice = input("\nSelect demo (1-4) [default: 4]: ").strip() or '4'
    
    if choice == '1':
        demo_single_refinement()
    elif choice == '2':
        demo_batch_refinement()
    elif choice == '3':
        demo_interactive_menu()
    elif choice == '4':
        demo_single_refinement()
        demo_batch_refinement()
        demo_interactive_menu()
    else:
        print("Invalid choice")
