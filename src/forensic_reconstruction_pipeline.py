"""
Complete Forensic Facial Reconstruction Pipeline
Orchestrates all modules into one end-to-end workflow
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import argparse
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.description_parser import ForensicDescriptionParser
from src.text_to_face import TextToFaceGenerator
from src.face_database import FaceDatabase
from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector
from src.mask_generator import MaskGenerator
from src.face_inpainter import FaceInpainter


class ForensicReconstructionPipeline:
    """Complete end-to-end forensic reconstruction system."""
    
    def __init__(self):
        """Initialize all components."""
        logger.info("="*60)
        logger.info("FORENSIC FACIAL RECONSTRUCTION PIPELINE")
        logger.info("="*60)
        
        logger.info("\nInitializing components...")
        
        self.parser = ForensicDescriptionParser()
        logger.info("  ✓ Description Parser")
        
        self.generator = TextToFaceGenerator()
        logger.info("  ✓ Text-to-Face Generator")
        
        self.database = FaceDatabase()
        logger.info("  ✓ Face Database")
        
        self.segmenter = FaceSegmenter()
        logger.info("  ✓ Face Segmenter")
        
        self.landmark_detector = LandmarkDetector()
        logger.info("  ✓ Landmark Detector")
        
        self.inpainter = FaceInpainter()
        logger.info("  ✓ Face Inpainter")
        
        logger.info("\n✓ All components initialized!")
    
    def process_witness_description(self, description: str, 
                                    num_faces: int = 2,
                                    inpaint_features: Optional[list] = None) -> Dict:
        """
        Complete pipeline: description → face generation → database storage → refinement.
        
        Args:
            description: Witness description text
            num_faces: Number of faces to generate
            inpaint_features: Optional list of features to refine (eyes, mouth, etc.)
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("PROCESSING WITNESS DESCRIPTION")
        logger.info("="*60)
        
        results = {
            'description': description,
            'parsed_attributes': None,
            'generated_faces': [],
            'database_records': [],
            'refined_faces': []
        }
        
        # STEP 1: Parse description
        logger.info("\n[1/5] Parsing description...")
        parsed = self.parser.parse(description)
        results['parsed_attributes'] = parsed['attributes']
        logger.info(f"  ✓ Confidence: {parsed['overall_confidence']:.2f}")
        
        # STEP 2: Generate faces
        logger.info(f"\n[2/5] Generating {num_faces} face(s)...")
        generated_images = []
        for i in range(num_faces):
            logger.info(f"  Generating face {i+1}/{num_faces}...")
            img = self.generator.generate(
                description,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42 + i  # Different seed for each face
            )
            generated_images.append(img)
            logger.info(f"    ✓ Face {i+1} generated")

        results['generated_faces'] = generated_images
        logger.info(f"  ✓ Generated {len(generated_images)} faces")


        results['generated_faces'] = generated_images
        logger.info(f"  ✓ Generated {len(generated_images)} faces")
        
        # STEP 3: Add to database
        logger.info("\n[3/5] Adding to database...")
        for i, img_path in enumerate(generated_images, 1):
            record_id = self.database.add_record(
                description,
                img_path,
                parsed['attributes'],
                parsed['overall_confidence']
            )
            results['database_records'].append(record_id)
            logger.info(f"  ✓ {record_id}")
        
        # STEP 4: Refine features (optional)
        if inpaint_features:
            logger.info(f"\n[4/5] Refining features: {inpaint_features}...")
            
            for i, img_path in enumerate(generated_images, 1):
                logger.info(f"  Image {i}/{len(generated_images)}:")
                
                for feature in inpaint_features:
                    # Segment
                    seg_result = self.segmenter.segment(str(img_path))
                    
                    # Detect landmarks
                    lm_result = self.landmark_detector.detect(str(img_path))
                    
                    # Generate mask
                    mg = MaskGenerator(
                        segmentation=seg_result['segmentation'],
                        component_ids=self.segmenter.face_component_ids,
                        landmarks=lm_result.get('groups')
                    )
                    mask = mg.generate([feature], margin_px=5, feather_px=2)
                    
                    # Inpaint
                    refined = self.inpainter.inpaint(
                        str(img_path),
                        mask,
                        prompt=f"high quality {feature}, realistic, detailed"
                    )
                    
                    results['refined_faces'].append({
                        'original': img_path,
                        'feature': feature,
                        'refined': refined
                    })
                    logger.info(f"    ✓ Refined {feature}")
        else:
            logger.info("\n[4/5] Feature refinement skipped")
        
        # STEP 5: Search database
        logger.info("\n[5/5] Searching database for similar faces...")
        matches = self.database.search_by_attributes(
            parsed['attributes'],
            threshold=0.5
        )
        logger.info(f"  ✓ Found {len(matches)} matching records")
        
        # Add top matches to results
        results['database_matches'] = [
            {
                'record_id': record_id,
                'similarity': score,
                'record': self.database.get_record(record_id)
            }
            for record_id, score in matches[:5]
        ]
        
        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("="*60)
        
        return results
    
    def display_results(self, results: Dict):
        """Display pipeline results in readable format."""
        
        print("\n" + "="*60)
        print("FORENSIC RECONSTRUCTION RESULTS")
        print("="*60)
        
        print(f"\nWitness Description:\n  {results['description']}")
        
        print(f"\nParsed Attributes:")
        for attr_name, attr_data in results['parsed_attributes'].items():
            if attr_data['value'] is not None:
                print(f"  {attr_name}: {attr_data['value']}")
        
        print(f"\nGenerated Faces: {len(results['generated_faces'])}")
        for i, face_path in enumerate(results['generated_faces'], 1):
            print(f"  [{i}] {face_path}")
        
        print(f"\nDatabase Records Added:")
        for record_id in results['database_records']:
            print(f"  ✓ {record_id}")
        
        if results.get('refined_faces'):
            print(f"\nRefined Features: {len(results['refined_faces'])}")
            for refined in results['refined_faces']:
                print(f"  {refined['feature']}: {refined['refined']}")
        
        print(f"\nDatabase Matches Found: {len(results['database_matches'])}")
        for i, match in enumerate(results['database_matches'], 1):
            record = match['record']
            score = match['similarity']
            print(f"  [{i}] {match['record_id']} (Similarity: {score:.2f})")
            print(f"      {record['description'][:60]}...")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Forensic Facial Reconstruction Pipeline'
    )
    parser.add_argument('--description', type=str, required=True,
                       help='Witness description')
    parser.add_argument('--num_faces', type=int, default=2,
                       help='Number of faces to generate')
    parser.add_argument('--refine', action='store_true',
                       help='Refine generated faces (eyes, mouth)')
    parser.add_argument('--output', type=str, default='output/pipeline_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ForensicReconstructionPipeline()
    
    # Process description
    features_to_refine = ['eyes', 'mouth'] if args.refine else None
    
    results = pipeline.process_witness_description(
        args.description,
        num_faces=args.num_faces,
        inpaint_features=features_to_refine
    )
    
    # Display results
    pipeline.display_results(results)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'pipeline_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
