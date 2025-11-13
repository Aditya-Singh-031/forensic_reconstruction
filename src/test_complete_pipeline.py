"""
Test Complete Forensic Reconstruction Pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forensic_reconstruction_pipeline import ForensicReconstructionPipeline


def main():
    # Initialize pipeline
    pipeline = ForensicReconstructionPipeline()
    
    # Test descriptions
    test_descriptions = [
        "Adult male, 40-45 years old, thick black mustache, large ears, Indian, dark complexion",
        "Young woman, 25-30, fair skin, long black hair, glasses, Indian"
    ]
    
    for desc in test_descriptions:
        print(f"\n{'='*60}")
        print(f"Processing: {desc}")
        print('='*60)
        
        results = pipeline.process_witness_description(
            description=desc,
            num_faces=1,
            inpaint_features=None  # Set to ['eyes', 'mouth'] to refine
        )
        
        pipeline.display_results(results)


if __name__ == '__main__':
    main()
