"""
Debug script to check what LandmarkDetector.detect() returns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
from src.landmark_detector import LandmarkDetector

# Initialize detector
detector = LandmarkDetector()

# Test image
test_image = "/DATA/facial_features_dataset/raw_images/ffhq/00000.png"

print("\n" + "="*60)
print("DEBUGGING LANDMARK DETECTOR OUTPUT")
print("="*60)

try:
    result = detector.detect(test_image, return_visualization=True, return_groups=True)
    
    print(f"\n✓ Detection successful!")
    print(f"\nResult keys: {result.keys()}")
    
    # Check groups
    if 'groups' in result:
        groups = result['groups']
        print(f"\nGroups detected: {len(groups)}")
        
        for group_name, group_data in groups.items():
            print(f"\n  {group_name}:")
            print(f"    - Keys: {group_data.keys()}")
            print(f"    - Count: {group_data.get('count', 0)}")
            
            if 'bbox' in group_data:
                bbox = group_data['bbox']
                if bbox:
                    print(f"    - BBox: {bbox}")
                else:
                    print(f"    - BBox: None")
            
            if 'landmarks_pixel' in group_data:
                lms = group_data['landmarks_pixel']
                print(f"    - Landmarks: {len(lms)} points")
                if len(lms) > 0:
                    print(f"      First point: {lms[0]}")
    
    print(f"\nLandmarks shape: {np.array(result.get('landmarks', [])).shape}")
    print(f"Num faces: {result.get('num_faces')}")
    print(f"Processing time: {result.get('processing_time'):.3f}s")
    
except Exception as e:
    print(f"\n✗ Detection failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
