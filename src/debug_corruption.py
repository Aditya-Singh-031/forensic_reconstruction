"""
Debug Corruption - Shows exactly what files are loaded
"""

import json
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from corruption_engine import CorruptionEngine

def debug_single_image(dataset_dir: str, image_name: str):
    """Debug what files are being loaded for a specific image."""
    
    # Load feature index
    index_path = Path(dataset_dir) / "metadata" / "feature_index.json"
    
    with open(index_path) as f:
        feature_index = json.load(f)
    
    if image_name not in feature_index:
        print(f"❌ Image {image_name} not in index!")
        print(f"Available: {list(feature_index.keys())[:10]}")
        return
    
    data = feature_index[image_name]
    
    print("\n" + "="*80)
    print(f"DEBUG: Image {image_name}")
    print("="*80)
    
    # Show paths
    print("\n📁 FILE PATHS:")
    for feature_type in ['face_contour', 'jawline']:
        if feature_type in data['features']:
            path = data['features'][feature_type]
            print(f"  {feature_type:15} → {path}")
            
            # Check if file exists
            if Path(path).exists():
                print(f"                    ✓ File exists")
                # Load and show size
                img = Image.open(path)
                print(f"                    Size: {img.size}, Mode: {img.mode}")
            else:
                print(f"                    ❌ FILE NOT FOUND!")
    
    # Now test visualization
    print("\n🎨 TESTING VISUALIZATION:")
    engine = CorruptionEngine(feature_index, dataset_dir)
    
    # Create vis directory
    vis_dir = Path(dataset_dir) / "visualizations_debug"
    vis_dir.mkdir(exist_ok=True)
    
    # Test level 1
    print(f"\nCreating corruption level 1...")
    try:
        corrupted, target, features = engine.create_corrupted_face(image_name, 1)
        
        print(f"  Corrupted features: {features}")
        print(f"  Corrupted size: {corrupted.size}")
        print(f"  Target size: {target.size}")
        
        # Save
        vis_path = vis_dir / f"debug_{image_name}.png"
        engine.visualize_corruption(image_name, 1, str(vis_path))
        
        print(f"\n✓ Saved to: {vis_path}")
        print(f"\n📸 OPEN THIS FILE TO SEE:")
        print(f"   Left   = face_contour (should have holes)")
        print(f"   Middle = corrupted (features from other images)")
        print(f"   Right  = jawline (original complete face)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug corruption for specific image")
    parser.add_argument("image_name", type=str, help="Image name to debug (e.g., '00001')")
    parser.add_argument("--dataset-dir", type=str,
                       default="/home/teaching/G14/forensic_reconstruction/dataset")
    
    args = parser.parse_args()
    
    debug_single_image(args.dataset_dir, args.image_name)