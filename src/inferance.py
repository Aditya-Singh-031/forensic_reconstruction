"""
Inference Script.
Run reconstruction on specific images to generate "Before vs After" results.
"""

import torch
import logging
from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Absolute imports
from src.data_loader import CorruptedFaceDataset
from src.model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference(count=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    base_dir = Path("/home/teaching/G14/forensic_reconstruction")
    output_dir = base_dir / "output/inference_results"
    checkpoint_path = base_dir / "output/training_run_1/checkpoints/best_model.pth"
    index_path = base_dir / "dataset/metadata/feature_index.json"
    split_path = base_dir / "dataset/metadata/splits.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    model = create_model('unet_attention', device=device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        # FIX: Key is 'model'
        model.load_state_dict(ckpt['model'])
        model.eval()
        logger.info(f"Loaded model from Epoch {ckpt['epoch']}")
    except FileNotFoundError:
        logger.error("Checkpoint not found.")
        return
    except KeyError:
        logger.error(f"Checkpoint key error. Keys found: {ckpt.keys()}")
        return

    # 2. Direct Dataset Creation (Level 3 Corruption for Demo)
    test_dataset = CorruptedFaceDataset(
        feature_index_path=str(index_path),
        split_path=str(split_path),
        split_name="test",
        corruption_level=3, # Hard mode for demo
        image_size=512,
        augment=False
    )
    
    # Batch size 1 to process individually
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    logger.info(f"Running inference on {count} random samples...")
    
    # 3. Process
    with torch.no_grad():
        found = 0
        for i, batch in enumerate(test_loader):
            if i >= count: break
            
            name = batch['name'][0]
            corrupted = batch['corrupted'].to(device)
            target = batch['target'].to(device)
            
            # Reconstruct
            reconstructed = model(corrupted)
            
            # Prepare visualization: Input | Output | Target
            # Denormalize [-1, 1] -> [0, 1]
            def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
            
            stack = torch.cat([
                denorm(corrupted[0]), 
                denorm(reconstructed[0]), 
                denorm(target[0])
            ], dim=2) # Concatenate horizontally
            
            # Save
            save_path = output_dir / f"result_{name}.png"
            save_image(stack, save_path)
            logger.info(f"Saved: {save_path}")

    logger.info(f"Done! Results in {output_dir}")

if __name__ == "__main__":
    run_inference(count=10)