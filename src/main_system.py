"""
FORENSIC RECONSTRUCTION SYSTEM - FINAL INTEGRATION
Combines: Voice/Text Input -> Generation -> Corruption -> U-Net Reconstruction -> Database Match
"""

import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import time

# --- ROBUST AUDIO IMPORT ---
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except OSError:
    # PortAudio missing (common on headless servers)
    AUDIO_AVAILABLE = False
    print("⚠️ WARNING: PortAudio not found. Live recording disabled. File input still works.")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ WARNING: Audio libs not installed. Live recording disabled.")

# --- IMPORT MODULES ---
from src.text_to_face import TextToFaceGenerator
from src.voice_processor import VoiceDescriptionProcessor
from src.landmark_detector import LandmarkDetector
from src.model import create_model
from src.multi_face_database import MultiFaceDatabase
from src.advanced_matching import AdvancedMatchingEngine
from src.description_parser import ForensicDescriptionParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ForensicSystem:
    def __init__(self):
        self.base_dir = Path("./")
        self.output_dir = self.base_dir / "output/final_demo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CPU Mode for safety/memory
        self.device = 'cpu'
        logger.info(f"🚀 Initializing Forensic System on {self.device}...")

        # 1. Voice Processor
        self.voice = VoiceDescriptionProcessor(model_size="base", device=self.device)
        
        # 2. Face Generator (Boosted Quality)
        self.generator = TextToFaceGenerator(device=self.device)
        
        # 3. Reconstruction Model (U-Net)
        self.reconstructor = create_model('unet_attention', device=self.device)
        self._load_checkpoint()
        
        # 4. Landmark Detector
        self.landmarks = LandmarkDetector()
        
        # 5. Database (Optional)
        self.db = MultiFaceDatabase(str(self.base_dir / "output/forensic_faces.db"))
        self.matcher = AdvancedMatchingEngine(self.db, ForensicDescriptionParser())
        
        logger.info("✅ System Ready.")

    def _load_checkpoint(self):
        ckpt_path = self.base_dir / "output/training_run_1/checkpoints/best_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            # Handle key mismatch if needed
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt['model_state_dict']
            self.reconstructor.load_state_dict(state_dict)
            self.reconstructor.eval()
            logger.info(f"Loaded U-Net from epoch {ckpt['epoch']}")
        else:
            logger.warning("No U-Net checkpoint found! Reconstruction will be untrained.")

    def record_audio(self, duration=5):
        """Record audio from microphone."""
        if not AUDIO_AVAILABLE:
            logger.error("Microphone recording not available on this server.")
            return None

        logger.info(f"🎤 Recording for {duration} seconds...")
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        filename = self.output_dir / "input_audio.wav"
        sf.write(filename, recording, fs)
        logger.info("✓ Recording saved.")
        return str(filename)

    def process_case(self, input_type="text", input_data=None):
        """Run the full pipeline."""
        
        # A. INPUT PARSING
        description = ""
        if input_type == "voice":
            if not input_data:
                # Try recording
                audio_path = self.record_audio()
                if not audio_path:
                    return None
            else:
                audio_path = input_data

            logger.info("Transcribing audio...")
            result = self.voice.transcribe_file(audio_path)
            description = result.text
            logger.info(f"📝 Transcribed: '{description}'")
        else:
            description = input_data
            logger.info(f"📝 Description: '{description}'")

        # B. GENERATION (High Quality)
        logger.info("🎨 Generating Suspect Sketch (this takes ~30-60s on CPU)...")
        # Enhance prompt for better results
        full_prompt = f"photorealistic mugshot of {description}, highly detailed face, neutral background, 8k uhd, dslr"
        
        face_img = self.generator.generate(
            full_prompt, 
            num_inference_steps=50, # Boosted for quality
            guidance_scale=8.5
        )
        gen_path = self.output_dir / "1_suspect_sketch.png"
        face_img.save(gen_path)

        # C. CORRUPTION SIMULATION (Masking Features)
        logger.info("🕶️ Simulating Evidence Corruption...")
        corrupted_img, mask_img = self._apply_corruption(face_img)
        corr_path = self.output_dir / "2_corrupted_evidence.png"
        corrupted_img.save(corr_path)

        # D. DEEP RECONSTRUCTION
        logger.info("🧠 Running U-Net Reconstruction...")
        restored_img = self._run_reconstruction(corrupted_img)
        rec_path = self.output_dir / "3_restored_face.png"
        restored_img.save(rec_path)
        
        # E. DATABASE MATCH
        logger.info("🔍 Searching Database...")
        # Ensure we have something in the DB first (just in case)
        if self.db.get_stats()['total_faces'] == 0:
            logger.info("Database empty, adding current generated face as a reference...")
            self.db.add_face(description, str(gen_path))

        matches = self.matcher.match_image(str(rec_path), top_k=3)
        
        print("\n" + "="*40)
        print("🔍 IDENTIFICATION RESULTS")
        print("="*40)
        for i, match in enumerate(matches):
            print(f"{i+1}. ID: {match.record_id} | Confidence: {match.similarity_score:.2f}")
        print("="*40 + "\n")
        
        return {
            "description": description,
            "paths": [str(gen_path), str(corr_path), str(rec_path)],
            "matches": matches
        }

    def _apply_corruption(self, image):
        """Mask out eyes/mouth using landmarks."""
        # Save temp for detector
        temp_path = self.output_dir / "temp_detect.png"
        image.save(temp_path)
        
        # Detect
        res = self.landmarks.detect(str(temp_path))
        
        corrupted = image.copy()
        draw = ImageDraw.Draw(corrupted)
        
        # Default mask (black out eyes/mouth)
        features = ['left_eye', 'right_eye', 'mouth_outer']
        
        if res and 'groups' in res:
            for ft in features:
                if ft in res['groups']:
                    bbox = res['groups'][ft]['bbox']
                    # Draw black box on image
                    draw.rectangle(
                        [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']], 
                        fill=(0,0,0)
                    )
        else:
            logger.warning("Landmarks failed. Applying fallback center crop.")
            w, h = image.size
            draw.rectangle([w//3, h//3, 2*w//3, 2*h//3], fill=(0,0,0))
            
        return corrupted, None

    def _run_reconstruction(self, image):
        # Preprocess
        img_t = torch.from_numpy(np.array(image.resize((512, 512)))).float() / 127.5 - 1.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Infer
        with torch.no_grad():
            out_t = self.reconstructor(img_t)
            
        # Postprocess
        out_np = out_t.squeeze().permute(1, 2, 0).cpu().numpy()
        out_np = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--input", type=str, help="Text description or Audio path")
    args = parser.parse_args()

    system = ForensicSystem()
    
    # Default description if none provided
    desc = args.input if args.input else "Adult male, 35 years old, short beard, serious expression"
    
    result = system.process_case(input_type=args.mode, input_data=desc)
    
    if result:
        print("\n✅ CASE CLOSED.")
        print(f"1. Sketch:       {result['paths'][0]}")
        print(f"2. Evidence:     {result['paths'][1]}")
        print(f"3. Reconstructed:{result['paths'][2]}")