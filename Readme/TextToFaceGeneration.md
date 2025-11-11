### Text-to-Face Generation (Stable Diffusion)

This module generates photorealistic faces from verbal descriptions using Stable Diffusion (text-to-image).

Files:
- `src/text_to_face.py`: TextToFaceGenerator class (API)
- `src/test_text_to_face.py`: CLI test script for batch generation
- `data/sample_descriptions.txt`: 20 example descriptions

#### Quick start

```bash
# Single description
python -m src.test_text_to_face --description "Adult male, 45 years old, thick mustache, dark complexion" --num_images 2

# From file (batch)
python -m src.test_text_to_face --descriptions_file data/sample_descriptions.txt --num_images 2

# Tuning quality
python -m src.test_text_to_face --description "Middle-aged Indian female, bindi, smiling" --steps 40 --guidance_scale 10 --seed 123
```

#### Parameters
- Steps (20–50): higher = better detail but slower (default 30)
- Guidance scale (7.5–15): higher = follow prompt more (default 7.5)
- Seed: set for reproducibility
- Device: CUDA (GPU) recommended; CPU works but slower

#### Prompting tips (what works)
- Be specific and concrete:
  - Demographics: age range, gender, ethnicity (e.g., "adult male, Indian")
  - Distinctive features: mustache, scars, glasses, hair style/color
  - Expression: neutral, smiling, angry
  - Photographic cues: "photorealistic", "natural lighting", "high detail"
- Example:
  - "Adult male, 45 years old, thick mustache, large ears, dark complexion, photorealistic, natural lighting"

#### What to avoid (bad prompts)
- Vague: "a face", "beautiful"
- Conflicting styles: "cartoon" and "photorealistic" together
- Overly long, redundant adjectives

#### Forensic description structure
- Demographics: age band, gender, ethnicity
- Hair: style, color, length
- Facial hair: mustache/beard type
- Eyes/eyebrows/nose/mouth: notable shapes/traits
- Distinguishing marks: scars, moles, birthmarks, accessories (glasses)
- Expression/pose: neutral, slight smile, frontal view (default)
- Photographic realism cues

#### Troubleshooting
- GPU OOM: reduce `--steps`, use `--device cpu`, or downscale via `height/width` in code
- Output not matching description: increase `--guidance_scale` to 10–12 and refine wording
- Reproducibility: set `--seed` and keep parameters constant

#### Outputs
- Images saved under `output/text_to_face/<sanitized-description>/NN.png`
- Console shows timing per description and generation settings

#### Notes
- First run may download model weights (~4GB–7GB). Cached in `~/.cache/huggingface/`.
- FP16 is used on GPU for speed and memory efficiency.
