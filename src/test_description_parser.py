"""
Test script for Forensic Description Parser
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.description_parser import ForensicDescriptionParser


def main():
    parser = argparse.ArgumentParser(description='Test Forensic Description Parser')
    parser.add_argument('--description', type=str, help='Description text to parse')
    parser.add_argument('--description_file', type=str, help='Text file of sample descriptions (one per line)')
    parser.add_argument('--output', type=str, default='output/parsed_descriptions',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize parser
    print("\nInitializing parser...")
    fdp = ForensicDescriptionParser()
    
    if args.description_file:
    # Read descriptions from file (ignore comment lines starting with #)
        with open(args.description_file, 'r') as f:
            descriptions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    # Sample descriptions if none provided
    elif not args.description:
        descriptions = [
            "Adult male, 40-45 years old, thick black mustache, large ears, Indian complexion",
            "Young woman, 25-30, fair skin, long brown hair, blue eyes, glasses",
            "Middle-aged man, around 50, gray hair, beard, dark complexion, serious expression",
            "Teenage girl, 15-18, black hair, medium build, neutral expression"
        ]
    else:
        descriptions = [args.description]
    
    # Parse descriptions
    results = []
    print("\n" + "="*60)
    print("Parsing Descriptions")
    print("="*60)
    
    for i, desc in enumerate(descriptions, 1):
        print(f"\n[{i}/{len(descriptions)}] {desc}")
        print("-" * 60)
        
        result = fdp.parse(desc)
        results.append(result)
        
        # Display attributes
        attrs = result['attributes']
        print(f"\nExtracted Attributes (Confidence: {result['overall_confidence']:.2f}):")
        
        for attr_name, attr_data in attrs.items():
            if attr_data['value'] is not None:
                value = attr_data['value']
                conf = attr_data['confidence']
                print(f"  {attr_name:20s}: {value} (confidence: {conf:.2f})")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / 'parsed_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {json_path}")
    print("\n" + "="*60)
    print("✓ Parsing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
