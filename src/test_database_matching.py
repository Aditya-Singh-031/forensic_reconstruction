"""
Test Face Database Matching
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_database import FaceDatabase
from src.description_parser import ForensicDescriptionParser
import json


def main():
    print("\n" + "="*60)
    print("Face Database Matching System")
    print("="*60)
    
    # Initialize
    db = FaceDatabase()
    parser = ForensicDescriptionParser()
    
    # Load existing parsed results
    parsed_file = Path('output/parsed_descriptions/parsed_results.json')
    if not parsed_file.exists():
        print("❌ No parsed results found. Run test_description_parser first.")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_results = json.load(f)
    
    print(f"\n✓ Loaded {len(parsed_results)} parsed descriptions")
    
    # Add records to database
    print("\nAdding records to database...")
    for i, result in enumerate(parsed_results, 1):
        description = result['raw_description']
        attributes = result['attributes']
        confidence = result['overall_confidence']
        
        # Placeholder image path (would be real generated image)
        image_path = f"output/text_to_face/face_{i:02d}.png"
        
        record_id = db.add_record(description, image_path, attributes, confidence)
        print(f"  [{i}/{len(parsed_results)}] {record_id}")
    
    # Test search
    print("\n" + "-"*60)
    print("Testing Database Search")
    print("-"*60)
    
    # Example search query
    query_desc = "Adult male, 35 years old, dark complexion, thick mustache"
    query_result = parser.parse(query_desc)
    query_attrs = query_result['attributes']
    
    print(f"\nSearch Query: {query_desc}")
    print(f"Query Attributes: {query_attrs['gender']['value']}, {query_attrs['complexion']['value']}")
    
    # Search database
    results = db.search_by_attributes(query_attrs, threshold=0.5)
    
    print(f"\n✓ Found {len(results)} matching records:\n")
    for rank, (record_id, score) in enumerate(results[:5], 1):
        record = db.get_record(record_id)
        print(f"  [{rank}] {record_id} (Similarity: {score:.2f})")
        print(f"      Description: {record['description'][:60]}...")
        print(f"      Image: {record['image_path']}")
        print()
    
    print("="*60)
    print("✓ Database matching test complete!")
    print("="*60)


if __name__ == '__main__':
    main()
