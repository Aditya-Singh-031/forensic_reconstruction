"""
Test Multi-Face Database and Advanced Matching
Enhanced version with embedding-based similarity search
"""

import sys
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.multi_face_database import MultiFaceDatabase, EmbeddingGenerator
from src.advanced_matching import AdvancedMatchingEngine
from src.description_parser import ForensicDescriptionParser


def demo_basic_database():
    """Demo 1: Basic multi-face database operations."""
    
    print("\n" + "="*60)
    print("DEMO 1: Multi-Face Database Basics")
    print("="*60)
    
    # Initialize database
    print("\nInitializing multi-face database...")
    db = MultiFaceDatabase(db_path='output/forensic_faces.db')
    
    # Load parsed descriptions from earlier step
    parsed_file = Path('output/parsed_descriptions/parsed_results.json')
    if not parsed_file.exists():
        print("❌ No parsed results found. Run test_description_parser first.")
        return db
    
    with open(parsed_file, 'r') as f:
        parsed_results = json.load(f)
    
    print(f"\n✓ Loaded {len(parsed_results)} parsed descriptions")
    
    # Add faces to database
    print("\nAdding faces to database with embeddings...")
    
    import os
    added_count = 0
    
    for i, result in enumerate(parsed_results, 1):
        description = result['raw_description']
        attributes = result['attributes']
        
        # Use generated face images if they exist
        image_path = f"output/text_to_face/face_{i:02d}.png"
        
        # If generated face doesn't exist, try sample image
        if not os.path.exists(image_path):
            image_path = 'output/pipeline_generated_faces/generated_face_00.png'
            if not os.path.exists(image_path):
                logger.warning(f"No image found for {description[:30]}...")
                continue
        
        try:
            record_id = db.add_face(
                description=description,
                image_path=image_path,
                attributes=attributes,
                created_by='demo'
            )
            
            if record_id:
                added_count += 1
                if added_count % 5 == 0:
                    print(f"  ✓ Added {added_count} faces...")
        
        except Exception as e:
            logger.debug(f"Could not add face: {e}")
    
    print(f"\n✓ Successfully added {added_count} faces to database")
    
    # Database statistics
    print("\nDatabase Statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return db


def demo_advanced_matching(db):
    """Demo 2: Advanced matching with embeddings."""
    
    print("\n" + "="*60)
    print("DEMO 2: Advanced Matching Engine")
    print("="*60)
    
    # Initialize components
    print("\nInitializing matching engine...")
    parser = ForensicDescriptionParser()
    matcher = AdvancedMatchingEngine(db, parser)
    
    # Test queries
    test_queries = [
        "Adult male, 35 years old, dark complexion, thick mustache",
        "Young female, 25-30, fair skin, long black hair, glasses",
        "Middle-aged Indian male, 40-50 years old",
    ]
    
    print(f"\nTesting {len(test_queries)} queries...\n")
    
    all_results = {}
    
    for query_idx, query_desc in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"[{query_idx}/{len(test_queries)}] Query: {query_desc}")
        print('='*60)
        
        try:
            # Perform advanced matching
            results = matcher.match_description(
                query_desc,
                top_k=5,
                threshold=0.5,
                use_embeddings=True,
                use_attributes=True,
                use_text=True
            )
            
            all_results[query_desc] = results
            
            # Display results
            if results:
                print(f"\n✓ Found {len(results)} matches:\n")
                
                for result in results:
                    matcher.print_result(result)
            else:
                print("\n⚠ No matches found with current threshold")
        
        except Exception as e:
            print(f"⚠ Matching error: {e}")
            logger.exception("Advanced matching failed")
    
    # Summary
    print("\n" + "="*60)
    print("MATCHING SUMMARY")
    print("="*60)
    
    total_matches = sum(len(results) for results in all_results.values())
    print(f"\nTotal matches found: {total_matches}")
    print(f"Average matches per query: {total_matches / len(all_results) if all_results else 0:.1f}")
    
    for query, results in all_results.items():
        if results:
            best = results[0]
            print(f"\nQuery: {query[:50]}...")
            print(f"  Best match: {best.record_id}")
            print(f"  Composite score: {best.similarity_score:.3f}")
            print(f"    - Embedding: {best.embedding_similarity:.3f}")
            print(f"    - Attributes: {best.attribute_similarity:.3f}")
            print(f"    - Text: {best.text_similarity:.3f}")


def demo_image_search(db):
    """Demo 3: Search by image."""
    
    print("\n" + "="*60)
    print("DEMO 3: Image-Based Search")
    print("="*60)
    
    # Initialize components
    parser = ForensicDescriptionParser()
    matcher = AdvancedMatchingEngine(db, parser)
    
    # Test image search
    test_image = 'output/pipeline_generated_faces/generated_face_00.png'
    
    if Path(test_image).exists():
        print(f"\nSearching for similar faces to: {test_image}")
        
        try:
            results = matcher.match_image(
                test_image,
                top_k=5,
                threshold=0.5
            )
            
            if results:
                print(f"\n✓ Found {len(results)} similar faces:\n")
                
                for result in results:
                    matcher.print_result(result)
            else:
                print("\n⚠ No similar faces found")
        
        except Exception as e:
            print(f"⚠ Image search failed: {e}")
    else:
        print(f"⚠ Test image not found: {test_image}")


def demo_database_export(db):
    """Demo 4: Export database."""
    
    print("\n" + "="*60)
    print("DEMO 4: Database Export")
    print("="*60)
    
    export_path = 'output/forensic_database_export.json'
    
    print(f"\nExporting database to {export_path}...")
    
    try:
        success = db.export_to_json(export_path)
        
        if success:
            print(f"✓ Export successful")
            
            # Show sample of exported data
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            print(f"\nExport summary:")
            print(f"  Total records: {data['metadata']['total_records']}")
            print(f"  Exported at: {data['metadata']['exported']}")
            
            if data['faces']:
                print(f"\nFirst record:")
                first = data['faces'][0]
                print(f"  Record ID: {first['record_id']}")
                print(f"  Description: {first['description'][:60]}...")
                print(f"  Image: {first['image_path']}")
        else:
            print("✗ Export failed")
    
    except Exception as e:
        print(f"⚠ Export error: {e}")


def demo_weight_adjustment(db):
    """Demo 5: Adjust matching weights."""
    
    print("\n" + "="*60)
    print("DEMO 5: Weight Adjustment")
    print("="*60)
    
    parser = ForensicDescriptionParser()
    matcher = AdvancedMatchingEngine(db, parser)
    
    query = "Adult male, 40 years old, dark complexion"
    
    print(f"\nQuery: {query}")
    print("\nTesting different weight configurations:\n")
    
    # Configuration 1: Embedding-heavy
    print("1. Embedding-heavy (0.7 embedding, 0.2 attributes, 0.1 text):")
    matcher.set_weights(embedding=0.7, attributes=0.2, text=0.1)
    
    try:
        results = matcher.match_description(query, top_k=3, threshold=0.4)
        if results:
            best = results[0]
            print(f"   Best match score: {best.similarity_score:.3f}\n")
    except:
        pass
    
    # Configuration 2: Attribute-heavy
    print("2. Attribute-heavy (0.3 embedding, 0.5 attributes, 0.2 text):")
    matcher.set_weights(embedding=0.3, attributes=0.5, text=0.2)
    
    try:
        results = matcher.match_description(query, top_k=3, threshold=0.4)
        if results:
            best = results[0]
            print(f"   Best match score: {best.similarity_score:.3f}\n")
    except:
        pass
    
    # Configuration 3: Balanced
    print("3. Balanced (0.4 embedding, 0.3 attributes, 0.3 text):")
    matcher.set_weights(embedding=0.4, attributes=0.3, text=0.3)
    
    try:
        results = matcher.match_description(query, top_k=3, threshold=0.4)
        if results:
            best = results[0]
            print(f"   Best match score: {best.similarity_score:.3f}\n")
    except:
        pass
    
    print("✓ Weight adjustment demo complete")


def main():
    """Run all demos."""
    
    print("\n" + "="*60)
    print("MULTI-FACE DATABASE & ADVANCED MATCHING SYSTEM")
    print("="*60)
    
    print("\nAvailable demos:")
    print("  1. Basic database operations")
    print("  2. Advanced matching (embeddings + attributes)")
    print("  3. Image-based search")
    print("  4. Database export")
    print("  5. Weight adjustment")
    print("  6. Run all demos")
    
    choice = input("\nSelect demo (1-6) [default: 6]: ").strip() or '6'
    
    # Initialize database first
    db = None
    
    if choice in ['1', '2', '3', '4', '5', '6']:
        db = demo_basic_database()
        
        if db is None:
            print("\n❌ Failed to initialize database. Exiting.")
            return
    
    if choice == '1':
        pass  # Already done in demo_basic_database
    
    elif choice == '2':
        demo_advanced_matching(db)
    
    elif choice == '3':
        demo_image_search(db)
    
    elif choice == '4':
        demo_database_export(db)
    
    elif choice == '5':
        demo_weight_adjustment(db)
    
    elif choice == '6':
        demo_advanced_matching(db)
        demo_image_search(db)
        demo_database_export(db)
        demo_weight_adjustment(db)
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("✓ All demos complete!")
    print("="*60)


if __name__ == '__main__':
    main()
