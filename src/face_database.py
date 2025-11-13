"""
Face Database for Storing and Matching Face Records
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Store and retrieve face records with attribute matching."""
    
    def __init__(self, db_path: str = 'output/face_database.json'):
        """Initialize face database."""
        logger.info("Initializing FaceDatabase...")
        self.db_path = Path(db_path)
        self.records = []
        
        # Load existing database if available
        if self.db_path.exists():
            self._load_database()
        
        logger.info(f"✓ FaceDatabase initialized ({len(self.records)} records)")
    
    def add_record(self, description: str, generated_image_path: str, 
                   parsed_attributes: Dict, confidence: float) -> str:
        """
        Add a new face record to database.
        
        Args:
            description: Original text description
            generated_image_path: Path to generated face image
            parsed_attributes: Structured attributes from parser
            confidence: Parser confidence score
            
        Returns:
            Record ID
        """
        record_id = f"FACE_{len(self.records):05d}"
        
        record = {
            'id': record_id,
            'description': description,
            'image_path': str(generated_image_path),
            'attributes': parsed_attributes,
            'confidence': confidence
        }
        
        self.records.append(record)
        self._save_database()
        
        logger.info(f"✓ Added record: {record_id}")
        return record_id
    
    def search_by_attributes(self, query_attributes: Dict, 
                            threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Search database for matching faces by attributes.
        
        Args:
            query_attributes: Attributes to search for
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (record_id, similarity_score) tuples, sorted by score
        """
        results = []
        
        for record in self.records:
            similarity = self._calculate_similarity(query_attributes, 
                                                   record['attributes'])
            
            if similarity >= threshold:
                results.append((record['id'], similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _calculate_similarity(self, query: Dict, record: Dict) -> float:
        """Calculate similarity between two attribute sets."""
        scores = []
        
        # Gender match
        if query.get('gender', {}).get('value') and record.get('gender', {}).get('value'):
            if query['gender']['value'] == record['gender']['value']:
                scores.append(0.95)
            else:
                scores.append(0.0)
        
        # Complexion match
        if query.get('complexion', {}).get('value') and record.get('complexion', {}).get('value'):
            if query['complexion']['value'] == record['complexion']['value']:
                scores.append(0.85)
            else:
                scores.append(0.3)  # Partial credit for different but close values
        
        # Age range match
        query_age = query.get('age', {}).get('value')
        record_age = record.get('age', {}).get('value')
        if query_age and record_age:
            age_similarity = self._age_similarity(query_age, record_age)
            scores.append(age_similarity)
        
        # Facial hair match
        if query.get('facial_hair', {}).get('value') and record.get('facial_hair', {}).get('value'):
            if query['facial_hair']['value'] == record['facial_hair']['value']:
                scores.append(0.8)
            else:
                scores.append(0.2)
        
        # Hair match
        if query.get('hair', {}).get('value') and record.get('hair', {}).get('value'):
            hair_sim = self._hair_similarity(query['hair']['value'], 
                                            record['hair']['value'])
            scores.append(hair_sim)
        
        # Overall similarity
        return sum(scores) / len(scores) if scores else 0.0
    
    def _age_similarity(self, query_age: str, record_age: str) -> float:
        """Calculate age similarity."""
        try:
            q_min, q_max = map(int, query_age.split('-'))
        except:
            try:
                q_min = q_max = int(query_age)
            except:
                return 0.0
        
        try:
            r_min, r_max = map(int, record_age.split('-'))
        except:
            try:
                r_min = r_max = int(record_age)
            except:
                return 0.0
        
        # Calculate overlap
        overlap_min = max(q_min, r_min)
        overlap_max = min(q_max, r_max)
        
        if overlap_min > overlap_max:
            return 0.0  # No overlap
        
        overlap = overlap_max - overlap_min + 1
        total = max(q_max, r_max) - min(q_min, r_min) + 1
        
        return overlap / total
    
    def _hair_similarity(self, query_hair: Dict, record_hair: Dict) -> float:
        """Calculate hair attribute similarity."""
        score = 0
        count = 0
        
        # Color
        if query_hair.get('color') and record_hair.get('color'):
            count += 1
            if query_hair['color'] == record_hair['color']:
                score += 0.7
            else:
                score += 0.2
        
        # Length
        if query_hair.get('length') and record_hair.get('length'):
            count += 1
            if query_hair['length'] == record_hair['length']:
                score += 0.7
            else:
                score += 0.2
        
        return score / count if count > 0 else 0.5
    
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get a specific record by ID."""
        for record in self.records:
            if record['id'] == record_id:
                return record
        return None
    
    def _save_database(self):
        """Save database to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"✓ Database saved to {self.db_path}")
    
    def _load_database(self):
        """Load database from JSON file."""
        try:
            with open(self.db_path, 'r') as f:
                self.records = json.load(f)
            logger.info(f"✓ Loaded {len(self.records)} records from {self.db_path}")
        except Exception as e:
            logger.warning(f"Could not load database: {e}")
            self.records = []
