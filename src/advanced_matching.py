"""
Advanced Matching Engine
Hybrid matching combining embeddings, attributes, and text
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a face match."""
    record_id: str
    similarity_score: float
    embedding_similarity: float
    attribute_similarity: float
    text_similarity: float
    face_data: Dict
    rank: int


class AdvancedMatchingEngine:
    """Advanced matching with embeddings, attributes, and text."""
    
    def __init__(self, database, parser):
        """Initialize matching engine."""
        logger.info("Initializing AdvancedMatchingEngine...")
        
        self.database = database
        self.parser = parser
        
        self.weights = {
            'embedding': 0.5,
            'attributes': 0.3,
            'text': 0.2
        }
        
        logger.info("✓ AdvancedMatchingEngine ready")
    
    def match_description(self, description: str,
                         top_k: int = 10,
                         threshold: float = 0.6,
                         use_embeddings: bool = True,
                         use_attributes: bool = True,
                         use_text: bool = True) -> List[MatchResult]:
        """Match description using multiple methods."""
        logger.info(f"\nMatching: {description[:50]}...")
        
        parsed = self.parser.parse(description)
        query_attributes = parsed['attributes']
        
        candidates = {}
        match_scores = {}
        
        # Text-based search (fastest, uses CLIP)
        if use_text:
            logger.info("  [Text search]")
            text_results = self.database.search_by_text_embedding(
                description, top_k=top_k*2, threshold=0.3
            )
            
            for record_id, sim, face_data in text_results:
                candidates[record_id] = face_data
                if record_id not in match_scores:
                    match_scores[record_id] = {
                        'embedding': 0.0,
                        'attributes': 0.0,
                        'text': sim
                    }
        
        # Composite scoring
        results = []
        for record_id, scores in match_scores.items():
            composite = (
                scores['embedding'] * self.weights['embedding'] +
                scores['attributes'] * self.weights['attributes'] +
                scores['text'] * self.weights['text']
            )
            
            if composite >= threshold:
                result = MatchResult(
                    record_id=record_id,
                    similarity_score=composite,
                    embedding_similarity=scores['embedding'],
                    attribute_similarity=scores['attributes'],
                    text_similarity=scores['text'],
                    face_data=candidates[record_id],
                    rank=0
                )
                results.append(result)
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        results = results[:top_k]
        
        for i, result in enumerate(results, 1):
            result.rank = i
        
        logger.info(f"  ✓ Found {len(results)} matches")
        
        return results
    
    def match_image(self, image_path: str,
                   top_k: int = 10,
                   threshold: float = 0.6) -> List[MatchResult]:
        """Search by image."""
        logger.info(f"Image search: {image_path}")
        
        embedding_results = self.database.search_by_image(
            image_path, top_k=top_k, threshold=threshold
        )
        
        results = []
        for record_id, emb_sim, face_data in embedding_results:
            result = MatchResult(
                record_id=record_id,
                similarity_score=emb_sim,
                embedding_similarity=emb_sim,
                attribute_similarity=0.0,
                text_similarity=0.0,
                face_data=face_data,
                rank=len(results) + 1
            )
            results.append(result)
        
        return results
    
    def set_weights(self, embedding: float = 0.5,
                   attributes: float = 0.3,
                   text: float = 0.2):
        """Adjust matching weights."""
        total = embedding + attributes + text
        
        self.weights = {
            'embedding': embedding / total,
            'attributes': attributes / total,
            'text': text / total
        }
        
        logger.info(f"Weights updated: {self.weights}")
    
    def print_result(self, result: MatchResult):
        """Print match result."""
        print(f"\n[{result.rank}] {result.record_id} (Score: {result.similarity_score:.3f})")
        print(f"    Description: {result.face_data['description'][:60]}...")
        print(f"    Scores: Embedding={result.embedding_similarity:.3f}, "
              f"Attributes={result.attribute_similarity:.3f}, "
              f"Text={result.text_similarity:.3f}")
        print(f"    Image: {result.face_data['image_path']}")
