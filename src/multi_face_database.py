"""
Multi-Face Database with Vector Embeddings
Scalable face storage and similarity-based matching
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import sqlite3
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate face embeddings using CLIP."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize CLIP model."""
        logger.info("Initializing EmbeddingGenerator with CLIP...")
        
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            self.device = device
            self.model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(self.model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            logger.info(f"✓ CLIP loaded on {device}")
        except Exception as e:
            logger.warning(f"CLIP initialization failed: {e}")
            self.model = None
            self.processor = None
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate 512-dim embedding for image."""
        if self.model is None:
            logger.warning("CLIP unavailable, using random embedding")
            return np.random.randn(512).astype(np.float32)
        
        try:
            import torch
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(512, dtype=np.float32)


class MultiFaceDatabase:
    """Scalable multi-face database with vector similarity search."""
    
    def __init__(self, db_path: str = 'output/forensic_faces.db',
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """Initialize database."""
        logger.info("Initializing MultiFaceDatabase...")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = 512
        
        self._init_database()
        logger.info(f"✓ Database ready at {db_path}")
    
    def _init_database(self):
        """Create database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE,
                description TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                face_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER,
                FOREIGN KEY(face_id) REFERENCES faces(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attributes (
                face_id INTEGER,
                attribute_name TEXT,
                attribute_value TEXT,
                confidence REAL,
                FOREIGN KEY(face_id) REFERENCES faces(id),
                PRIMARY KEY(face_id, attribute_name)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_record_id ON faces(record_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON faces(timestamp)')
        
        conn.commit()
        conn.close()
    
    def add_face(self, description: str, image_path: str,
                 attributes: Optional[Dict] = None,
                 created_by: str = 'system') -> str:
        """Add face with embedding."""
        try:
            record_id = f"FACE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(description) % 10000:04d}"
            embedding = self.embedding_gen.generate_embedding(image_path)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO faces (record_id, description, image_path, created_by)
                VALUES (?, ?, ?, ?)
            ''', (record_id, description, str(image_path), created_by))
            
            face_id = cursor.lastrowid
            
            cursor.execute('''
                INSERT INTO embeddings (face_id, embedding, embedding_dim)
                VALUES (?, ?, ?)
            ''', (face_id, embedding.tobytes(), self.embedding_dim))
            
            if attributes:
                for attr_name, attr_info in attributes.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO attributes 
                        (face_id, attribute_name, attribute_value, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (face_id, attr_name, str(attr_info.get('value')), 
                          attr_info.get('confidence', 1.0)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Added {record_id}")
            return record_id
        
        except Exception as e:
            logger.error(f"Failed to add face: {e}")
            return None
    
    def search_by_embedding(self, query_embedding: np.ndarray,
                           top_k: int = 10,
                           threshold: float = 0.5) -> List[Tuple]:
        """Search by embedding similarity."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT face_id, embedding FROM embeddings')
            results = []
            
            for face_id, embedding_bytes in cursor.fetchall():
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                
                if similarity >= threshold:
                    results.append((face_id, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            final_results = []
            for face_id, similarity in results:
                cursor.execute('''
                    SELECT record_id, description, image_path, timestamp
                    FROM faces WHERE id = ?
                ''', (face_id,))
                
                row = cursor.fetchone()
                if row:
                    final_results.append((row[0], float(similarity), {
                        'description': row[1],
                        'image_path': row[2],
                        'timestamp': row[3]
                    }))
            
            conn.close()
            return final_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_image(self, image_path: str, top_k: int = 10,
                       threshold: float = 0.5) -> List[Tuple]:
        """Search by image."""
        query_embedding = self.embedding_gen.generate_embedding(image_path)
        return self.search_by_embedding(query_embedding, top_k, threshold)
    
    def search_by_text_embedding(self, text: str, top_k: int = 10,
                                threshold: float = 0.5) -> List[Tuple]:
        """Search by text description."""
        try:
            import torch
            
            inputs = self.embedding_gen.processor(text=text, return_tensors="pt").to(
                self.embedding_gen.device
            )
            
            with torch.no_grad():
                text_features = self.embedding_gen.model.get_text_features(**inputs)
            
            query_embedding = text_features.cpu().numpy()[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            return self.search_by_embedding(query_embedding.astype(np.float32), top_k, threshold)
        
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM faces')
            total_faces = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT face_id) FROM attributes')
            faces_with_attrs = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_faces': total_faces,
                'total_embeddings': total_embeddings,
                'faces_with_attributes': faces_with_attrs,
                'database_path': str(self.db_path),
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }
        
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
    
    def export_to_json(self, output_path: str) -> bool:
        """Export database to JSON."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT record_id, description, image_path, timestamp FROM faces')
            
            data = {
                'metadata': {
                    'exported': datetime.now().isoformat(),
                    'total_records': 0
                },
                'faces': []
            }
            
            for record_id, description, image_path, timestamp in cursor.fetchall():
                data['faces'].append({
                    'record_id': record_id,
                    'description': description,
                    'image_path': image_path,
                    'timestamp': timestamp
                })
            
            data['metadata']['total_records'] = len(data['faces'])
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✓ Exported to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
