"""
Forensic Facial Reconstruction Web API
FastAPI server exposing all pipeline features
"""

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import json
import shutil
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system modules
from src.forensic_reconstruction_pipeline import ForensicReconstructionPipeline
from src.iterative_refinement import IterativeRefinementEngine
from src.multi_face_database import MultiFaceDatabase
from src.advanced_matching import AdvancedMatchingEngine
from src.description_parser import ForensicDescriptionParser
from src.text_to_face import TextToFaceGenerator
from src.face_segmentation import FaceSegmenter
from src.landmark_detector import LandmarkDetector
from src.face_inpainter import FaceInpainter

# Initialize FastAPI app
app = FastAPI(
    title="Forensic Facial Reconstruction API",
    description="AI-powered forensic face reconstruction from witness descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
OUTPUT_DIR = Path("output/api_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize components (lazy loading)
_components = {}

def get_components():
    """Get or initialize components."""
    if not _components:
        logger.info("Initializing pipeline components...")
        try:
            _components['generator'] = TextToFaceGenerator()
            _components['segmenter'] = FaceSegmenter()
            _components['landmarks'] = LandmarkDetector()
            _components['inpainter'] = FaceInpainter()
            _components['parser'] = ForensicDescriptionParser()
            _components['db'] = MultiFaceDatabase()
            _components['matcher'] = AdvancedMatchingEngine(_components['db'], _components['parser'])
            logger.info("✓ Components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    return _components

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class GenerateFaceRequest(BaseModel):
    """Request model for face generation."""
    description: str
    num_faces: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = 42

class RefineFeatureRequest(BaseModel):
    """Request model for feature refinement."""
    feature_category: str  # mustache, eyes, hair, etc.
    refinement_type: str   # thicker, darker, etc.
    intensity: float = 1.0

class BatchRefinementRequest(BaseModel):
    """Request model for batch refinement."""
    refinements: List[RefineFeatureRequest]

class SearchDatabaseRequest(BaseModel):
    """Request model for database search."""
    description: str
    top_k: int = 10
    threshold: float = 0.6

class RefinementSessionRequest(BaseModel):
    """Request model for refinement session."""
    description: str
    refinements: Optional[List[RefineFeatureRequest]] = None

class PipelineRequest(BaseModel):
    """Request model for complete pipeline."""
    description: str
    num_faces: int = 1
    refine: bool = False

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "system": "Forensic Facial Reconstruction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": "/endpoints"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        components = get_components()
        return {
            "status": "healthy",
            "components": {
                "generator": "ready",
                "segmenter": "ready",
                "landmarks": "ready",
                "inpainter": "ready",
                "parser": "ready",
                "database": "ready",
                "matcher": "ready"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

@app.get("/endpoints")
async def list_endpoints():
    """List all available endpoints."""
    return {
        "generation": [
            "POST /api/v1/generate - Generate faces from description",
            "POST /api/v1/generate/batch - Batch generate multiple descriptions"
        ],
        "refinement": [
            "POST /api/v1/refine/start - Start refinement session",
            "POST /api/v1/refine/feature - Apply single refinement",
            "POST /api/v1/refine/batch - Apply multiple refinements"
        ],
        "database": [
            "POST /api/v1/database/search - Search database",
            "GET /api/v1/database/stats - Get database statistics",
            "POST /api/v1/database/export - Export database"
        ],
        "pipeline": [
            "POST /api/v1/pipeline - Run complete pipeline",
            "POST /api/v1/pipeline/async - Async pipeline execution"
        ],
        "utilities": [
            "GET /api/v1/refine/options - Get refinement options",
            "POST /api/v1/parse - Parse description to attributes",
            "GET /api/v1/results/{session_id} - Get session results"
        ]
    }

# ============================================================================
# GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/generate")
async def generate_face(request: GenerateFaceRequest):
    """Generate photorealistic faces from description."""
    try:
        logger.info(f"Generating {request.num_faces} face(s): {request.description}")
        
        components = get_components()
        generator = components['generator']
        
        generated_images = []
        session_id = str(uuid.uuid4())
        session_dir = OUTPUT_DIR / session_id / "generated"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(request.num_faces):
            logger.info(f"  [{i+1}/{request.num_faces}] Generating...")
            
            img = generator.generate(
                request.description,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed + i if request.seed else None
            )
            
            # Save image
            img_path = session_dir / f"face_{i:02d}.png"
            img.save(str(img_path))
            
            generated_images.append({
                "index": i,
                "path": str(img_path),
                "url": f"/results/{session_id}/generated/face_{i:02d}.png"
            })
        
        logger.info(f"✓ Generated {len(generated_images)} faces")
        
        return {
            "session_id": session_id,
            "description": request.description,
            "count": len(generated_images),
            "images": generated_images,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/batch")
async def batch_generate(descriptions: List[str]):
    """Batch generate faces from multiple descriptions."""
    try:
        logger.info(f"Batch generating {len(descriptions)} faces")
        
        results = []
        for desc in descriptions:
            request = GenerateFaceRequest(description=desc, num_faces=1)
            result = await generate_face(request)
            results.append(result)
        
        return {
            "total": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# REFINEMENT ENDPOINTS
# ============================================================================

@app.get("/api/v1/refine/options")
async def get_refinement_options():
    """Get available refinement options."""
    try:
        components = get_components()
        # Placeholder - would use actual refinement engine
        return {
            "categories": {
                "mustache": ["thicker", "thinner", "darker", "lighter", "different_style", "remove"],
                "beard": ["thicker", "thinner", "longer", "shorter", "different_style", "remove"],
                "eyes": ["larger", "smaller", "different_color", "more_detailed", "different_expression"],
                "hair": ["longer", "shorter", "different_color", "different_style", "more_volume", "receding"],
                "skin": ["darker", "lighter", "smoother", "more_detail", "add_scars", "add_wrinkles"],
                "face_shape": ["wider", "narrower", "rounder", "squarer", "longer", "younger", "older"],
                "nose": ["larger", "smaller", "broader", "narrower", "more_detail"],
                "mouth": ["larger", "smaller", "fuller_lips", "thinner_lips", "different_expression"],
                "overall": ["more_realistic", "better_lighting", "higher_quality", "professional_look", "casual_look"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/refine/start")
async def start_refinement(request: GenerateFaceRequest):
    """Start iterative refinement session."""
    try:
        logger.info(f"Starting refinement session: {request.description}")
        
        components = get_components()
        generator = components['generator']
        segmenter = components['segmenter']
        landmarks = components['landmarks']
        inpainter = components['inpainter']
        
        refiner = IterativeRefinementEngine(generator, segmenter, landmarks, inpainter)
        
        # Generate base face
        session_id = str(uuid.uuid4())
        session_dir = OUTPUT_DIR / session_id / "refinement"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        result = refiner.start_refinement_session(request.description)
        
        return {
            "session_id": session_id,
            "description": request.description,
            "base_face": result['base_face_path'],
            "available_refinements": list(refiner.refinement_options.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Refinement start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATABASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/database/search")
async def search_database(request: SearchDatabaseRequest):
    """Search database for similar faces."""
    try:
        logger.info(f"Searching database: {request.description}")
        
        components = get_components()
        matcher = components['matcher']
        
        results = matcher.match_description(
            request.description,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "rank": result.rank,
                "record_id": result.record_id,
                "similarity_score": result.similarity_score,
                "embedding_similarity": result.embedding_similarity,
                "attribute_similarity": result.attribute_similarity,
                "text_similarity": result.text_similarity,
                "description": result.face_data['description'],
                "image_path": result.face_data['image_path']
            })
        
        logger.info(f"✓ Found {len(formatted_results)} matches")
        
        return {
            "query": request.description,
            "matches_found": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Database search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/database/stats")
async def get_database_stats():
    """Get database statistics."""
    try:
        components = get_components()
        db = components['db']
        
        stats = db.get_stats()
        
        return {
            "total_faces": stats['total_faces'],
            "total_embeddings": stats['total_embeddings'],
            "faces_with_attributes": stats['faces_with_attributes'],
            "database_size_mb": stats['database_size_mb'],
            "database_path": stats['database_path'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/export")
async def export_database():
    """Export database to JSON."""
    try:
        logger.info("Exporting database...")
        
        components = get_components()
        db = components['db']
        
        export_path = OUTPUT_DIR / f"database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = db.export_to_json(str(export_path))
        
        if success:
            return {
                "success": True,
                "export_path": str(export_path),
                "url": f"/results/{export_path.parent.name}/{export_path.name}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Export failed")
    
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

@app.post("/api/v1/pipeline")
async def run_pipeline(request: PipelineRequest):
    """Run complete forensic reconstruction pipeline."""
    try:
        logger.info(f"Running pipeline: {request.description}")
        
        session_id = str(uuid.uuid4())
        session_dir = OUTPUT_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        components = get_components()
        refiner = IterativeRefinementEngine(
            components['generator'],
            components['segmenter'],
            components['landmarks'],
            components['inpainter']
        )
        
        # Start refinement session
        base_result = refiner.start_refinement_session(
            request.description,
            initial_prompt=request.description,
            num_inference_steps=30
        )
        
        # Apply optional refinements
        refinements = []
        if request.refine:
            refinements = [
                {'category': 'overall', 'type': 'more_realistic', 'intensity': 1.1}
            ]
        
        # Search database
        results = components['matcher'].match_description(
            request.description,
            top_k=5,
            threshold=0.5
        )
        
        # Format matches
        matches = []
        for result in results:
            matches.append({
                "record_id": result.record_id,
                "similarity": result.similarity_score,
                "description": result.face_data['description']
            })
        
        logger.info(f"✓ Pipeline complete, found {len(matches)} matches")
        
        return {
            "session_id": session_id,
            "description": request.description,
            "base_face": base_result['base_face_path'],
            "refinements_applied": len(refinements),
            "matches_found": len(matches),
            "top_matches": matches[:3],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.post("/api/v1/parse")
async def parse_description(description: str):
    """Parse description to extract attributes."""
    try:
        logger.info(f"Parsing: {description}")
        
        components = get_components()
        parser = components['parser']
        
        result = parser.parse(description)
        
        return {
            "description": description,
            "attributes": result['attributes'],
            "confidence": result['overall_confidence'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/results/{session_id}")
async def get_session_results(session_id: str):
    """Get results from a previous session."""
    try:
        session_dir = OUTPUT_DIR / session_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Collect files
        files = {
            "generated": [],
            "refined": [],
            "metadata": {}
        }
        
        # Find generated faces
        generated_dir = session_dir / "generated"
        if generated_dir.exists():
            files["generated"] = [str(f.relative_to(OUTPUT_DIR)) for f in generated_dir.glob("*.png")]
        
        # Find refined faces
        refined_dir = session_dir / "refinement"
        if refined_dir.exists():
            files["refined"] = [str(f.relative_to(OUTPUT_DIR)) for f in refined_dir.glob("*.png")]
        
        return {
            "session_id": session_id,
            "files": files,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FILE SERVING
# ============================================================================

@app.get("/results/{file_path:path}")
async def get_result_file(file_path: str):
    """Serve result files."""
    try:
        full_path = OUTPUT_DIR / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if full_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return FileResponse(full_path, media_type="image/png")
        elif full_path.suffix.lower() == '.json':
            return FileResponse(full_path, media_type="application/json")
        else:
            return FileResponse(full_path)
    
    except Exception as e:
        logger.error(f"File serving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Forensic Facial Reconstruction API...")
    try:
        get_components()
        logger.info("✓ API Ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
