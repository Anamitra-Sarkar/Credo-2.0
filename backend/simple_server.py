"""
Simplified FastAPI Gateway for VeriFy AI - Development Mode
This version uses mock responses and doesn't require database or AI models.
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import time
import random
import uuid
import asyncio
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="VeriFy AI Gateway",
    description="Fake News & Deepfake Detection API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    service: str

class TextDetectionRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

class DetectionResponse(BaseModel):
    detection_id: int
    verdict: str
    confidence: float
    explanation: str
    model_used: str
    processing_time_ms: int
    original_language: str = "en"
    translated_to_english: bool = False

class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str

class VideoResultResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None

class TrendingTopic(BaseModel):
    id: int
    title: str
    category: str
    fake_count: int
    real_count: int
    total_checks: int
    trending_score: float
    created_at: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict

# ===== Endpoints =====

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="operational",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        service="VeriFy AI Gateway (Development Mode)"
    )

@app.post("/api/v1/check-text", response_model=DetectionResponse)
async def check_text(request: TextDetectionRequest):
    """Check text for fake news."""
    start_time = time.time()
    
    # Simulate processing
    await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # Random verdict
    verdicts = ["fake", "real", "unverified"]
    verdict = random.choice(verdicts)
    confidence = random.uniform(0.6, 0.95)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return DetectionResponse(
        detection_id=random.randint(1000, 9999),
        verdict=verdict,
        confidence=confidence,
        explanation=f"Analysis indicates this content is likely {verdict}. Confidence: {confidence*100:.1f}%",
        model_used="LIAR Political Fact-Checker" if "politic" in request.text.lower() else "Brain2 General Fact-Checker",
        processing_time_ms=processing_time,
        original_language=request.language,
        translated_to_english=False
    )

@app.post("/api/v1/check-image", response_model=DetectionResponse)
async def check_image(file: UploadFile = File(...)):
    """Check image for deepfakes."""
    start_time = time.time()
    
    # Simulate processing
    await asyncio.sleep(random.uniform(1.0, 2.0))
    
    verdicts = ["fake", "real"]
    verdict = random.choice(verdicts)
    confidence = random.uniform(0.7, 0.98)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return DetectionResponse(
        detection_id=random.randint(1000, 9999),
        verdict=verdict,
        confidence=confidence,
        explanation=f"Image analysis complete. Detected as {verdict} with {confidence*100:.1f}% confidence.",
        model_used="Deepfake Image Detector",
        processing_time_ms=processing_time
    )

@app.post("/api/v1/check-video", response_model=VideoJobResponse)
async def check_video(file: UploadFile = File(...)):
    """Upload video for async processing."""
    job_id = str(uuid.uuid4())
    
    return VideoJobResponse(
        job_id=job_id,
        status="processing",
        progress=0.0,
        message="Video uploaded successfully. Processing in background."
    )

@app.get("/api/v1/check-video/result/{job_id}", response_model=VideoResultResponse)
async def get_video_result(job_id: str):
    """Get video detection result."""
    # Simulate progress
    progress = random.uniform(50, 100)
    
    if progress < 95:
        return VideoResultResponse(
            job_id=job_id,
            status="processing",
            progress=progress
        )
    
    verdict = random.choice(["fake", "real"])
    confidence = random.uniform(0.75, 0.95)
    
    return VideoResultResponse(
        job_id=job_id,
        status="completed",
        progress=100.0,
        verdict=verdict,
        confidence=confidence,
        explanation=f"Video analysis complete. Detected as {verdict}."
    )

@app.post("/api/v1/check-voice", response_model=DetectionResponse)
async def check_voice(file: UploadFile = File(...)):
    """Check voice/audio for deepfakes."""
    start_time = time.time()
    
    await asyncio.sleep(random.uniform(1.0, 2.0))
    
    verdict = random.choice(["fake", "real"])
    confidence = random.uniform(0.8, 0.97)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return DetectionResponse(
        detection_id=random.randint(1000, 9999),
        verdict=verdict,
        confidence=confidence,
        explanation=f"Voice analysis complete. Detected as {verdict} with {confidence*100:.1f}% confidence.",
        model_used="Deepfake Voice Detector",
        processing_time_ms=processing_time
    )

@app.get("/api/v1/trending", response_model=List[TrendingTopic])
async def get_trending(limit: int = 10):
    """Get trending fake news topics."""
    topics = [
        TrendingTopic(
            id=1,
            title="Climate Change Misinformation",
            category="politics",
            fake_count=150,
            real_count=50,
            total_checks=200,
            trending_score=0.85,
            created_at=datetime.utcnow().isoformat()
        ),
        TrendingTopic(
            id=2,
            title="Vaccine Hoax Claims",
            category="health",
            fake_count=200,
            real_count=100,
            total_checks=300,
            trending_score=0.75,
            created_at=datetime.utcnow().isoformat()
        )
    ]
    return topics[:limit]

@app.post("/api/v1/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register new user."""
    return TokenResponse(
        access_token="mock_access_token_" + str(uuid.uuid4()),
        refresh_token="mock_refresh_token_" + str(uuid.uuid4()),
        user={
            "id": str(uuid.uuid4()),
            "email": request.email,
            "name": request.name or "User"
        }
    )

@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login user."""
    return TokenResponse(
        access_token="mock_access_token_" + str(uuid.uuid4()),
        refresh_token="mock_refresh_token_" + str(uuid.uuid4()),
        user={
            "id": str(uuid.uuid4()),
            "email": request.email,
            "name": "User"
        }
    )

@app.get("/api/v1/explanation/{detection_id}")
async def get_explanation(detection_id: int):
    """Get detailed explanation."""
    return {
        "detection_id": detection_id,
        "detailed_explanation": "This is a detailed analysis of the content...",
        "sources": ["https://example.com/fact-check"],
        "confidence_breakdown": {
            "text_analysis": 0.85,
            "context_analysis": 0.78
        }
    }

# ===== Routes without /api/v1 prefix for frontend compatibility =====

@app.post("/check-text")
async def check_text_short(request: TextDetectionRequest):
    """Check text - short route."""
    result = await check_text(request)
    return {
        "is_fake": result.verdict == "fake",
        "confidence": result.confidence,
        "analysis": result.explanation
    }

@app.post("/check-image")
async def check_image_short(file: UploadFile = File(...)):
    """Check image - short route."""
    result = await check_image(file)
    return {
        "is_fake": result.verdict == "fake",
        "confidence": result.confidence,
        "analysis": result.explanation
    }

@app.post("/check-video")
async def check_video_short(file: UploadFile = File(...)):
    """Check video - short route."""
    job = await check_video(file)
    # Return immediate result instead of job
    verdict = random.choice(["fake", "real"])
    confidence = random.uniform(0.75, 0.95)
    return {
        "is_fake": verdict == "fake",
        "confidence": confidence,
        "analysis": f"Video analysis complete. Detected as {verdict}."
    }

@app.post("/check-voice")
async def check_voice_short(file: UploadFile = File(...)):
    """Check voice - short route."""
    result = await check_voice(file)
    return {
        "is_fake": result.verdict == "fake",
        "confidence": result.confidence,
        "analysis": result.explanation
    }

@app.get("/trending")
async def trending_short(limit: int = 10):
    """Get trending topics - short route."""
    return await get_trending(limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
