"""
Detection API endpoints for text, image, video, and voice analysis.
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, Form, BackgroundTasks, HTTPException, status
from fastapi import APIRouter, Depends, File, UploadFile, Form, BackgroundTasks, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from shared.database.session import get_db
from shared.database.models import Detection, DetectionType, DetectionVerdict, VideoJob, JobStatus    
from shared.auth.jwt import get_current_user_id, get_optional_user_id
from shared.config import settings
# from ..services.detection_service import DetectionService
# from ..services.translation_service import TranslationService
import time
import random

router = APIRouter()


# Request/Response Models
class TextDetectionRequest(BaseModel):
    """Request model for text detection."""
    text: str = Field(..., min_length=1, max_length=settings.max_text_length)
    language: Optional[str] = Field(None, pattern="^[a-z]{2}$")


class DetectionResponse(BaseModel):
    """Response model for detection results."""
    detection_id: int
    verdict: DetectionVerdict
    confidence: float
    explanation: Optional[str] = None
    model_used: str
    processing_time_ms: int
    original_language: Optional[str] = None
    translated_to_english: bool = False


class VideoJobResponse(BaseModel):
    """Response model for video processing job."""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str


class VideoResultResponse(BaseModel):
    """Response model for video detection result."""
    job_id: str
    status: JobStatus
    progress: float
    verdict: Optional[DetectionVerdict] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    error_message: Optional[str] = None


@router.post("/check-text", response_model=DetectionResponse)
async def check_text(
    request: TextDetectionRequest,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Check text content for fake news or misinformation.
    
    - Automatically detects language
    - Translates to English if necessary
    - Routes to appropriate model (LIAR for US politics, Brain2 for general)
    - Returns verdict with explanation
    """
    detection_service = DetectionService(db)
    translation_service = TranslationService()
    
    # Detect language
    detected_language = request.language or await translation_service.detect_language(request.text)
    
    # Translate if necessary
    text_to_analyze = request.text
    translated = False
    
    if detected_language != "en":
        text_to_analyze = await translation_service.translate_to_english(request.text, detected_language)
        translated = True
    
    # Perform detection
    result = await detection_service.check_text(
        text=text_to_analyze,
        original_text=request.text,
        language=detected_language,
        user_id=user_id,
        translated=translated
    )
    
    return result


@router.post("/check-image", response_model=DetectionResponse)
async def check_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Check image for deepfake manipulation.
    
    - Accepts JPG, PNG, WebP formats
    - Maximum size: 10MB
    - Returns verdict with confidence score
    """
    # Validate file
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: JPG, PNG, WebP"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_image_size_mb}MB"
        )
    
    detection_service = DetectionService(db)
    
    # Perform detection
    result = await detection_service.check_image(
        file_content=file_content,
        filename=file.filename,
        user_id=user_id
    )
    
    return result


@router.post("/check-video", response_model=VideoJobResponse)
async def check_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Check video for deepfake manipulation (async processing).
    
    - Accepts MP4, WebM, MOV formats
    - Maximum size: 100MB
    - Returns job_id for status checking
    - Processing happens asynchronously via Cloud Pub/Sub
    """
    # Validate file
    if file.content_type not in ["video/mp4", "video/webm", "video/quicktime"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: MP4, WebM, MOV"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_video_size_mb}MB"
        )
    
    detection_service = DetectionService(db)
    
    # Create job and enqueue for processing
    job = await detection_service.create_video_job(
        file_content=file_content,
        filename=file.filename,
        user_id=user_id
    )
    
    return VideoJobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=0.0,
        message="Video uploaded successfully. Processing in background."
    )


@router.get("/check-video/result/{job_id}", response_model=VideoResultResponse)
async def get_video_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Get video detection result by job ID.
    
    - Returns current status and progress
    - If completed, includes verdict and explanation
    """
    detection_service = DetectionService(db)
    
    job = await detection_service.get_video_job(job_id, user_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return VideoResultResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        verdict=job.verdict,
        confidence=job.confidence,
        explanation=job.explanation,
        error_message=job.error_message
    )


@router.post("/check-voice", response_model=DetectionResponse)
async def check_voice(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Check audio/voice for deepfake manipulation.
    
    - Accepts MP3, WAV, M4A, OGG formats
    - Maximum size: 20MB
    - Returns verdict with confidence score
    """
    # Validate file
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/mp4", "audio/ogg", "audio/x-m4a"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: MP3, WAV, M4A, OGG"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_audio_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_audio_size_mb}MB"
        )
    
    detection_service = DetectionService(db)
    
    # Perform detection
    result = await detection_service.check_voice(
        file_content=file_content,
        filename=file.filename,
        user_id=user_id
    )
    
    return result


@router.get("/explanation/{detection_id}")
async def get_explanation(
    detection_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Get detailed explanation for a detection result.
    
    - Returns comprehensive analysis
    - May include references and fact-check sources (via Tavily API)
    """
    detection_service = DetectionService(db)
    
    explanation = await detection_service.get_detailed_explanation(detection_id, user_id)
    
    if not explanation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found or explanation not available"
        )
    
    return explanation
