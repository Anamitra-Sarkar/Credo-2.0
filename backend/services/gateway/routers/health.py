"""
Health Check Router for VeriFy AI.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    service: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="operational",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        service="VeriFy AI Gateway"
    )
