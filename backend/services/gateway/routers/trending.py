"""
Trending Router for VeriFy AI.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter()

class TrendingTopic(BaseModel):
    id: int
    title: str
    category: str
    fake_count: int
    real_count: int
    total_checks: int
    trending_score: float
    created_at: str

@router.get("/trending", response_model=List[TrendingTopic])
async def get_trending_topics(limit: int = 10):
    """Get trending fake news topics."""
    # TODO: Implement actual trending logic from database
    # Mock data for now
    return [
        TrendingTopic(
            id=1,
            title="Climate Change Hoax Claims",
            category="politics",
            fake_count=150,
            real_count=50,
            total_checks=200,
            trending_score=0.85,
            created_at=datetime.utcnow().isoformat()
        ),
        TrendingTopic(
            id=2,
            title="Vaccine Misinformation",
            category="health",
            fake_count=200,
            real_count=100,
            total_checks=300,
            trending_score=0.75,
            created_at=datetime.utcnow().isoformat()
        )
    ]
