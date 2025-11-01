"""
Database models for VeriFy AI platform.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, 
    Integer, JSON, String, Text, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """User roles."""
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


class DetectionType(str, enum.Enum):
    """Type of detection performed."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    VOICE = "voice"


class DetectionVerdict(str, enum.Enum):
    """Detection verdict labels."""
    REAL = "real"
    FAKE = "fake"
    UNVERIFIED = "unverified"


class JobStatus(str, enum.Enum):
    """Async job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Preferences
    preferred_language = Column(String(10), default="en", nullable=False)
    
    # Relationships
    detections = relationship("Detection", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
    )


class Detection(Base):
    """Detection result model."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Detection metadata
    detection_type = Column(Enum(DetectionType), nullable=False, index=True)
    verdict = Column(Enum(DetectionVerdict), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
    # Input data
    input_text = Column(Text, nullable=True)
    input_language = Column(String(10), nullable=True)
    file_url = Column(String(1024), nullable=True)
    file_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    
    # Model info
    model_used = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    # Results
    explanation = Column(Text, nullable=True)
    detailed_results = Column(JSON, nullable=True)  # Store logits, probabilities, etc.
    
    # Metadata
    processing_time_ms = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    reports = relationship("Report", back_populates="detection", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_detection_type_verdict", "detection_type", "verdict"),
        Index("idx_detection_created", "created_at"),
        Index("idx_detection_file_hash", "file_hash"),
    )


class VideoJob(Base):
    """Video processing job model."""
    __tablename__ = "video_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Job status
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    progress = Column(Float, default=0.0, nullable=False)
    
    # Video info
    file_url = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Results (populated when completed)
    verdict = Column(Enum(DetectionVerdict), nullable=True)
    confidence = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("idx_video_job_id", "job_id"),
        Index("idx_video_status", "status"),
    )


class Report(Base):
    """User-submitted content report model."""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    detection_id = Column(Integer, ForeignKey("detections.id", ondelete="CASCADE"), nullable=True)
    
    # Report details
    content_url = Column(String(1024), nullable=True)
    content_text = Column(Text, nullable=True)
    report_reason = Column(String(500), nullable=False)
    report_category = Column(String(100), nullable=True)  # e.g., "politics", "health", "celebrity"
    
    # Geographic data
    user_location = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    country_code = Column(String(2), nullable=True, index=True)
    
    # Metadata
    verified = Column(Boolean, default=False, nullable=False)
    verified_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="reports")
    detection = relationship("Detection", back_populates="reports")
    
    __table_args__ = (
        Index("idx_report_country", "country_code"),
        Index("idx_report_created", "created_at"),
        Index("idx_report_category", "report_category"),
    )


class TrendingTopic(Base):
    """Trending misinformation topic model."""
    __tablename__ = "trending_topics"

    id = Column(Integer, primary_key=True, index=True)
    
    # Topic info
    topic_name = Column(String(500), nullable=False)
    topic_category = Column(String(100), nullable=True, index=True)
    keywords = Column(JSON, nullable=True)  # List of keywords
    
    # Metrics
    report_count = Column(Integer, default=1, nullable=False)
    fake_ratio = Column(Float, nullable=True)  # Ratio of fake to total detections
    
    # Geographic distribution
    country_distribution = Column(JSON, nullable=True)  # {"US": 45, "IN": 30, ...}
    
    # Time window
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_trending_category", "topic_category"),
        Index("idx_trending_window", "window_start", "window_end"),
    )


class APIKey(Base):
    """API key model for service-to-service authentication."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    service_name = Column(String(100), nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Limits
    rate_limit_per_minute = Column(Integer, nullable=True)
    rate_limit_per_hour = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("idx_api_key", "key"),
    )
