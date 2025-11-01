"""
Main FastAPI Gateway Application for VeriFy AI.
Entry point for all API requests.
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from shared.config import settings
from shared.database.session import init_db, close_db
from shared.monitoring.logging import setup_logging, logger

# Import routers
from .routers import auth, detection, report, trending, health


# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting VeriFy AI API Gateway...")
    setup_logging()
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VeriFy AI API Gateway...")
    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="VeriFy AI API",
    description="Production API for multilingual deepfake and fake news detection",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)


# Request timing and metrics middleware
@app.middleware("http")
async def add_process_time_and_metrics(request: Request, call_next):
    """Add request timing and Prometheus metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred" if settings.is_production else str(exc)
        },
    )


# Include routers
app.include_router(health.router, prefix=f"/api/{settings.api_version}", tags=["Health"])
app.include_router(auth.router, prefix=f"/api/{settings.api_version}/auth", tags=["Authentication"])
app.include_router(detection.router, prefix=f"/api/{settings.api_version}", tags=["Detection"])
app.include_router(report.router, prefix=f"/api/{settings.api_version}", tags=["Reports"])
app.include_router(trending.router, prefix=f"/api/{settings.api_version}", tags=["Trending"])


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "service": "VeriFy AI API Gateway",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs" if not settings.is_production else "Contact admin for API docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.is_production,
        log_config=None  # Use our custom logging
    )
