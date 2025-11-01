"""
Startup script for VeriFy AI Backend Gateway.
Run from backend directory: python start_gateway.py
"""
import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "services.gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
