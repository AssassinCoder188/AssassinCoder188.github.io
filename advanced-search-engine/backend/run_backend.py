#!/usr/bin/env python3
import uvicorn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
