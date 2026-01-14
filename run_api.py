#!/usr/bin/env python3
"""
Convenience script to run the Forensic RAG FastAPI server.

Usage:
    python run_api.py              # Start with defaults
    python run_api.py --reload     # Start with auto-reload (dev mode)
    python run_api.py --port 8080  # Use custom port

Environment variables:
    API_HOST    - Host to bind to (default: 0.0.0.0)
    API_PORT    - Port to listen on (default: 8000)
    API_KEYS    - Comma-separated API keys for authentication
    
Example:
    API_KEYS="my-secret-key" python run_api.py
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)


def main():
    parser = argparse.ArgumentParser(description="Run Forensic RAG API Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()
    
    # Import after path setup
    import uvicorn
    from config.settings import API_HOST, API_PORT, ENVIRONMENT
    
    # Use args or fall back to config
    host = args.host or API_HOST
    port = args.port or API_PORT
    reload = args.reload or (ENVIRONMENT == "dev")
    
    print("\n" + "="*60)
    print("🚀 STARTING FORENSIC RAG API SERVER")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Workers: {args.workers}")
    print(f"Environment: {ENVIRONMENT}")
    print("="*60)
    print(f"\n📖 API Docs: http://{host}:{port}/docs")
    print(f"❤️  Health:   http://{host}:{port}/api/v1/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=args.workers if not reload else 1,  # Can't use workers with reload
        log_level="info"
    )


if __name__ == "__main__":
    main()
