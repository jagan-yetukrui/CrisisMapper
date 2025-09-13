#!/usr/bin/env python3
"""
Script to run the CrisisMapper FastAPI server.
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Run the FastAPI server."""
    try:
        # Run uvicorn server
        subprocess.run([
            "uvicorn", "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nAPI server stopped by user")
    except Exception as e:
        print(f"Failed to run API server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
