#!/usr/bin/env python3
"""
Script to run the CrisisMapper Streamlit dashboard.
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Run the Streamlit dashboard."""
    try:
        # Run streamlit dashboard
        subprocess.run([
            "streamlit", "run", 
            "src/visualization/dashboard.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Failed to run dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
