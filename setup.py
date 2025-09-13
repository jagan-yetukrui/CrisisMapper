#!/usr/bin/env python3
"""
Setup script for CrisisMapper

AI-Powered Disaster Detection & Geospatial Analytics Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="crisismapper",
    version="2.0.0",
    author="CrisisMapper Team",
    author_email="team@crisismapper.com",
    description="AI-powered disaster detection from satellite and drone imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/crisismapper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crisismapper-api=inference_api:main",
            "crisismapper-dashboard=dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ai",
        "machine-learning",
        "computer-vision",
        "disaster-detection",
        "satellite-imagery",
        "geospatial",
        "yolov8",
        "opencv",
        "fastapi",
        "streamlit",
    ],
)