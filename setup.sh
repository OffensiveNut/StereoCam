#!/bin/bash

# Stereo Vision Environment Setup Script

echo "Setting up Stereo Vision Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "stereo_vision"; then
    echo "✓ stereo_vision environment already exists"
else
    echo "Creating stereo_vision environment..."
    conda env create -f environment.yml
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate stereo_vision"
echo ""
echo "To run your stereo vision code:"
echo "  conda activate stereo_vision"
echo "  python stereoVision.py"
echo ""
echo "To install additional packages:"
echo "  conda activate stereo_vision"
echo "  pip install <package_name>"
echo ""
echo "Files created:"
echo "  - requirements.txt (pip requirements)"
echo "  - environment.yml (conda environment file)"
echo "  - calibration.py (calibration utilities)"
echo "  - triangulation.py (triangulation and depth computation)"
