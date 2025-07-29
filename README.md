# Stereo Vision Depth Estimation Project

This project implements stereo vision for depth estimation using OpenCV and MediaPipe.

## Setup

### Prerequisites
- Conda (Anaconda or Miniconda)
- Python 3.11+ (will be installed in conda environment)

### Installation

1. **Quick Setup (recommended):**
   ```bash
   ./setup.sh
   ```

2. **Manual Setup:**
   ```bash
   # Create conda environment
   conda env create -f environment.yml
   
   # Or create environment and install packages manually
   conda create -n stereo_vision python=3.11 -y
   conda activate stereo_vision
   pip install -r requirements.txt
   ```

### Activation
```bash
conda activate stereo_vision
```

## Project Structure

```
├── stereo_calibration.py     # Camera calibration script
├── stereoVision.py          # Main stereo vision application
├── calibration.py           # Calibration utilities
├── triangulation.py         # Triangulation and depth computation
├── images/                  # Calibration images
│   ├── stereoLeft/         # Left camera images
│   └── stereoRight/        # Right camera images
├── stereo_map.xml          # Generated calibration parameters
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment specification
└── setup.sh               # Environment setup script
```

## Usage

1. **Calibrate your stereo cameras:**
   ```bash
   conda activate stereo_vision
   python stereo_calibration.py
   ```

2. **Run stereo vision depth estimation:**
   ```bash
   conda activate stereo_vision
   python stereoVision.py
   ```

## Dependencies

- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **MediaPipe**: Hand/pose detection (if used)
- **imutils**: Image processing utilities
- **SciPy**: Scientific computing

## Calibration

The project uses chessboard calibration with a **5x10** inner corner pattern (6x11 squares). Make sure your calibration images contain clear, well-lit chessboard patterns. The rectangle size is 25 mm.

## Troubleshooting

1. **MediaPipe installation issues**: Make sure you're using Python 3.11 or 3.12 (not 3.13)
2. **Calibration failing**: Verify chessboard size matches your actual calibration pattern
3. **Import errors**: Make sure the conda environment is activated

## Notes

- The calibration parameters are saved in `stereo_map.xml`
- Both `calibration.py` and `triangulation.py` modules were created to support the main application
- The project is configured to work with Python 3.11 for maximum compatibility
