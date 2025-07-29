# Stereo Vision Depth Estimation Project

This project implements stereo vision for depth estimation using OpenCV and YOLO object detection.

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
- **Ultralytics YOLO**: Object detection model
- **PyTorch**: Deep learning backend for YOLO
- **imutils**: Image processing utilities
- **SciPy**: Scientific computing

## Features

- **Real-time object detection**: Uses YOLOv8 nano model for fast and accurate object detection
- **Stereo depth estimation**: Calculates distance to detected objects using stereo vision
- **Confidence filtering**: Only processes objects with >51% confidence threshold
- **Top-5 detection**: Shows the 5 most confident detections per frame
- **Multi-object support**: Can detect and track multiple object types simultaneously

## Calibration

The project uses chessboard calibration with a **5x10** inner corner pattern (6x11 squares). Make sure your calibration images contain clear, well-lit chessboard patterns. The rectangle size is 25 mm.

## Troubleshooting

1. **YOLO model download**: On first run, YOLOv8 model will be automatically downloaded
2. **PyTorch installation**: If conda fails to install PyTorch, it will be installed via pip
3. **Calibration failing**: Verify chessboard size matches your actual calibration pattern
4. **Import errors**: Make sure the conda environment is activated
5. **Low detection accuracy**: Adjust confidence threshold in stereoVision.py (currently set to 51%)

## Notes

- The calibration parameters are saved in `stereo_map.xml`
- Both `calibration.py` and `triangulation.py` modules were created to support the main application
- The project uses YOLOv8 nano model for optimal speed/accuracy balance
- YOLO model weights are automatically downloaded on first run
- The system filters detections to show only objects with >51% confidence
- Supports detection of 80+ object classes (COCO dataset)
