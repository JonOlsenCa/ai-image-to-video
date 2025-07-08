# AI Image to Video Generator

A local GPU-accelerated web application that converts static images into dynamic videos using Stable Video Diffusion.

## Features

- Local GPU acceleration for fast video generation
- Web-based interface with drag-and-drop image upload
- Customizable video parameters (duration, motion intensity, noise strength)
- Real-time GPU status monitoring
- Video preview and download

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- Node.js 14+
- CUDA 11.8 or later
- At least 8GB VRAM (16GB recommended)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. In a new terminal, start the frontend:
```bash
cd frontend
npm start
```

3. Open http://localhost:3000 in your browser

## Usage

1. The app will show your GPU status at the top
2. Drag and drop an image or click to select
3. Adjust video settings:
   - Duration: Length of the video (1-10 seconds)
   - Motion Intensity: Controls the amount of movement (1-255)
   - Noise Strength: Adds variation to the generation (0-0.1)
4. Click "Generate Video" and wait for processing
5. Preview and download your generated video

## Model Information

This app uses Stable Video Diffusion (SVD) from Stability AI. The model will be downloaded automatically on first use (~5GB).

## Troubleshooting

- If GPU is not detected, ensure CUDA is properly installed
- For out of memory errors, try reducing the video duration or closing other GPU applications
- The first generation will be slower as the model needs to be downloaded and loaded