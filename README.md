# 🎬 AI Image-to-Video Generator with NSFW Support

A powerful AI-powered tool that transforms static images into dynamic videos using advanced machine learning models, including **uncensored text-to-video generation** with NSFW-gen-v2.

## ✨ Features

### 🤖 **Multiple AI Models**
- **Stable Video Diffusion**: High-quality image-to-video generation
- **NSFW-gen-v2**: Uncensored text-to-image and text-to-video generation
- **AnimateDiff**: Animation-focused video generation
- **DynamiCrafter**: Advanced person/object animation
- **Simple Video Generator**: Fallback with 12 basic animations

### 🎯 **Generation Modes**
- **Image-to-Video**: Transform static images into dynamic videos
- **Text-to-Video**: Generate videos directly from text prompts (NSFW model)
- **Image-Only**: Generate standalone uncensored images

### 🌐 **Web Interface**
- Easy-to-use drag-and-drop interface
- Real-time progress tracking via WebSocket
- GPU status monitoring
- Model configuration controls

### ⚡ **Performance**
- GPU acceleration with NVIDIA CUDA
- Automatic model fallback system
- Memory optimization and CPU offloading
- Async processing for multiple requests

## 🚀 Quick Start

### **One-Click Launch**
Simply double-click `start_ai_video.bat` to automatically:
- Set up virtual environment
- Install dependencies
- Start the server
- Open the web interface

### **Manual Setup**

#### Prerequisites
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (recommended)
- Windows 10/11

#### Installation
```bash
git clone https://github.com/your-username/ai-image-to-video.git
cd ai-image-to-video
start_ai_video.bat
```

## 📖 Usage

### **Web Interface**
1. Open `http://localhost:8000` in your browser
2. **For Image-to-Video**: Upload an image and set parameters
3. **For Text-to-Video**: Enable NSFW model and enter text prompt
4. Click "Generate" and monitor progress
5. Download your generated content

### **API Usage**
```bash
# Enable NSFW model for text-to-video
curl -X POST "http://localhost:8000/configure-nsfw" \
     -H "Content-Type: application/json" \
     -d '{"enable_nsfw": true}'

# Generate video from text (no image needed)
curl -X POST "http://localhost:8000/generate-video" \
     -F "prompt=your text prompt here" \
     -F "duration=10"

# Generate image only
curl -X POST "http://localhost:8000/generate-image" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "your prompt here"}'
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/generate-video` | POST | Generate video from image or text |
| `/generate-image` | POST | Generate image only (NSFW) |
| `/configure-nsfw` | POST | Enable/disable NSFW model |
| `/model-info` | GET | Get current model information |
| `/progress/{job_id}` | GET | Get generation progress |
| `/download/{job_id}` | GET | Download generated video |
| `/download-image/{job_id}` | GET | Download generated image |
| `/ws/{job_id}` | WebSocket | Real-time progress updates |

## ⚠️ NSFW Model Notice

The NSFW-gen-v2 model provides uncensored content generation:
- **Age Restriction**: 18+ only
- **Legal Compliance**: Ensure compliance with local laws
- **Responsible Use**: Use ethically and responsibly
- **Content Warning**: Generates explicit/adult content

## 📁 Project Structure

```
ai-image-to-video/
├── start_ai_video.bat          # 🚀 One-click launcher
├── README.md                   # 📖 This file
├── backend/                    # 🔧 Server code
│   ├── main.py                # FastAPI server
│   ├── requirements.txt       # Dependencies
│   └── services/              # AI model services
├── frontend/                   # 🌐 Web interface
├── docs/                      # 📚 Documentation
├── tests/                     # 🧪 Test files
├── scripts/                   # 🔨 Utility scripts
└── temp/                      # 🗂️ Temporary files
```

## 🎛️ Configuration

The application automatically:
- Detects available GPU/CUDA
- Downloads models on first use (~10GB)
- Falls back gracefully between models
- Optimizes memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python tests/test_nsfw_integration_simple.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⚡ Ready to create amazing AI videos? Run `start_ai_video.bat` and get started!**