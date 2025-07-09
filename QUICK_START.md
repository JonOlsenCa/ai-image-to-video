# 🚀 Quick Start Guide

## One-Click Launch Options

### Option 1: Backend Only (Recommended)
```bash
start_ai_video.bat
```
- ✅ Fastest startup
- ✅ All AI features included
- ✅ Web interface at http://localhost:8000
- ✅ NSFW model support

### Option 2: Full Stack (Backend + Frontend)
```bash
start_full_stack.bat
```
- ✅ Complete React frontend
- ✅ Backend API server
- ✅ Frontend at http://localhost:3000
- ✅ Backend at http://localhost:8000

## What Happens on First Run

1. **Virtual Environment Setup** - Creates isolated Python environment
2. **Dependency Installation** - Installs all required packages
3. **Model Download** - Downloads AI models (~10GB) on first generation
4. **Server Start** - Launches the web interface

## Features Available

### 🎬 Video Generation
- **Image-to-Video**: Upload image → Generate video
- **Text-to-Video**: Text prompt → Generate video (NSFW model)

### 🎨 Image Generation
- **Text-to-Image**: Generate standalone images (NSFW model)

### ⚙️ Model Configuration
- **Runtime Switching**: Switch between NSFW and standard models
- **Automatic Fallback**: Falls back if models fail

## API Quick Reference

```bash
# Enable NSFW model
curl -X POST "http://localhost:8000/configure-nsfw" \
     -d '{"enable_nsfw": true}'

# Generate video from text
curl -X POST "http://localhost:8000/generate-video" \
     -F "prompt=your prompt" -F "duration=10"

# Generate image only
curl -X POST "http://localhost:8000/generate-image" \
     -d '{"prompt": "your prompt"}'

# Check model status
curl "http://localhost:8000/model-info"
```

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.8+
- **GPU**: NVIDIA with 8GB+ VRAM (recommended)
- **Storage**: 15GB free space (for models)
- **RAM**: 16GB recommended

## Troubleshooting

### Common Issues
1. **Python not found**: Install Python 3.8+ from python.org
2. **CUDA errors**: Install NVIDIA drivers and CUDA toolkit
3. **Out of memory**: Close other GPU applications
4. **Slow generation**: First run downloads models (~10GB)

### Getting Help
- Check `temp/server.log` for error details
- Run `python tests/test_nsfw_integration_simple.py` to test setup
- View API docs at `http://localhost:8000/docs`

## Next Steps

1. **Start the app**: Run `start_ai_video.bat`
2. **Open browser**: Go to `http://localhost:8000`
3. **Test generation**: Try image-to-video first
4. **Enable NSFW**: Use `/configure-nsfw` for text-to-video
5. **Explore API**: Check `/docs` for full API reference

---

**🎉 You're ready to create amazing AI videos!**
