# 📁 Project Structure

This document outlines the organized structure of the AI Image-to-Video Generator project.

## 🗂️ Root Directory

```
ai-image-to-video/
├── start_ai_video.bat          # 🚀 Main launcher script
├── README.md                   # 📖 Project documentation
├── backend/                    # 🔧 Backend server code
├── frontend/                   # 🌐 Frontend web interface
├── docs/                       # 📚 Documentation files
├── tests/                      # 🧪 Test files
├── scripts/                    # 🔨 Utility scripts
└── temp/                       # 🗂️ Temporary files
```

## 🔧 Backend Structure

```
backend/
├── main.py                     # FastAPI server entry point
├── run_server.py              # Alternative server launcher
├── requirements.txt           # Python dependencies
├── requirements_*.txt         # Specialized requirements
├── venv/                      # Virtual environment
├── services/                  # AI model services
│   ├── __init__.py
│   ├── nsfw_text_to_video_generator.py    # NSFW model service
│   ├── enhanced_video_generator.py        # Enhanced generator with NSFW
│   ├── stable_video_generator.py          # Stable Video Diffusion
│   ├── animatediff_generator.py           # AnimateDiff model
│   ├── dynamicrafter_generator.py         # DynamiCrafter model
│   ├── simple_video_generator.py          # Fallback generator
│   ├── text_to_video_generator.py         # Text-to-video base
│   ├── progress_manager.py                # Progress tracking
│   └── ...
├── models/                    # Downloaded AI models
├── outputs/                   # Generated videos
├── uploads/                   # Uploaded images
└── __pycache__/              # Python cache
```

## 🌐 Frontend Structure

```
frontend/
├── package.json               # Node.js dependencies
├── package-lock.json         # Dependency lock file
├── node_modules/             # Node.js packages
├── public/                   # Static files
│   ├── index.html
│   └── ...
└── src/                      # React source code
    ├── components/
    ├── pages/
    ├── styles/
    └── ...
```

## 📚 Documentation

```
docs/
├── PROJECT_STRUCTURE.md      # This file
├── NSFW_INTEGRATION.md       # NSFW model integration guide
├── CLAUDE.md                 # Development notes
└── API_REFERENCE.md          # API documentation (future)
```

## 🧪 Tests

```
tests/
├── test_nsfw_integration.py          # Full NSFW integration test
├── test_nsfw_integration_simple.py   # Simple import/setup test
├── test_imageio.py                   # Video encoding test
├── test_server.py                    # Server functionality test
├── test_stable_generator.py          # Stable Video Diffusion test
└── test_video_generation.py          # General video generation test
```

## 🔨 Scripts

```
scripts/
├── git_commit_push.bat       # Git automation script
├── start_wsl.bat            # WSL launcher
└── ...                      # Other utility scripts
```

## 🗂️ Temporary Files

```
temp/
├── commit_message.txt        # Temporary commit messages
├── server.log               # Old server logs
└── ...                      # Other temporary files
```

## 🎯 Key Files

### **Main Entry Points**
- `start_ai_video.bat` - Primary launcher for the entire application
- `backend/main.py` - FastAPI server with all endpoints
- `frontend/src/App.js` - React frontend application

### **Core Services**
- `backend/services/nsfw_text_to_video_generator.py` - NSFW-gen-v2 integration
- `backend/services/enhanced_video_generator.py` - Enhanced generator with model switching
- `backend/services/stable_video_generator.py` - Stable Video Diffusion implementation

### **Configuration**
- `backend/requirements.txt` - Python dependencies
- `frontend/package.json` - Node.js dependencies

### **Documentation**
- `README.md` - Main project documentation
- `docs/NSFW_INTEGRATION.md` - NSFW model integration details

## 🔄 Data Flow

```
User Input → Frontend → Backend API → AI Services → Model Processing → Output
     ↑                                      ↓
     └── Progress Updates ← WebSocket ← Progress Manager
```

## 📦 Model Storage

Models are automatically downloaded to:
- `backend/models/` - Local model cache
- `~/.cache/huggingface/` - Hugging Face model cache

## 🚀 Quick Navigation

- **Start the app**: Run `start_ai_video.bat`
- **View logs**: Check `backend/server.log` (if exists)
- **Test integration**: Run `python tests/test_nsfw_integration_simple.py`
- **API docs**: Visit `http://localhost:8000/docs` when server is running
- **Generated content**: Check `backend/outputs/` and `backend/uploads/`

## 🧹 Cleanup

The project structure has been organized to separate:
- **Production code** (backend/, frontend/)
- **Documentation** (docs/)
- **Testing** (tests/)
- **Utilities** (scripts/)
- **Temporary files** (temp/)

This organization makes the project easier to navigate, maintain, and deploy.
