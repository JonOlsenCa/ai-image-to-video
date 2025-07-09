# NSFW Model Integration

This document describes the integration of the UnfilteredAI/NSFW-gen-v2 model into the AI Image-to-Video codebase.

## Overview

The NSFW-gen-v2 model has been integrated to provide uncensored text-to-image generation capabilities, which can then be converted to videos using the existing Stable Video Diffusion pipeline.

## Features Added

### 1. NSFW Text-to-Video Generator (`nsfw_text_to_video_generator.py`)
- **Text-to-Image**: Uses UnfilteredAI/NSFW-gen-v2 for uncensored image generation
- **Image-to-Video**: Uses Stable Video Diffusion for video conversion
- **Progress Tracking**: Full integration with the existing progress management system
- **Image-Only Generation**: Can generate standalone images without video conversion

### 2. Enhanced Video Generator (`enhanced_video_generator.py`)
- **Model Switching**: Runtime switching between NSFW and standard models
- **Backward Compatibility**: Maintains all existing functionality
- **Automatic Fallback**: Falls back to standard models if NSFW model fails
- **Progress Integration**: Seamless integration with existing progress tracking

### 3. New API Endpoints

#### `/model-info` (GET)
Get information about the current model configuration:
```json
{
  "generator_class": "EnhancedVideoGenerator",
  "supports_nsfw": true,
  "nsfw_model_enabled": false,
  "supports_text_to_video": false,
  "supports_image_to_video": true
}
```

#### `/configure-nsfw` (POST)
Enable or disable NSFW model:
```json
{
  "enable_nsfw": true
}
```

#### `/generate-image` (POST)
Generate standalone images using NSFW model:
```json
{
  "prompt": "your prompt here",
  "width": 1024,
  "height": 576
}
```

#### `/download-image/{job_id}` (GET)
Download generated images.

## Usage

### 1. Basic Text-to-Video with NSFW Model

```python
from services.enhanced_video_generator import EnhancedVideoGenerator

# Initialize with NSFW model enabled
generator = EnhancedVideoGenerator(use_nsfw_model=True)

# Generate video from text (no image required)
await generator.generate_video(
    prompt="your uncensored prompt here",
    output_path="output.mp4",
    duration_seconds=10
)
```

### 2. Image-Only Generation

```python
# Generate just an image
await generator.generate_image_only(
    prompt="your prompt here",
    output_path="image.png",
    width=1024,
    height=576
)
```

### 3. Runtime Model Switching

```python
# Switch to NSFW model
generator.switch_model(use_nsfw=True)

# Switch back to standard model
generator.switch_model(use_nsfw=False)
```

### 4. API Usage

```bash
# Enable NSFW model
curl -X POST "http://localhost:8000/configure-nsfw" \
     -H "Content-Type: application/json" \
     -d '{"enable_nsfw": true}'

# Generate text-to-video (no image upload needed)
curl -X POST "http://localhost:8000/generate-video" \
     -F "prompt=your prompt here" \
     -F "duration=10"

# Generate image only
curl -X POST "http://localhost:8000/generate-image" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "your prompt here", "width": 1024, "height": 576}'
```

## Model Information

### NSFW-gen-v2 Specifications
- **Base Model**: Stable Diffusion XL
- **Parameters**: 3.47 billion
- **Tensor Type**: FP16 for efficiency
- **Features**: 
  - Uncensored output
  - 3D style rendering (use "3d" or "3d style" in prompts)
  - High-quality detailed generation

### Performance Considerations
- **VRAM Requirements**: ~8GB minimum, 16GB recommended
- **Generation Time**: 
  - Image: ~10-30 seconds
  - Video: ~1-3 minutes (depending on duration)
- **Quality**: Enhanced with automatic prompt optimization

## Safety and Legal Considerations

⚠️ **Important Notice**: 
- This model generates uncensored content
- Use responsibly and in compliance with local laws
- Age restriction applies (18+ only)
- Not suitable for all audiences

## Installation and Setup

### 1. Dependencies
The NSFW model uses the same dependencies as the existing system:
- PyTorch with CUDA support
- Diffusers
- Transformers
- Safetensors

### 2. Model Download
The model will be automatically downloaded on first use (~3.5GB).

### 3. Testing
Run the integration test:
```bash
cd backend
python test_nsfw_integration.py
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space (~4GB)
   - Verify Hugging Face access

2. **CUDA Out of Memory**
   - Reduce image resolution
   - Close other GPU applications
   - Use CPU fallback (slower)

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration

### Debug Information
Use the `/model-info` endpoint to check current configuration and troubleshoot issues.

## Integration Architecture

```
Text Prompt
     ↓
NSFW-gen-v2 (Text → Image)
     ↓
Stable Video Diffusion (Image → Video)
     ↓
Final Video Output
```

The integration maintains the existing architecture while adding the NSFW text-to-image capability as an optional component.

## Future Enhancements

Potential improvements:
- Multiple NSFW model support
- Custom model fine-tuning
- Advanced prompt engineering
- Batch processing capabilities
- Content filtering options
