from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch
import os
import sys
import uuid
from pathlib import Path
import shutil
import asyncio
# Try to import video generators in order of preference
video_generator_class = None
video_generator = None

try:
    from services.enhanced_video_generator import EnhancedVideoGenerator
    video_generator_class = EnhancedVideoGenerator
    print("🎨 Using Enhanced Video Generator with NSFW support")
except Exception as e:
    print(f"Enhanced Video Generator not available: {e}")
    try:
        from services.stable_video_generator import StableVideoGenerator
        video_generator_class = StableVideoGenerator
        print("🤖 Using Stable Video Generator - Advanced person animation AI")
    except Exception as e2:
        print(f"Stable Video Generator not available: {e2}")
        try:
            from services.simple_video_generator import SimpleVideoGenerator
            video_generator_class = SimpleVideoGenerator
            print("📹 Using simple video generator with 12 animations")
        except Exception as e3:
            print(f"Simple Video Generator not available: {e3}")
            print("⚠️ No video generators available - API will have limited functionality")

# Create a dummy generator if none are available
class DummyVideoGenerator:
    def __init__(self):
        self.device = "cpu"

    async def generate_video(self, *args, **kwargs):
        raise HTTPException(status_code=503, detail="No video generators available. Please install dependencies.")

    def cleanup(self):
        pass

if video_generator_class is None:
    video_generator_class = DummyVideoGenerator
    print("🔧 Using dummy generator - install dependencies for full functionality")
from services.progress_manager import progress_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize video generator safely
try:
    video_generator = video_generator_class()
    print(f"✅ Video generator initialized: {type(video_generator).__name__}")
except Exception as e:
    print(f"⚠️ Failed to initialize video generator: {e}")
    video_generator = DummyVideoGenerator()
    print("🔧 Using dummy generator")

@app.get("/")
async def root():
    from fastapi.responses import HTMLResponse

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Image-to-Video Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .header { text-align: center; margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .section { margin: 20px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #007bff; border-radius: 8px; padding: 40px; text-align: center; margin: 20px 0; cursor: pointer; transition: all 0.3s; }
            .upload-area:hover { border-color: #0056b3; background: #f8f9fa; }
            .upload-area.dragover { border-color: #28a745; background: #d4edda; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            .form-group textarea { height: 80px; resize: vertical; }
            .button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
            .button:hover { background: #0056b3; }
            .button:disabled { background: #6c757d; cursor: not-allowed; }
            .button.secondary { background: #6c757d; }
            .button.secondary:hover { background: #545b62; }
            .status { padding: 15px; margin: 15px 0; border-radius: 4px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .progress { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-bar { height: 100%; background: #007bff; transition: width 0.3s; }
            .tabs { display: flex; margin-bottom: 20px; }
            .tab { padding: 10px 20px; background: #e9ecef; border: none; cursor: pointer; border-radius: 4px 4px 0 0; margin-right: 5px; }
            .tab.active { background: #007bff; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            @media (max-width: 768px) { .two-column { grid-template-columns: 1fr; } }
            .result-area { min-height: 200px; border: 1px solid #ddd; border-radius: 4px; padding: 20px; text-align: center; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎬 AI Image-to-Video Generator</h1>
            <p>Transform images into videos with AI • Generate videos from text prompts</p>
            <div class="status info">
                <strong>GPU:</strong> """ + ("✅ " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ CPU Only") + """ •
                <strong>Generator:</strong> """ + type(video_generator).__name__ + """
            </div>
        </div>

        <div class="section">
            <div class="tabs">
                <button class="tab active" onclick="showTab('image-to-video')">📷 Image-to-Video</button>
                <button class="tab" onclick="showTab('text-to-video')">📝 Text-to-Video</button>
                <button class="tab" onclick="showTab('image-only')">🎨 Image Only</button>
                <button class="tab" onclick="showTab('settings')">⚙️ Settings</button>
            </div>

            <!-- Image-to-Video Tab -->
            <div id="image-to-video" class="tab-content active">
                <h2>📷 Image-to-Video Generation</h2>
                <form id="imageToVideoForm" enctype="multipart/form-data">
                    <div class="upload-area" onclick="document.getElementById('imageFile').click()">
                        <p>🖼️ Click to upload an image or drag & drop</p>
                        <p style="color: #666; font-size: 14px;">Supported: JPG, PNG, GIF</p>
                        <input type="file" id="imageFile" name="image" accept="image/*" style="display: none;" onchange="handleImageUpload(this)">
                    </div>
                    <div id="imagePreview" class="hidden">
                        <img id="previewImg" style="max-width: 300px; max-height: 200px; border-radius: 4px;">
                        <p id="imageInfo"></p>
                    </div>

                    <div class="two-column">
                        <div class="form-group">
                            <label for="prompt">Prompt (optional):</label>
                            <textarea id="prompt" name="prompt" placeholder="Describe the motion you want (e.g., 'gentle swaying', 'dancing', 'walking')"></textarea>
                        </div>
                        <div>
                            <div class="form-group">
                                <label for="duration">Duration (seconds):</label>
                                <input type="range" id="duration" name="duration" min="1" max="10" value="5" oninput="updateDurationLabel(this.value)">
                                <span id="durationLabel">5 seconds</span>
                            </div>
                            <div class="form-group">
                                <label for="motion">Motion Intensity:</label>
                                <select id="motion" name="motion_bucket_id">
                                    <option value="127">Normal (127)</option>
                                    <option value="100">Gentle (100)</option>
                                    <option value="150">Strong (150)</option>
                                    <option value="200">Very Strong (200)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <button type="submit" class="button">🎬 Generate Video</button>
                </form>
            </div>

            <!-- Text-to-Video Tab -->
            <div id="text-to-video" class="tab-content">
                <h2>📝 Text-to-Video Generation</h2>
                <div class="status warning">
                    <strong>⚠️ NSFW Model Required:</strong> Enable NSFW model in Settings tab for text-to-video generation
                </div>
                <form id="textToVideoForm">
                    <div class="form-group">
                        <label for="textPrompt">Text Prompt:</label>
                        <textarea id="textPrompt" name="prompt" placeholder="Describe the video you want to generate (e.g., 'beautiful landscape with flowing water', 'person walking in a park')" required></textarea>
                    </div>

                    <div class="two-column">
                        <div class="form-group">
                            <label for="textDuration">Duration (seconds):</label>
                            <input type="range" id="textDuration" name="duration" min="1" max="10" value="5" oninput="updateTextDurationLabel(this.value)">
                            <span id="textDurationLabel">5 seconds</span>
                        </div>
                        <div class="form-group">
                            <label for="textMotion">Motion Intensity:</label>
                            <select id="textMotion" name="motion_bucket_id">
                                <option value="127">Normal (127)</option>
                                <option value="100">Gentle (100)</option>
                                <option value="150">Strong (150)</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="button">🎬 Generate Video from Text</button>
                </form>
            </div>

            <!-- Image Only Tab -->
            <div id="image-only" class="tab-content">
                <h2>🎨 Image-Only Generation</h2>
                <div class="status warning">
                    <strong>⚠️ NSFW Model Required:</strong> Enable NSFW model in Settings tab for image generation
                </div>
                <form id="imageOnlyForm">
                    <div class="form-group">
                        <label for="imagePrompt">Image Prompt:</label>
                        <textarea id="imagePrompt" name="prompt" placeholder="Describe the image you want to generate (e.g., 'beautiful portrait', 'landscape painting', 'artistic scene')" required></textarea>
                    </div>

                    <div class="two-column">
                        <div class="form-group">
                            <label for="imageWidth">Width:</label>
                            <select id="imageWidth" name="width">
                                <option value="1024">1024px</option>
                                <option value="768">768px</option>
                                <option value="512">512px</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="imageHeight">Height:</label>
                            <select id="imageHeight" name="height">
                                <option value="576">576px</option>
                                <option value="768">768px</option>
                                <option value="512">512px</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="button">🎨 Generate Image</button>
                </form>
            </div>

            <!-- Settings Tab -->
            <div id="settings" class="tab-content">
                <h2>⚙️ Settings</h2>

                <div class="form-group">
                    <label>
                        <input type="checkbox" id="nsfwEnabled" onchange="toggleNSFW(this.checked)">
                        Enable NSFW Model (Required for text-to-video and image-only generation)
                    </label>
                    <div class="status warning">
                        <strong>⚠️ Content Warning:</strong> NSFW model generates uncensored content. 18+ only.
                    </div>
                </div>

                <div class="form-group">
                    <button type="button" class="button secondary" onclick="checkModelInfo()">📊 Check Model Status</button>
                    <button type="button" class="button secondary" onclick="window.open('/docs', '_blank')">📚 API Documentation</button>
                </div>

                <div id="modelStatus" class="status info">
                    <strong>Current Status:</strong> Click "Check Model Status" to update
                </div>
            </div>
        </div>

        <!-- Progress Section -->
        <div id="progressSection" class="section hidden">
            <h3>🔄 Generation Progress</h3>
            <div class="progress">
                <div id="progressBar" class="progress-bar" style="width: 0%"></div>
            </div>
            <p id="progressText">Initializing...</p>
            <button type="button" class="button secondary" onclick="cancelGeneration()">❌ Cancel</button>
        </div>

        <!-- Results Section -->
        <div id="resultsSection" class="section">
            <h3>📁 Results</h3>
            <div id="resultArea" class="result-area">
                <p>Generated videos and images will appear here</p>
            </div>
        </div>

        <script>
            let currentJobId = null;
            let progressInterval = null;

            // Tab switching
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

                event.target.classList.add('active');
                document.getElementById(tabName).classList.add('active');
            }

            // Duration label updates
            function updateDurationLabel(value) {
                document.getElementById('durationLabel').textContent = value + ' seconds';
            }

            function updateTextDurationLabel(value) {
                document.getElementById('textDurationLabel').textContent = value + ' seconds';
            }

            // Image upload handling
            function handleImageUpload(input) {
                const file = input.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('previewImg').src = e.target.result;
                        document.getElementById('imageInfo').textContent = `${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
                        document.getElementById('imagePreview').classList.remove('hidden');
                    };
                    reader.readAsDataURL(file);
                }
            }

            // NSFW toggle
            async function toggleNSFW(enabled) {
                try {
                    const response = await fetch('/configure-nsfw', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enable_nsfw: enabled })
                    });
                    const result = await response.json();

                    if (result.status === 'success') {
                        showStatus(`NSFW model ${enabled ? 'enabled' : 'disabled'}`, 'success');
                    } else {
                        showStatus(`Failed to ${enabled ? 'enable' : 'disable'} NSFW model: ${result.message}`, 'error');
                    }
                } catch (error) {
                    showStatus('Error configuring NSFW model: ' + error.message, 'error');
                }
            }

            // Model status check
            async function checkModelInfo() {
                try {
                    const response = await fetch('/model-info');
                    const info = await response.json();

                    document.getElementById('modelStatus').innerHTML = `
                        <strong>Generator:</strong> ${info.generator_class || 'Unknown'}<br>
                        <strong>NSFW Support:</strong> ${info.supports_nsfw ? '✅ Available' : '❌ Not available'}<br>
                        <strong>GPU:</strong> ${info.gpu_available ? '✅ Available' : '❌ CPU only'}<br>
                        <strong>Text-to-Video:</strong> ${info.supports_text_to_video ? '✅ Ready' : '❌ Enable NSFW model'}
                    `;
                } catch (error) {
                    document.getElementById('modelStatus').innerHTML = `<strong>Error:</strong> ${error.message}`;
                }
            }

            // Form submissions
            document.getElementById('imageToVideoForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(this);
                if (!formData.get('image') || !formData.get('image').name) {
                    showStatus('Please select an image first', 'error');
                    return;
                }

                await submitGeneration('/generate-video', formData);
            });

            document.getElementById('textToVideoForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(this);
                await submitGeneration('/generate-video', formData);
            });

            document.getElementById('imageOnlyForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(this);
                await submitGeneration('/generate-image', formData);
            });

            // Generation submission
            async function submitGeneration(endpoint, formData) {
                try {
                    showProgress(true);

                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (result.status === 'success') {
                        currentJobId = result.job_id;
                        startProgressTracking();
                        showStatus('Generation started successfully!', 'success');
                    } else {
                        showProgress(false);
                        showStatus('Generation failed: ' + result.message, 'error');
                    }
                } catch (error) {
                    showProgress(false);
                    showStatus('Error: ' + error.message, 'error');
                }
            }

            // Progress tracking
            function startProgressTracking() {
                if (!currentJobId) return;

                progressInterval = setInterval(async () => {
                    try {
                        const response = await fetch(`/progress/${currentJobId}`);
                        const progress = await response.json();

                        updateProgress(progress.progress, progress.message);

                        if (progress.progress >= 100 || progress.progress < 0) {
                            clearInterval(progressInterval);
                            showProgress(false);

                            if (progress.progress >= 100) {
                                showResult(currentJobId, progress.download_url);
                                showStatus('Generation completed successfully!', 'success');
                            } else {
                                showStatus('Generation failed: ' + progress.message, 'error');
                            }
                        }
                    } catch (error) {
                        clearInterval(progressInterval);
                        showProgress(false);
                        showStatus('Error tracking progress: ' + error.message, 'error');
                    }
                }, 1000);
            }

            function updateProgress(percent, message) {
                document.getElementById('progressBar').style.width = Math.max(0, percent) + '%';
                document.getElementById('progressText').textContent = message || 'Processing...';
            }

            function showProgress(show) {
                document.getElementById('progressSection').classList.toggle('hidden', !show);
                if (!show) {
                    updateProgress(0, 'Ready');
                }
            }

            function cancelGeneration() {
                if (progressInterval) {
                    clearInterval(progressInterval);
                }
                showProgress(false);
                currentJobId = null;
                showStatus('Generation cancelled', 'warning');
            }

            // Results display
            function showResult(jobId, downloadUrl) {
                const resultArea = document.getElementById('resultArea');
                const isVideo = downloadUrl.includes('.mp4');

                resultArea.innerHTML = `
                    <div style="text-align: center;">
                        ${isVideo ?
                            `<video controls style="max-width: 100%; max-height: 400px;">
                                <source src="${downloadUrl}" type="video/mp4">
                                Your browser does not support video playback.
                            </video>` :
                            `<img src="${downloadUrl}" style="max-width: 100%; max-height: 400px;" alt="Generated image">`
                        }
                        <br><br>
                        <a href="${downloadUrl}" download class="button">📥 Download ${isVideo ? 'Video' : 'Image'}</a>
                    </div>
                `;
            }

            // Status messages
            function showStatus(message, type) {
                const statusDiv = document.createElement('div');
                statusDiv.className = `status ${type}`;
                statusDiv.textContent = message;

                document.body.insertBefore(statusDiv, document.body.firstChild);

                setTimeout(() => {
                    statusDiv.remove();
                }, 5000);
            }

            // Drag and drop
            const uploadArea = document.querySelector('.upload-area');

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('imageFile').files = files;
                    handleImageUpload(document.getElementById('imageFile'));
                }
            });

            // Initialize
            checkModelInfo();
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

@app.get("/test")
async def test():
    return {"status": "ok", "time": str(asyncio.get_event_loop().time())}

@app.get("/debug")
async def debug():
    return {
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "python_version": f"{sys.version}",
        "torch_version": torch.__version__,
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "output_dir_exists": OUTPUT_DIR.exists(),
        "active_jobs": len(progress_manager.active_jobs)
    }

@app.post("/generate-video")
async def generate_video(
    prompt: str = Form(""),
    duration: int = Form(10),
    motion_bucket_id: int = Form(127),
    image: UploadFile = File(None)
):
    """Generate video from text prompt and optional image"""
    print(f"Received request: prompt='{prompt}', duration={duration}, motion_bucket_id={motion_bucket_id}, has_image={image is not None}")

    # For image-to-video, prompt can be empty
    # For text-to-video, prompt is required
    if not image and (not prompt or len(prompt.strip()) == 0):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Either an image or a text prompt is required"
            }
        )
    
    job_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    image_path = None
    
    try:
        # Save uploaded image if provided
        if image:
            image_path = UPLOAD_DIR / f"{job_id}_input{Path(image.filename).suffix}"
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
        
        # Start progress tracking
        progress_manager.start_job(job_id, total_steps=100)
        
        # Generate video
        print(f"Starting video generation task for job {job_id}")
        print(f"Using generator: {type(video_generator).__name__}")
        print(f"Image path: {image_path}")
        print(f"Prompt: {prompt}")
        
        task = asyncio.create_task(video_generator.generate_with_progress(
            job_id=job_id,
            prompt=prompt,
            output_path=str(output_path),
            duration_seconds=duration,
            progress_manager=progress_manager,
            image_path=str(image_path) if image_path else None,
            motion_bucket_id=motion_bucket_id
        ))
        
        # Add error logging
        def log_task_exception(task):
            if task.exception():
                print(f"ERROR: Task failed with exception: {task.exception()}")
                import traceback
                traceback.print_exception(type(task.exception()), task.exception(), task.exception().__traceback__)
            else:
                print(f"SUCCESS: Task completed for job {job_id}")
        
        task.add_done_callback(log_task_exception)
        
        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "message": "Video generation started",
            "prompt": prompt,
            "duration": duration,
            "has_image": image is not None
        })
    
    except Exception as e:
        print(f"Error in generate_video: {str(e)}")
        import traceback
        traceback.print_exc()
        # Cleanup uploaded image on error
        if image_path and image_path.exists():
            image_path.unlink()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Video generation failed: {str(e)}",
                "job_id": job_id if 'job_id' in locals() else None
            }
        )


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    print(f"WebSocket connection attempt for job {job_id}")
    await websocket.accept()
    print(f"WebSocket accepted for job {job_id}")
    progress_manager.add_websocket(job_id, websocket)
    
    try:
        # Send initial status if job exists
        if job_id in progress_manager.active_jobs:
            print(f"Sending initial status for job {job_id}")
            await websocket.send_json({
                "type": "progress",
                "data": progress_manager.active_jobs[job_id]
            })
        else:
            print(f"No active job found for {job_id}")
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
        progress_manager.remove_websocket(job_id, websocket)

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get the progress of a video generation job"""
    if job_id in progress_manager.active_jobs:
        job_data = progress_manager.active_jobs[job_id]
        return {
            "progress": job_data.get("progress", 0),
            "message": job_data.get("message", "Processing..."),
            "download_url": job_data.get("download_url", None)
        }

    # Check if video exists (job might be completed)
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        return {
            "progress": 100,
            "message": "Generation completed",
            "download_url": f"/download-video/{job_id}"
        }

    # Job not found
    return {
        "progress": -1,
        "message": "Job not found or failed",
        "download_url": None
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a video generation job"""
    if job_id in progress_manager.active_jobs:
        return progress_manager.active_jobs[job_id]

    # Check if video exists (job might be completed)
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        return {
            "status": "completed",
            "progress": 100,
            "message": "Video ready for download"
        }
    
    return {
        "status": "not_found",
        "message": "Job not found"
    }

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"generated_{job_id}.mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/gpu-info")
async def gpu_info():
    if torch.cuda.is_available():
        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "cuda_version": torch.version.cuda
        }
    return {"gpu_available": False}

@app.get("/test-generate")
async def test_generate():
    """Test endpoint to quickly verify video generation works"""
    try:
        job_id = "test-" + str(uuid.uuid4())[:8]
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        
        print(f"Test generate: Creating video at {output_path}")
        
        # Generate a simple test video directly (bypassing progress manager)
        from services.test_video_generator import TestVideoGenerator
        test_gen = TestVideoGenerator()
        
        # Simple direct generation
        result = await test_gen.generate_video(
            prompt="Direct Test Video",
            output_path=str(output_path),
            duration_seconds=2,
            progress_callback=None,
            image_path=None
        )
        
        # Check if file was created
        if output_path.exists():
            file_size = output_path.stat().st_size
            return {
                "status": "success",
                "message": "Test video generated successfully",
                "job_id": job_id,
                "output_path": str(output_path),
                "file_size": file_size,
                "file_exists": True
            }
        else:
            return {
                "status": "error",
                "message": "Video file was not created",
                "output_path": str(output_path)
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/test-stable-generator")
async def test_stable_generator():
    """Test the stable video generator directly"""
    try:
        # Import and test
        print("Testing stable video generator...")
        
        # Check what generator is loaded
        generator_name = type(video_generator).__name__
        
        # Try to generate a test video
        test_image = UPLOAD_DIR / "test_debug.jpg"
        if not test_image.exists():
            # Create a test image
            from PIL import Image
            img = Image.new('RGB', (512, 512), color='blue')
            img.save(test_image)
        
        job_id = "debug-" + str(uuid.uuid4())[:8]
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        
        # Try direct generation
        try:
            await video_generator.generate_video(
                prompt="person walking",
                output_path=str(output_path),
                duration_seconds=2,
                progress_callback=None,
                image_path=str(test_image)
            )
            
            return {
                "status": "success",
                "generator": generator_name,
                "output_exists": output_path.exists(),
                "output_size": output_path.stat().st_size if output_path.exists() else 0
            }
        except Exception as gen_error:
            return {
                "status": "generation_error",
                "generator": generator_name,
                "error": str(gen_error),
                "traceback": traceback.format_exc()
            }
            
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/model-info")
async def get_model_info():
    """Get information about the current video generator and model configuration"""
    try:
        generator_info = {
            "generator_class": type(video_generator).__name__,
            "supports_nsfw": hasattr(video_generator, 'use_nsfw_model'),
            "gpu_available": torch.cuda.is_available()
        }

        # Get detailed model info if available
        if hasattr(video_generator, 'get_model_info'):
            generator_info.update(video_generator.get_model_info())

        return generator_info
    except Exception as e:
        return {
            "error": str(e),
            "generator_class": type(video_generator).__name__
        }

@app.post("/configure-nsfw")
async def configure_nsfw_model(request: dict):
    """Enable or disable NSFW model for text-to-video generation"""
    try:
        enable_nsfw = request.get("enable_nsfw", False)

        if hasattr(video_generator, 'switch_model'):
            video_generator.switch_model(use_nsfw=enable_nsfw)
            return {
                "status": "success",
                "nsfw_enabled": enable_nsfw,
                "message": f"NSFW model {'enabled' if enable_nsfw else 'disabled'}",
                "generator": type(video_generator).__name__
            }
        else:
            return {
                "status": "error",
                "message": f"Current generator ({type(video_generator).__name__}) does not support NSFW model switching"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to {'enable' if request.get('enable_nsfw', False) else 'disable'} NSFW model: {str(e)}"
        }

@app.post("/generate-image")
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 576
):
    """Generate an image from text prompt (requires NSFW model)"""
    try:
        if not hasattr(video_generator, 'generate_image_only'):
            raise HTTPException(status_code=400, detail="Image-only generation not supported by current generator")

        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        job_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{job_id}.png"

        # Generate image
        result = await video_generator.generate_image_only(
            prompt=prompt,
            output_path=str(output_path),
            width=width,
            height=height
        )

        return {
            "status": "success",
            "job_id": job_id,
            "download_url": f"/download-image/{job_id}",
            "message": "Image generated successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/download-image/{job_id}")
async def download_image(job_id: str):
    """Download generated image"""
    output_path = OUTPUT_DIR / f"{job_id}.png"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*"
        }
    )

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AI Image-to-Video Server...")
    print("📝 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print()

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        if 'video_generator' in globals():
            try:
                video_generator.cleanup()
            except:
                pass