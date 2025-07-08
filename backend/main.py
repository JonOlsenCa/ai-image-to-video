from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch
import os
import uuid
from pathlib import Path
import shutil
import asyncio
from services.simple_video_generator import SimpleVideoGenerator
from services.progress_manager import progress_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

video_generator = SimpleVideoGenerator()

@app.get("/")
async def root():
    return {"message": "AI Image to Video API", "gpu_available": torch.cuda.is_available()}

@app.get("/test")
async def test():
    return {"status": "ok", "time": str(asyncio.get_event_loop().time())}

@app.post("/generate-video")
async def generate_video(
    file: UploadFile = File(...),
    duration: int = 4,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        
        # Start progress tracking
        num_frames = min(duration * fps, 14)  # Limit to 14 frames (2 seconds) for faster generation
        progress_manager.start_job(job_id, total_steps=30)  # 25 denoising steps + 5 for init/finalization
        
        # Generate video with progress tracking
        asyncio.create_task(video_generator.generate_fast(
            job_id=job_id,
            input_path=str(input_path),
            output_path=str(output_path),
            progress_manager=progress_manager
        ))
        
        return JSONResponse({
            "job_id": job_id,
            "status": "processing"
        })
    
    except Exception as e:
        if input_path.exists():
            input_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_manager.add_websocket(job_id, websocket)
    
    try:
        # Send initial status if job exists
        if job_id in progress_manager.active_jobs:
            await websocket.send_json({
                "type": "progress",
                "data": progress_manager.active_jobs[job_id]
            })
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        progress_manager.remove_websocket(job_id, websocket)

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"generated_{job_id}.mp4"
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