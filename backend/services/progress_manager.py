from typing import Dict, Optional
import asyncio
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

class ProgressManager:
    def __init__(self):
        self.active_jobs: Dict[str, dict] = {}
        self.websocket_connections: Dict[str, list] = {}
    
    def start_job(self, job_id: str, total_steps: int):
        self.active_jobs[job_id] = {
            "total_steps": total_steps,
            "current_step": 0,
            "start_time": time.time(),
            "status": "starting",
            "message": "Initializing...",
            "eta": None,
            "progress": 0
        }
    
    async def update_progress(self, job_id: str, current_step: int, message: str = ""):
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return
        
        job = self.active_jobs[job_id]
        job["current_step"] = current_step
        job["message"] = message
        job["progress"] = (current_step / job["total_steps"]) * 100
        job["status"] = "processing"
        
        # Calculate ETA
        elapsed_time = time.time() - job["start_time"]
        if current_step > 0:
            time_per_step = elapsed_time / current_step
            remaining_steps = job["total_steps"] - current_step
            eta_seconds = remaining_steps * time_per_step
            job["eta"] = int(eta_seconds)
        
        logger.info(f"Progress update: job={job_id}, step={current_step}/{job['total_steps']}, progress={job['progress']:.1f}%")
        
        # Send update to all connected websockets for this job
        await self._broadcast_update(job_id)
    
    async def complete_job(self, job_id: str, success: bool = True, error: str = None):
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job["status"] = "completed" if success else "failed"
            job["progress"] = 100 if success else job["progress"]
            job["eta"] = 0
            if success:
                job["message"] = "Generation completed successfully!"
                job["download_url"] = f"/download-video/{job_id}"
            elif error:
                job["message"] = f"Error: {error}"
            await self._broadcast_update(job_id)
            # Keep job info for 60 seconds after completion
            await asyncio.sleep(60)
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _broadcast_update(self, job_id: str):
        if job_id not in self.websocket_connections:
            logger.warning(f"No websocket connections for job {job_id}")
            return
        
        job_data = self.active_jobs.get(job_id)
        if not job_data:
            return
        
        # Send to all connected clients
        websockets = self.websocket_connections[job_id].copy()
        logger.info(f"Broadcasting to {len(websockets)} websocket(s) for job {job_id}")
        
        for ws in websockets:
            try:
                await ws.send_json({
                    "type": "progress",
                    "data": job_data
                })
                logger.debug(f"Sent progress update to websocket for job {job_id}")
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                # Remove disconnected websockets
                self.websocket_connections[job_id].remove(ws)
    
    def add_websocket(self, job_id: str, websocket):
        if job_id not in self.websocket_connections:
            self.websocket_connections[job_id] = []
        self.websocket_connections[job_id].append(websocket)
    
    def remove_websocket(self, job_id: str, websocket):
        if job_id in self.websocket_connections:
            if websocket in self.websocket_connections[job_id]:
                self.websocket_connections[job_id].remove(websocket)
            if not self.websocket_connections[job_id]:
                del self.websocket_connections[job_id]

progress_manager = ProgressManager()