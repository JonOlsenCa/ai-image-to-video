import torch
from diffusers import DiffusionPipeline, MotionAdapter
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAnimateDiffGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pipe = None
        
    def load_model(self):
        """Load AnimateDiff model"""
        if self.pipe is None:
            logger.info("Loading AnimateDiff model...")
            try:
                # Load motion adapter
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-v1-5-2",
                    torch_dtype=self.dtype,
                    cache_dir="./backend/models"
                )
                
                # Load pipeline - using the standard AnimateDiff pipeline
                model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    motion_adapter=adapter,
                    torch_dtype=self.dtype,
                    cache_dir="./backend/models",
                    custom_pipeline="animatediff"
                )
                
                self.pipe = self.pipe.to(self.device)
                
                if self.device == "cuda":
                    self.pipe.enable_vae_slicing()
                    self.pipe.enable_model_cpu_offload()
                        
                logger.info("AnimateDiff model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading AnimateDiff: {str(e)}")
                raise
    
    def _generate_clip_sync(self, prompt, num_frames=16, guidance_scale=7.5, seed=None):
        """Generate a single video clip"""
        self.load_model()
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate video frames
        output = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=25,
            generator=generator,
            height=512,
            width=512
        )
        
        # Extract frames
        if hasattr(output, 'frames'):
            frames = output.frames[0] if isinstance(output.frames, list) else output.frames
        else:
            frames = output
            
        return frames
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None):
        """Generate a video from text prompt (image support can be added later)"""
        loop = asyncio.get_event_loop()
        
        try:
            if progress_callback:
                await progress_callback(10, "Loading model...")
            
            # For now, just generate a single clip
            # We can add stitching later if needed
            logger.info(f"Generating video with prompt: {prompt}")
            
            if progress_callback:
                await progress_callback(30, "Generating video frames...")
            
            # Generate frames
            frames = await loop.run_in_executor(
                self.executor,
                self._generate_clip_sync,
                prompt,
                16,  # num_frames
                7.5,  # guidance_scale
                42   # seed
            )
            
            if progress_callback:
                await progress_callback(80, "Encoding video...")
            
            # Save video
            export_to_video(frames, output_path, fps=8)
            
            if progress_callback:
                await progress_callback(100, "Video generation complete!")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None):
        """Generate video with progress tracking"""
        try:
            await progress_manager.update_progress(job_id, 1, "Starting video generation...")
            
            async def progress_callback(progress, message):
                await progress_manager.update_progress(job_id, int(progress), message)
            
            await self.generate_video(
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                progress_callback=progress_callback,
                image_path=image_path
            )
            
            await progress_manager.complete_job(job_id, success=True)
            return output_path
            
        except Exception as e:
            logger.error(f"Error in generate_with_progress: {str(e)}")
            await progress_manager.complete_job(job_id, success=False, error=str(e))
            raise