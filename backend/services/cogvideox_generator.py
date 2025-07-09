import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CogVideoXGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_to_video_pipe = None
        self.image_to_video_pipe = None
        
    def load_models(self):
        """Load CogVideoX models"""
        if self.text_to_video_pipe is None:
            logger.info("Loading CogVideoX text-to-video model...")
            try:
                # Use the 2B model for lower VRAM requirements
                self.text_to_video_pipe = CogVideoXPipeline.from_pretrained(
                    "THUDM/CogVideoX-2b",
                    torch_dtype=self.dtype,
                    revision="main"
                )
                self.text_to_video_pipe = self.text_to_video_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.text_to_video_pipe.enable_model_cpu_offload()
                    try:
                        self.text_to_video_pipe.enable_xformers_memory_efficient_attention()
                    except:
                        logger.info("xformers not available, using default attention")
                        
                logger.info("CogVideoX text-to-video model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading CogVideoX: {str(e)}")
                raise
    
    def _generate_single_clip_sync(self, prompt, num_frames=49, guidance_scale=7.5, seed=None):
        """Generate a single video clip from text prompt"""
        self.load_models()
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # CogVideoX generates 49 frames (6 seconds at 8 FPS)
        video_frames = self.text_to_video_pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=50,  # CogVideoX default
            generator=generator
        ).frames[0]
        
        return video_frames
    
    def _stitch_videos_sync(self, video_paths, output_path):
        """Stitch multiple video clips together"""
        clips = []
        for path in video_paths:
            clip = VideoFileClip(path)
            clips.append(clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Write the final video
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio=False,
            fps=8  # CogVideoX default FPS
        )
        
        # Cleanup
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
    
    async def generate_long_video(self, prompt, output_path, duration_seconds=10, progress_callback=None):
        """Generate a long video by creating multiple clips and stitching them"""
        loop = asyncio.get_event_loop()
        
        # CogVideoX generates 6-second clips
        clip_duration = 6
        num_clips = max(1, (duration_seconds + clip_duration - 1) // clip_duration)
        
        logger.info(f"Generating {num_clips} clips for {duration_seconds} second video")
        
        temp_dir = tempfile.mkdtemp()
        clip_paths = []
        
        try:
            # Generate multiple clips with slight prompt variations
            for i in range(num_clips):
                if progress_callback:
                    progress = (i / num_clips) * 80  # 80% for generation
                    await progress_callback(progress, f"Generating clip {i+1}/{num_clips}")
                
                # Add slight variation to prompt for continuity
                clip_prompt = prompt
                if i > 0:
                    clip_prompt = f"{prompt}, continuation"
                
                logger.info(f"Generating clip {i+1} with prompt: {clip_prompt}")
                
                # Generate clip
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_single_clip_sync,
                    clip_prompt,
                    49,  # num_frames
                    7.5,  # guidance_scale
                    42 + i  # seed variation
                )
                
                # Save clip
                clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                export_to_video(frames, clip_path, fps=8)
                clip_paths.append(clip_path)
            
            if progress_callback:
                await progress_callback(85, "Stitching clips together...")
            
            # Stitch clips together
            await loop.run_in_executor(
                self.executor,
                self._stitch_videos_sync,
                clip_paths,
                output_path
            )
            
            if progress_callback:
                await progress_callback(100, "Video generation complete!")
            
            return output_path
            
        finally:
            # Cleanup temp files
            for path in clip_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager):
        """Generate video with progress tracking"""
        try:
            await progress_manager.update_progress(job_id, 1, "Loading CogVideoX model...")
            
            async def progress_callback(progress, message):
                await progress_manager.update_progress(job_id, int(progress), message)
            
            await self.generate_long_video(
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                progress_callback=progress_callback
            )
            
            await progress_manager.complete_job(job_id, success=True)
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            await progress_manager.complete_job(job_id, success=False)
            raise e