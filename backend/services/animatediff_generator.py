import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimateDiffGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_to_video_pipe = None
        self.image_to_video_pipe = None
        
    def load_text_to_video_model(self):
        """Load AnimateDiff text-to-video model"""
        if self.text_to_video_pipe is None:
            logger.info("Loading AnimateDiff text-to-video model...")
            try:
                # Load motion adapter
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-v1-5-2",
                    torch_dtype=self.dtype
                )
                
                # Load pipeline with Realistic Vision model
                self.text_to_video_pipe = AnimateDiffPipeline.from_pretrained(
                    "SG161222/Realistic_Vision_V5.1_noVAE",
                    motion_adapter=adapter,
                    torch_dtype=self.dtype
                )
                self.text_to_video_pipe.scheduler = DDIMScheduler.from_config(
                    self.text_to_video_pipe.scheduler.config,
                    beta_schedule="linear",
                    steps_offset=1,
                    clip_sample=False
                )
                
                self.text_to_video_pipe = self.text_to_video_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.text_to_video_pipe.enable_vae_slicing()
                    self.text_to_video_pipe.enable_model_cpu_offload()
                        
                logger.info("AnimateDiff text-to-video model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading AnimateDiff: {str(e)}")
                raise
    
    def load_image_to_video_model(self):
        """Load AnimateDiff image-to-video model"""
        if self.image_to_video_pipe is None:
            logger.info("Loading AnimateDiff image-to-video model...")
            try:
                from diffusers import AnimateDiffI2VPipeline
                
                # Load motion adapter
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-v1-5-2",
                    torch_dtype=self.dtype
                )
                
                # Load I2V pipeline
                self.image_to_video_pipe = AnimateDiffI2VPipeline.from_pretrained(
                    "SG161222/Realistic_Vision_V5.1_noVAE",
                    motion_adapter=adapter,
                    torch_dtype=self.dtype
                )
                
                self.image_to_video_pipe = self.image_to_video_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.image_to_video_pipe.enable_vae_slicing()
                    self.image_to_video_pipe.enable_model_cpu_offload()
                        
                logger.info("AnimateDiff image-to-video model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading AnimateDiff I2V: {str(e)}")
                # Fallback to text-to-video if I2V not available
                logger.info("Falling back to text-to-video model for image animation")
                self.load_text_to_video_model()
                self.image_to_video_pipe = self.text_to_video_pipe
    
    def _generate_text_clip_sync(self, prompt, num_frames=16, guidance_scale=7.5, seed=None):
        """Generate a single video clip from text prompt"""
        self.load_text_to_video_model()
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # AnimateDiff generates 16 frames (2 seconds at 8 FPS)
        video_frames = self.text_to_video_pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=25,
            generator=generator,
            height=512,
            width=512
        ).frames[0]
        
        return video_frames
    
    def _generate_image_clip_sync(self, image, prompt, num_frames=16, guidance_scale=7.5, seed=None):
        """Generate a video clip from image and text prompt"""
        self.load_image_to_video_model()
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Resize image to 512x512
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        
        if hasattr(self.image_to_video_pipe, '__class__') and 'I2V' in self.image_to_video_pipe.__class__.__name__:
            # Use I2V pipeline
            video_frames = self.image_to_video_pipe(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=25,
                generator=generator
            ).frames[0]
        else:
            # Fallback: use img2img-like approach with text-to-video
            video_frames = self.text_to_video_pipe(
                prompt=f"{prompt}, high quality, detailed",
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=25,
                generator=generator,
                height=512,
                width=512
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
            fps=8
        )
        
        # Cleanup
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
    
    async def generate_video_from_text(self, prompt, output_path, duration_seconds=10, progress_callback=None):
        """Generate a video from text prompt"""
        loop = asyncio.get_event_loop()
        
        # AnimateDiff generates 2-second clips
        clip_duration = 2
        num_clips = max(1, (duration_seconds + clip_duration - 1) // clip_duration)
        
        logger.info(f"Generating {num_clips} clips for {duration_seconds} second video")
        
        temp_dir = tempfile.mkdtemp()
        clip_paths = []
        
        try:
            for i in range(num_clips):
                if progress_callback:
                    progress = (i / num_clips) * 80
                    await progress_callback(progress, f"Generating clip {i+1}/{num_clips}")
                
                # Add variation to prompt for continuity
                clip_prompt = prompt
                if i > 0:
                    clip_prompt = f"{prompt}, dynamic motion, continuation"
                
                logger.info(f"Generating clip {i+1} with prompt: {clip_prompt}")
                
                # Generate clip
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_text_clip_sync,
                    clip_prompt,
                    16,  # num_frames
                    7.5,  # guidance_scale
                    42 + i  # seed variation
                )
                
                # Save clip
                clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                export_to_video(frames, clip_path, fps=8)
                clip_paths.append(clip_path)
            
            if len(clip_paths) > 1:
                if progress_callback:
                    await progress_callback(85, "Stitching clips together...")
                
                # Stitch clips
                await loop.run_in_executor(
                    self.executor,
                    self._stitch_videos_sync,
                    clip_paths,
                    output_path
                )
            else:
                # Single clip, just copy it
                import shutil
                shutil.copy(clip_paths[0], output_path)
            
            if progress_callback:
                await progress_callback(100, "Video generation complete!")
            
            return output_path
            
        finally:
            # Cleanup
            for path in clip_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    async def generate_video_from_image(self, image_path, prompt, output_path, duration_seconds=10, progress_callback=None):
        """Generate a video from image and text prompt"""
        loop = asyncio.get_event_loop()
        
        # AnimateDiff generates 2-second clips
        clip_duration = 2
        num_clips = max(1, (duration_seconds + clip_duration - 1) // clip_duration)
        
        logger.info(f"Generating {num_clips} clips for {duration_seconds} second video from image")
        
        temp_dir = tempfile.mkdtemp()
        clip_paths = []
        
        try:
            # Load the image once
            image = Image.open(image_path)
            
            for i in range(num_clips):
                if progress_callback:
                    progress = (i / num_clips) * 80
                    await progress_callback(progress, f"Animating clip {i+1}/{num_clips}")
                
                # Add variation for different movements
                movement_variations = [
                    "subtle camera movement",
                    "gentle zoom in",
                    "slow pan right",
                    "slight rotation",
                    "zoom out slowly"
                ]
                variation = movement_variations[i % len(movement_variations)]
                clip_prompt = f"{prompt}, {variation}"
                
                logger.info(f"Generating clip {i+1} with prompt: {clip_prompt}")
                
                # Generate clip
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_image_clip_sync,
                    image,
                    clip_prompt,
                    16,  # num_frames
                    7.5,  # guidance_scale
                    42 + i  # seed variation
                )
                
                # Save clip
                clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                export_to_video(frames, clip_path, fps=8)
                clip_paths.append(clip_path)
            
            if len(clip_paths) > 1:
                if progress_callback:
                    await progress_callback(85, "Stitching clips together...")
                
                # Stitch clips
                await loop.run_in_executor(
                    self.executor,
                    self._stitch_videos_sync,
                    clip_paths,
                    output_path
                )
            else:
                # Single clip
                import shutil
                shutil.copy(clip_paths[0], output_path)
            
            if progress_callback:
                await progress_callback(100, "Video generation complete!")
            
            return output_path
            
        finally:
            # Cleanup
            for path in clip_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None):
        """Generate video with progress tracking"""
        try:
            await progress_manager.update_progress(job_id, 1, "Loading AnimateDiff model...")
            
            async def progress_callback(progress, message):
                await progress_manager.update_progress(job_id, int(progress), message)
            
            if image_path:
                await self.generate_video_from_image(
                    image_path=image_path,
                    prompt=prompt,
                    output_path=output_path,
                    duration_seconds=duration_seconds,
                    progress_callback=progress_callback
                )
            else:
                await self.generate_video_from_text(
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