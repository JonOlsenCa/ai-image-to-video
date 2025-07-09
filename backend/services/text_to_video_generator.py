import torch
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import io

logging.basicConfig(level=logging.INFO)

class TextToVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_to_image_pipe = None
        self.image_to_video_pipe = None
        
    def load_models(self):
        print(f"Loading models on {self.device}")
        
        # Load text-to-image model (SDXL)
        if self.text_to_image_pipe is None:
            print("Loading SDXL text-to-image model...")
            self.text_to_image_pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None
            )
            self.text_to_image_pipe = self.text_to_image_pipe.to(self.device)
            
            if self.device == "cuda":
                self.text_to_image_pipe.enable_model_cpu_offload()
                try:
                    self.text_to_image_pipe.enable_xformers_memory_efficient_attention()
                except ModuleNotFoundError:
                    print("xformers not available for text-to-image, using default attention")
        
        # Load image-to-video model (SVD)
        if self.image_to_video_pipe is None:
            print("Loading SVD image-to-video model...")
            self.image_to_video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None
            )
            self.image_to_video_pipe = self.image_to_video_pipe.to(self.device)
            
            if self.device == "cuda":
                self.image_to_video_pipe.enable_model_cpu_offload()
                try:
                    self.image_to_video_pipe.unet.enable_xformers_memory_efficient_attention()
                except ModuleNotFoundError:
                    print("xformers not available for image-to-video, using default attention")
        
        print("All models loaded successfully")
    
    def _generate_image_from_text(self, prompt, negative_prompt="blurry, low quality, distorted", width=1024, height=576):
        """Generate image from text prompt using SDXL"""
        self.load_models()
        
        generator = torch.manual_seed(42)
        
        image = self.text_to_image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        
        return image
    
    def _generate_video_from_image(self, image, num_frames, motion_bucket_id, noise_aug_strength):
        """Generate video from image using SVD"""
        # Ensure image is the right size
        image = image.resize((1024, 576))
        
        generator = torch.manual_seed(42)
        
        frames = self.image_to_video_pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=15,
        ).frames[0]
        
        return frames
    
    async def generate_from_text(self, prompt, output_path, num_frames=25, motion_bucket_id=127, noise_aug_strength=0.02):
        """Generate video from text prompt (text -> image -> video)"""
        loop = asyncio.get_event_loop()
        
        # Generate image from text
        image = await loop.run_in_executor(
            self.executor,
            self._generate_image_from_text,
            prompt
        )
        
        # Generate video from image
        frames = await loop.run_in_executor(
            self.executor,
            self._generate_video_from_image,
            image,
            num_frames,
            motion_bucket_id,
            noise_aug_strength
        )
        
        # Export video
        await loop.run_in_executor(
            self.executor,
            export_to_video,
            frames,
            output_path,
            7
        )
        
        return output_path
    
    async def generate_from_text_with_progress(self, job_id, prompt, output_path, num_frames, motion_bucket_id, noise_aug_strength, fps, progress_manager):
        """Generate video from text with progress tracking"""
        try:
            # Update progress: Loading models
            await progress_manager.update_progress(job_id, 1, "Loading AI models...")
            
            # Load models
            self.load_models()
            await progress_manager.update_progress(job_id, 10, "Generating image from text...")
            
            # Generate image from text
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                self.executor,
                self._generate_image_from_text,
                prompt
            )
            
            await progress_manager.update_progress(job_id, 40, "Image generated! Creating video...")
            
            # Create progress update task for video generation
            async def update_video_progress():
                for i in range(41, 90):
                    await asyncio.sleep(8)  # Update every 8 seconds
                    if i < 90:
                        await progress_manager.update_progress(
                            job_id, i, 
                            f"Generating video frames... ({i-40}/50 estimated)"
                        )
            
            # Start progress updates
            progress_task = asyncio.create_task(update_video_progress())
            
            # Generate video from image
            try:
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_video_from_image,
                    image,
                    num_frames,
                    motion_bucket_id,
                    noise_aug_strength
                )
            finally:
                # Cancel progress updates
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
            await progress_manager.update_progress(job_id, 90, "Encoding video...")
            
            # Export video
            await loop.run_in_executor(
                self.executor,
                export_to_video,
                frames,
                output_path,
                fps
            )
            
            await progress_manager.update_progress(job_id, 100, "Video generation complete!")
            await progress_manager.complete_job(job_id, success=True)
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating video from text: {str(e)}")
            await progress_manager.complete_job(job_id, success=False)
            raise e