import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class VideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            print(f"Loading model on {self.device}")
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None
            )
            self.pipe = self.pipe.to(self.device)
            
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                try:
                    self.pipe.unet.enable_xformers_memory_efficient_attention()
                except ModuleNotFoundError:
                    print("xformers not available, using default attention")
            
            print("Model loaded successfully")
    
    def _generate_sync(self, input_path, output_path, num_frames, motion_bucket_id, noise_aug_strength):
        self.load_model()
        
        image = Image.open(input_path)
        image = image.resize((1024, 576))
        
        generator = torch.manual_seed(42)
        
        frames = self.pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        ).frames[0]
        
        export_to_video(frames, output_path, fps=7)
        
        return output_path
    
    async def generate(self, input_path, output_path, num_frames=25, motion_bucket_id=127, noise_aug_strength=0.02):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_sync,
            input_path,
            output_path,
            num_frames,
            motion_bucket_id,
            noise_aug_strength
        )
    
    async def generate_with_progress(self, job_id, input_path, output_path, num_frames, motion_bucket_id, noise_aug_strength, progress_manager):
        try:
            # Update progress: Loading model
            await progress_manager.update_progress(job_id, 1, "Loading AI model...")
            
            # Load model first
            self.load_model()
            await progress_manager.update_progress(job_id, 5, "Model loaded, preparing image...")
            
            # Prepare image
            image = Image.open(input_path)
            image = image.resize((1024, 576))
            await progress_manager.update_progress(job_id, 7, "Starting video generation...")
            
            # Generate video (this is the slow part)
            generator = torch.manual_seed(42)
            
            # Create a task to update progress periodically
            async def update_generation_progress():
                for i in range(8, 28):  # Progress from 8 to 27
                    await asyncio.sleep(10)  # Update every 10 seconds (since each step takes ~20s)
                    if i < 28:
                        await progress_manager.update_progress(
                            job_id, i, 
                            f"Generating video... ({i-7}/20 estimated)"
                        )
            
            # Start progress updates
            progress_task = asyncio.create_task(update_generation_progress())
            
            # Run generation in executor
            loop = asyncio.get_event_loop()
            try:
                frames = await loop.run_in_executor(
                    self.executor,
                    lambda: self.pipe(
                        image,
                        num_frames=num_frames,
                        decode_chunk_size=8,
                        generator=generator,
                        motion_bucket_id=motion_bucket_id,
                        noise_aug_strength=noise_aug_strength,
                        num_inference_steps=15,  # Reduced for faster generation
                    ).frames[0]
                )
            finally:
                # Cancel progress updates
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
            await progress_manager.update_progress(job_id, 28, "Encoding video...")
            
            # Export video
            await loop.run_in_executor(
                self.executor,
                export_to_video,
                frames,
                output_path,
                7
            )
            
            await progress_manager.update_progress(job_id, 29, "Video generation complete!")
            await progress_manager.complete_job(job_id, success=True)
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating video: {str(e)}")
            await progress_manager.complete_job(job_id, success=False)
            raise e
    
