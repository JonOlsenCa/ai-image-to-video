import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

class SimpleVideoGenerator:
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
            
            # Use CPU offloading to save memory
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
            
            print("Model loaded successfully")
    
    async def generate_fast(self, job_id, input_path, output_path, progress_manager):
        try:
            # Update progress: Loading model
            await progress_manager.update_progress(job_id, 1, "Loading AI model...")
            
            # Load model first
            self.load_model()
            await progress_manager.update_progress(job_id, 10, "Model loaded, preparing image...")
            
            # Prepare image
            image = Image.open(input_path)
            # Smaller size for faster generation
            image = image.resize((512, 288))
            await progress_manager.update_progress(job_id, 15, "Starting video generation...")
            
            # Generate video with minimal settings for speed
            generator = torch.manual_seed(42)
            
            # Run generation in executor
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                self.executor,
                lambda: self.pipe(
                    image,
                    num_frames=7,  # Only 7 frames (1 second)
                    decode_chunk_size=4,
                    generator=generator,
                    num_inference_steps=10,  # Very few steps for speed
                    motion_bucket_id=64,  # Lower motion
                    noise_aug_strength=0.02,
                ).frames[0]
            )
            
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