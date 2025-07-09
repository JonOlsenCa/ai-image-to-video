import torch
from diffusers import StableVideoDiffusionPipeline, AnimateDiffPipeline, MotionAdapter
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

class GuidedVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.svd_pipe = None
        self.animatediff_pipe = None
        
    def load_svd_model(self):
        """Load Stable Video Diffusion model for image-to-video"""
        if self.svd_pipe is None:
            print("Loading SVD model...")
            self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None
            )
            self.svd_pipe = self.svd_pipe.to(self.device)
            
            if self.device == "cuda":
                self.svd_pipe.enable_model_cpu_offload()
                try:
                    self.svd_pipe.unet.enable_xformers_memory_efficient_attention()
                except ModuleNotFoundError:
                    print("xformers not available, using default attention")
    
    def _generate_with_prompt_sync(self, input_path, output_path, prompt, num_frames, motion_bucket_id, noise_aug_strength, fps=7):
        """Generate video with text prompt guidance"""
        print(f"Guided video generator called with prompt: '{prompt}', frames: {num_frames}, fps: {fps}")
        self.load_svd_model()
        
        # Load and prepare image
        image = Image.open(input_path)
        image = image.resize((1024, 576))
        
        generator = torch.manual_seed(42)
        
        # For now, use SVD with adjusted parameters based on prompt
        # In the future, this could use a more sophisticated text-guided approach
        
        # Adjust motion intensity based on prompt keywords
        adjusted_motion = motion_bucket_id
        adjusted_noise = noise_aug_strength
        
        if prompt:
            prompt_lower = prompt.lower()
            print(f"Processing prompt: '{prompt}' for motion guidance")
            
            # High motion keywords
            high_motion_words = ['fast', 'quick', 'racing', 'running', 'flying', 'spinning', 'dancing', 'energetic', 'dynamic', 'rapid', 'intense']
            # Low motion keywords  
            low_motion_words = ['slow', 'gentle', 'calm', 'peaceful', 'still', 'quiet', 'subtle', 'soft', 'minimal', 'slight']
            # Water/fluid motion
            fluid_words = ['water', 'waves', 'flowing', 'liquid', 'ocean', 'river', 'stream', 'fluid', 'ripples', 'splash']
            # Wind/air motion
            wind_words = ['wind', 'breeze', 'air', 'floating', 'drifting', 'swaying', 'rustling', 'blowing']
            # Fire/smoke motion
            fire_words = ['fire', 'flame', 'smoke', 'burning', 'flickering', 'ember', 'spark']
            
            if any(word in prompt_lower for word in high_motion_words):
                adjusted_motion = min(255, int(motion_bucket_id * 1.8))
                adjusted_noise = min(0.15, noise_aug_strength * 1.5)
                print(f"High motion detected: motion={adjusted_motion}, noise={adjusted_noise}")
            elif any(word in prompt_lower for word in low_motion_words):
                adjusted_motion = max(30, int(motion_bucket_id * 0.5))
                adjusted_noise = max(0.01, noise_aug_strength * 0.5)
                print(f"Low motion detected: motion={adjusted_motion}, noise={adjusted_noise}")
            elif any(word in prompt_lower for word in fluid_words):
                adjusted_motion = min(220, int(motion_bucket_id * 1.4))
                adjusted_noise = min(0.12, noise_aug_strength * 1.4)
                print(f"Fluid motion detected: motion={adjusted_motion}, noise={adjusted_noise}")
            elif any(word in prompt_lower for word in wind_words):
                adjusted_motion = min(180, int(motion_bucket_id * 1.2))
                adjusted_noise = min(0.08, noise_aug_strength * 1.2)
                print(f"Wind motion detected: motion={adjusted_motion}, noise={adjusted_noise}")
            elif any(word in prompt_lower for word in fire_words):
                adjusted_motion = min(200, int(motion_bucket_id * 1.3))
                adjusted_noise = min(0.1, noise_aug_strength * 1.3)
                print(f"Fire motion detected: motion={adjusted_motion}, noise={adjusted_noise}")
            else:
                # If prompt provided but no specific keywords, still increase motion slightly
                adjusted_motion = min(200, int(motion_bucket_id * 1.1))
                print(f"General prompt motion boost: motion={adjusted_motion}")
        
        # Update the values to use adjusted parameters
        motion_bucket_id = adjusted_motion
        noise_aug_strength = adjusted_noise
        
        frames = self.svd_pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=adjusted_motion,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=25,  # Higher quality for guided generation
        ).frames[0]
        
        export_to_video(frames, output_path, fps=fps)
        return output_path
    
    async def generate_with_prompt(self, input_path, output_path, prompt="", num_frames=25, motion_bucket_id=127, noise_aug_strength=0.02, fps=7):
        """Generate video with text prompt guidance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_with_prompt_sync,
            input_path,
            output_path,
            prompt,
            num_frames,
            motion_bucket_id,
            noise_aug_strength,
            fps
        )
    
    async def generate_with_prompt_and_progress(self, job_id, input_path, output_path, prompt, num_frames, motion_bucket_id, noise_aug_strength, fps, progress_manager):
        """Generate video with text prompt guidance and progress tracking"""
        try:
            # Update progress: Loading model
            await progress_manager.update_progress(job_id, 1, "Loading AI model...")
            
            # Load model
            self.load_svd_model()
            await progress_manager.update_progress(job_id, 5, "Model loaded, analyzing prompt...")
            
            # Analyze prompt and prepare image
            image = Image.open(input_path)
            image = image.resize((1024, 576))
            
            if prompt:
                await progress_manager.update_progress(job_id, 10, f"Generating video with guidance: '{prompt[:50]}...'")
            else:
                await progress_manager.update_progress(job_id, 10, "Generating video from image...")
            
            # Create progress update task
            async def update_generation_progress():
                for i in range(11, 90):
                    await asyncio.sleep(12)  # Update every 12 seconds (higher quality = slower)
                    if i < 90:
                        await progress_manager.update_progress(
                            job_id, i, 
                            f"Generating frames with text guidance... ({i-10}/80)"
                        )
            
            # Start progress updates
            progress_task = asyncio.create_task(update_generation_progress())
            
            # Generate video
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    self._generate_with_prompt_sync,
                    input_path,
                    output_path,
                    prompt,
                    num_frames,
                    motion_bucket_id,
                    noise_aug_strength,
                    fps
                )
            finally:
                # Cancel progress updates
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
            await progress_manager.update_progress(job_id, 90, "Finalizing video...")
            await progress_manager.update_progress(job_id, 100, "Video generation complete!")
            await progress_manager.complete_job(job_id, success=True)
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating guided video: {str(e)}")
            await progress_manager.complete_job(job_id, success=False)
            raise e