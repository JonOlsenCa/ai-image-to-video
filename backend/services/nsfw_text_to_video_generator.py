import torch
from diffusers import StableVideoDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSFWTextToVideoGenerator:
    """
    Text-to-Video generator using UnfilteredAI/NSFW-gen-v2 for text-to-image
    and Stable Video Diffusion for image-to-video conversion.
    
    This generator provides uncensored image generation capabilities.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_to_image_pipe = None
        self.image_to_video_pipe = None
        
    def load_models(self):
        """Load NSFW-gen-v2 text-to-image and SVD image-to-video models"""
        logger.info(f"Loading models on {self.device}")
        
        # Load NSFW-gen-v2 text-to-image model
        if self.text_to_image_pipe is None:
            logger.info("Loading NSFW-gen-v2 text-to-image model...")
            try:
                self.text_to_image_pipe = StableDiffusionXLPipeline.from_pretrained(
                    "UnfilteredAI/NSFW-gen-v2",
                    torch_dtype=self.dtype,
                    variant="fp16" if self.device == "cuda" else None,
                    use_safetensors=True
                )
                self.text_to_image_pipe = self.text_to_image_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.text_to_image_pipe.enable_model_cpu_offload()
                    try:
                        self.text_to_image_pipe.enable_xformers_memory_efficient_attention()
                    except ModuleNotFoundError:
                        logger.info("xformers not available for text-to-image, using default attention")
                
                logger.info("NSFW-gen-v2 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NSFW-gen-v2 model: {e}")
                raise
        
        # Load image-to-video model (SVD)
        if self.image_to_video_pipe is None:
            logger.info("Loading SVD image-to-video model...")
            try:
                self.image_to_video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=self.dtype,
                    variant="fp16" if self.device == "cuda" else None
                )
                self.image_to_video_pipe = self.image_to_video_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.image_to_video_pipe.enable_model_cpu_offload()
                    try:
                        self.image_to_video_pipe.enable_xformers_memory_efficient_attention()
                    except ModuleNotFoundError:
                        logger.info("xformers not available for image-to-video, using default attention")
                
                logger.info("SVD image-to-video model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SVD model: {e}")
                raise
    
    def _generate_image_from_text(self, prompt, negative_prompt="blurry, low quality, distorted, censored", width=1024, height=576):
        """Generate uncensored image from text prompt using NSFW-gen-v2"""
        self.load_models()
        
        generator = torch.manual_seed(42)
        
        # Enhanced prompt for better quality with NSFW-gen-v2
        enhanced_prompt = f"{prompt}, high quality, detailed, 3d style" if "3d" not in prompt.lower() else f"{prompt}, high quality, detailed"
        
        image = self.text_to_image_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=25,  # Slightly higher for better quality
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        
        return image
    
    def _generate_video_from_image(self, image, num_frames, motion_bucket_id, noise_aug_strength):
        """Generate video from image using SVD"""
        # Ensure image is the right size for SVD
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
            await progress_manager.update_progress(job_id, 1, "Loading NSFW-gen-v2 AI models...")
            
            # Load models
            self.load_models()
            await progress_manager.update_progress(job_id, 10, "Generating uncensored image from text...")
            
            # Generate image from text
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                self.executor,
                self._generate_image_from_text,
                prompt
            )
            
            await progress_manager.update_progress(job_id, 40, "Image generated, creating video...")
            
            # Generate video from image
            frames = await loop.run_in_executor(
                self.executor,
                self._generate_video_from_image,
                image,
                num_frames,
                motion_bucket_id,
                noise_aug_strength
            )
            
            await progress_manager.update_progress(job_id, 80, "Encoding final video...")
            
            # Export video
            await loop.run_in_executor(
                self.executor,
                export_to_video,
                frames,
                output_path,
                fps
            )
            
            await progress_manager.update_progress(job_id, 100, "Video generation complete!")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in NSFW text-to-video generation: {e}")
            await progress_manager.update_progress(job_id, -1, f"Error: {str(e)}")
            raise
    
    async def generate_image_only(self, prompt, output_path, width=1024, height=576):
        """Generate only an image from text prompt (no video)"""
        loop = asyncio.get_event_loop()
        
        image = await loop.run_in_executor(
            self.executor,
            self._generate_image_from_text,
            prompt,
            "blurry, low quality, distorted, censored",
            width,
            height
        )
        
        # Save image
        image.save(output_path)
        return output_path
    
    def cleanup(self):
        """Clean up resources"""
        if self.text_to_image_pipe is not None:
            del self.text_to_image_pipe
            self.text_to_image_pipe = None
        
        if self.image_to_video_pipe is not None:
            del self.image_to_video_pipe
            self.image_to_video_pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("NSFW generator resources cleaned up")
