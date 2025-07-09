import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from PIL import Image
import imageio
import subprocess
import tempfile
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamiCrafterGenerator:
    """DynamiCrafter-based video generator for realistic content animation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model = None
        self.setup_complete = False
        
    def check_dependencies(self):
        """Check if DynamiCrafter dependencies are available"""
        try:
            import diffusers
            from diffusers import DiffusionPipeline
            return True
        except ImportError:
            return False
    
    def setup_dynamicrafter(self):
        """Setup DynamiCrafter model"""
        if self.setup_complete:
            return True
            
        try:
            logger.info("Setting up DynamiCrafter...")
            
            # Check if we can use Hugging Face pipeline
            from diffusers import DiffusionPipeline
            
            # Try to load DynamiCrafter model from Hugging Face
            try:
                self.model = DiffusionPipeline.from_pretrained(
                    "Doubiiu/DynamiCrafter",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.model = self.model.to(self.device)
                logger.info("✅ DynamiCrafter model loaded successfully")
                self.setup_complete = True
                return True
            except Exception as e:
                logger.warning(f"Could not load DynamiCrafter from HF: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup DynamiCrafter: {e}")
            return False
    
    def get_available_animations(self):
        """Return animations available with DynamiCrafter"""
        return {
            "Character Movement": [
                "person walking", "person dancing", "person waving", "person jumping",
                "person running", "person sitting down", "person standing up", "person nodding"
            ],
            "Facial Animation": [
                "person smiling", "person talking", "person blinking", "person looking around",
                "person laughing", "person surprised expression", "person thinking", "person winking"
            ],
            "Object Animation": [
                "leaves rustling", "water flowing", "fire flickering", "clouds moving",
                "flag waving", "curtains blowing", "smoke rising", "rain falling"
            ],
            "Camera Movement": [
                "camera zoom in", "camera zoom out", "camera pan left", "camera pan right",
                "camera tilt up", "camera tilt down", "camera orbit", "camera dolly"
            ]
        }
    
    def analyze_prompt(self, prompt):
        """Analyze prompt for DynamiCrafter-style animation"""
        if not prompt:
            return "person moving naturally"
        
        prompt_lower = prompt.lower().strip()
        logger.info(f"🎬 DynamiCrafter analyzing: '{prompt_lower}'")
        
        # Map simple commands to DynamiCrafter-compatible prompts
        animation_mapping = {
            'bounce': 'person jumping up and down',
            'shake': 'person shaking head',
            'wave': 'person waving hand',
            'dance': 'person dancing',
            'walk': 'person walking in place',
            'nod': 'person nodding head',
            'smile': 'person smiling',
            'talk': 'person talking',
            'zoom in': 'camera slowly zooming in',
            'zoom out': 'camera slowly zooming out',
            'pan left': 'camera panning left',
            'pan right': 'camera panning right',
            'rotate': 'camera rotating around subject'
        }
        
        # Find the best match
        for key, description in animation_mapping.items():
            if key in prompt_lower:
                logger.info(f"✅ Mapped '{prompt_lower}' to '{description}'")
                return description
        
        # If no mapping found, use the prompt as-is but make it more descriptive
        enhanced_prompt = f"realistic {prompt_lower} with natural movement"
        logger.info(f"✅ Enhanced prompt: '{enhanced_prompt}'")
        return enhanced_prompt
    
    def _generate_with_dynamicrafter_sync(self, image_path, prompt, output_path, num_frames=16):
        """Generate video using DynamiCrafter"""
        try:
            if not self.setup_dynamicrafter():
                raise Exception("DynamiCrafter not available")
            
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Enhanced prompt for better results
            enhanced_prompt = self.analyze_prompt(prompt)
            
            logger.info(f"Generating video with DynamiCrafter: '{enhanced_prompt}'")
            
            # Generate video frames
            with torch.no_grad():
                output = self.model(
                    prompt=enhanced_prompt,
                    image=image,
                    num_frames=num_frames,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    height=512,
                    width=512
                )
            
            # Extract frames
            if hasattr(output, 'frames'):
                frames = output.frames[0]
            else:
                frames = output
            
            # Convert to numpy arrays
            if not isinstance(frames[0], np.ndarray):
                frames = [np.array(frame) for frame in frames]
            
            # Save video
            imageio.mimwrite(output_path, frames, fps=8, codec='libx264')
            
            logger.info(f"✅ DynamiCrafter video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"DynamiCrafter generation failed: {e}")
            raise
    
    def _generate_fallback_sync(self, image_path, prompt, output_path, num_frames=80):
        """Fallback to simple animation if DynamiCrafter fails"""
        logger.info("Using fallback animation...")
        
        # Import our simple generator as fallback
        from .simple_video_generator import SimpleVideoGenerator
        fallback = SimpleVideoGenerator()
        
        # Use the simple generator
        return fallback._generate_video_sync(prompt, output_path, image_path, num_frames)
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None):
        """Generate video with DynamiCrafter or fallback"""
        if not image_path:
            raise ValueError("Image required")
        
        loop = asyncio.get_event_loop()
        num_frames = min(duration_seconds * 8, 80)  # DynamiCrafter works best with shorter clips
        
        try:
            if progress_callback:
                await progress_callback(10, "Loading DynamiCrafter model...")
            
            # Try DynamiCrafter first
            if self.check_dependencies():
                try:
                    result = await loop.run_in_executor(
                        self.executor,
                        self._generate_with_dynamicrafter_sync,
                        image_path,
                        prompt,
                        output_path,
                        min(num_frames, 16)  # DynamiCrafter optimal frame count
                    )
                    
                    if progress_callback:
                        await progress_callback(100, "AI animation complete!")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"DynamiCrafter failed: {e}")
                    if progress_callback:
                        await progress_callback(50, "Falling back to simple animation...")
            
            # Fallback to simple animation
            result = await loop.run_in_executor(
                self.executor,
                self._generate_fallback_sync,
                image_path,
                prompt,
                output_path,
                num_frames
            )
            
            if progress_callback:
                await progress_callback(100, "Animation complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None):
        """Generate video with progress manager"""
        try:
            await progress_manager.update_progress(job_id, 1, "Initializing AI animation...")
            
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
            logger.error(f"Error: {e}")
            await progress_manager.complete_job(job_id, success=False, error=str(e))
            raise