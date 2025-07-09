import torch
from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile
from transformers import CLIPTextModel, CLIPTokenizer
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pipe = None
        
    def load_model(self):
        """Load Stable Video Diffusion model"""
        if self.pipe is None:
            logger.info("Loading Stable Video Diffusion model...")
            try:
                # Try I2VGen-XL first (better for image animation with prompts)
                try:
                    self.pipe = I2VGenXLPipeline.from_pretrained(
                        "ali-vilab/i2vgen-xl",
                        torch_dtype=self.dtype,
                        variant="fp16" if self.device == "cuda" else None
                    )
                    logger.info("Loaded I2VGen-XL model")
                except:
                    # Fallback to SVD
                    logger.info("I2VGen-XL not available, loading SVD...")
                    self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-video-diffusion-img2vid",
                        torch_dtype=self.dtype,
                        variant="fp16" if self.device == "cuda" else None
                    )
                    logger.info("Loaded Stable Video Diffusion model")
                
                self.pipe = self.pipe.to(self.device)
                
                if self.device == "cuda":
                    self.pipe.enable_model_cpu_offload()
                    self.pipe.enable_vae_slicing()
                        
                logger.info("Video model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading video model: {str(e)}")
                raise
    
    def analyze_prompt(self, prompt):
        """Analyze prompt to understand requested actions"""
        prompt_lower = prompt.lower()
        
        actions = {
            'zoom_in': any(word in prompt_lower for word in ['zoom in', 'close up', 'closer', 'zoom']),
            'zoom_out': any(word in prompt_lower for word in ['zoom out', 'wider', 'pull back', 'distant']),
            'pan_left': any(word in prompt_lower for word in ['pan left', 'move left', 'slide left']),
            'pan_right': any(word in prompt_lower for word in ['pan right', 'move right', 'slide right']),
            'rotate': any(word in prompt_lower for word in ['rotate', 'spin', 'turn']),
            'fade': any(word in prompt_lower for word in ['fade', 'dissolve']),
            'shake': any(word in prompt_lower for word in ['shake', 'vibrate', 'tremor']),
            'tilt': any(word in prompt_lower for word in ['tilt', 'angle']),
        }
        
        # Detect motion speed
        if any(word in prompt_lower for word in ['slow', 'slowly', 'gentle', 'smooth']):
            actions['speed'] = 'slow'
        elif any(word in prompt_lower for word in ['fast', 'quick', 'rapid']):
            actions['speed'] = 'fast'
        else:
            actions['speed'] = 'normal'
            
        # Detect style
        if any(word in prompt_lower for word in ['dramatic', 'cinematic', 'epic']):
            actions['style'] = 'cinematic'
        elif any(word in prompt_lower for word in ['dreamy', 'soft', 'ethereal']):
            actions['style'] = 'dreamy'
        else:
            actions['style'] = 'normal'
            
        return actions
    
    def apply_prompt_based_effects(self, image, prompt, frame_num, total_frames):
        """Apply effects to image based on prompt understanding"""
        actions = self.analyze_prompt(prompt)
        img = image.copy()
        width, height = img.size
        progress = frame_num / max(total_frames - 1, 1)
        
        # Speed modifier
        speed_mult = {'slow': 0.5, 'normal': 1.0, 'fast': 2.0}[actions.get('speed', 'normal')]
        
        # Apply zoom
        if actions.get('zoom_in'):
            zoom_factor = 1.0 + (0.5 * progress * speed_mult)
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            # Crop to center
            left = (img.width - width) // 2
            top = (img.height - height) // 2
            img = img.crop((left, top, left + width, top + height))
            
        elif actions.get('zoom_out'):
            zoom_factor = 1.0 - (0.3 * progress * speed_mult)
            zoom_factor = max(0.5, zoom_factor)  # Don't zoom out too much
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            # Place on black background
            img = Image.new('RGB', (width, height), 'black')
            paste_x = (width - img_resized.width) // 2
            paste_y = (height - img_resized.height) // 2
            img.paste(img_resized, (paste_x, paste_y))
        
        # Apply pan
        if actions.get('pan_left') or actions.get('pan_right'):
            pan_distance = int(width * 0.3 * progress * speed_mult)
            if actions.get('pan_left'):
                pan_distance = -pan_distance
            
            # Create larger canvas
            canvas = Image.new('RGB', (width + abs(pan_distance), height), 'black')
            if pan_distance > 0:
                canvas.paste(img, (0, 0))
                img = canvas.crop((pan_distance, 0, pan_distance + width, height))
            else:
                canvas.paste(img, (abs(pan_distance), 0))
                img = canvas.crop((0, 0, width, height))
        
        # Apply rotation
        if actions.get('rotate'):
            angle = 360 * progress * speed_mult
            img = img.rotate(angle, expand=False, fillcolor='black')
        
        # Apply tilt
        if actions.get('tilt'):
            angle = 15 * np.sin(progress * np.pi * 2) * speed_mult
            img = img.rotate(angle, expand=False, fillcolor='black')
        
        # Apply shake
        if actions.get('shake'):
            shake_x = int(10 * np.sin(frame_num * 0.5) * speed_mult)
            shake_y = int(10 * np.cos(frame_num * 0.5) * speed_mult)
            canvas = Image.new('RGB', (width + 20, height + 20), 'black')
            canvas.paste(img, (10 + shake_x, 10 + shake_y))
            img = canvas.crop((10, 10, 10 + width, 10 + height))
        
        # Apply style effects
        if actions.get('style') == 'cinematic':
            # Add letterbox bars
            bar_height = int(height * 0.1)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (width, bar_height)], fill='black')
            draw.rectangle([(0, height - bar_height), (width, height)], fill='black')
            # Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
        elif actions.get('style') == 'dreamy':
            # Apply soft focus
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            # Reduce saturation
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.8)
            # Add fade
            if actions.get('fade') or progress > 0.8:
                alpha = 1.0 - ((progress - 0.8) / 0.2) if progress > 0.8 else 1.0
                img = Image.blend(Image.new('RGB', img.size, 'white'), img, alpha)
        
        return img
    
    def _generate_video_frames_sync(self, image_path, prompt, num_frames=25):
        """Generate video frames with prompt-based animation"""
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        frames = []
        for i in range(num_frames):
            # Apply prompt-based effects to create animation
            frame = self.apply_prompt_based_effects(image, prompt, i, num_frames)
            frames.append(np.array(frame))
        
        return frames
    
    def _generate_ai_video_sync(self, image_path, prompt, num_frames=25):
        """Generate video using AI model if available"""
        self.load_model()
        
        try:
            image = load_image(image_path)
            image = image.resize((512, 512))
            
            if hasattr(self.pipe, 'generate'):
                # I2VGen-XL style
                frames = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_frames=num_frames,
                    guidance_scale=7.5,
                    num_inference_steps=25
                ).frames[0]
            else:
                # SVD style (no prompt support)
                frames = self.pipe(
                    image,
                    num_frames=num_frames,
                    decode_chunk_size=8,
                    generator=torch.manual_seed(42),
                    motion_bucket_id=127,
                    fps=8
                ).frames[0]
            
            return frames
            
        except Exception as e:
            logger.warning(f"AI generation failed, using prompt-based animation: {e}")
            # Fallback to prompt-based animation
            return self._generate_video_frames_sync(image_path, prompt, num_frames)
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None):
        """Generate video with AI or prompt-based animation"""
        loop = asyncio.get_event_loop()
        
        try:
            if not image_path:
                raise ValueError("Image is required for video generation")
            
            if progress_callback:
                await progress_callback(10, "Analyzing your prompt...")
            
            # Analyze what the user wants
            actions = self.analyze_prompt(prompt)
            logger.info(f"Detected actions from prompt: {actions}")
            
            if progress_callback:
                await progress_callback(20, "Preparing video generation...")
            
            # Calculate frames (8 fps)
            num_frames = min(duration_seconds * 8, 80)
            
            # Try AI generation first, fallback to prompt-based
            try:
                if progress_callback:
                    await progress_callback(30, "Generating AI video...")
                
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_ai_video_sync,
                    image_path,
                    prompt,
                    num_frames
                )
            except Exception as e:
                logger.info(f"Using prompt-based animation: {e}")
                if progress_callback:
                    await progress_callback(30, "Creating animation based on your prompt...")
                
                frames = await loop.run_in_executor(
                    self.executor,
                    self._generate_video_frames_sync,
                    image_path,
                    prompt,
                    num_frames
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
            await progress_manager.update_progress(job_id, 1, "Starting intelligent video generation...")
            
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