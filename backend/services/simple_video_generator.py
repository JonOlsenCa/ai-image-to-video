import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from PIL import Image, ImageEnhance
import imageio
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoGenerator:
    """Simple, reliable video generator"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def analyze_prompt(self, prompt):
        """Simple but reliable prompt analysis"""
        if not prompt:
            return 'zoom_in'
        
        prompt = prompt.lower().strip()
        print(f"🎬 Analyzing: '{prompt}'")
        
        if 'zoom out' in prompt:
            print("✅ Detected: zoom_out")
            return 'zoom_out'
        elif 'pan right' in prompt:
            print("✅ Detected: pan_right")
            return 'pan_right'
        elif 'pan left' in prompt:
            print("✅ Detected: pan_left")
            return 'pan_left'
        elif 'rotate clockwise' in prompt:
            print("✅ Detected: rotate_cw")
            return 'rotate_cw'
        elif 'rotate counter' in prompt:
            print("✅ Detected: rotate_ccw")
            return 'rotate_ccw'
        elif 'shake' in prompt:
            print("✅ Detected: shake")
            return 'shake'
        elif 'bounce' in prompt:
            print("✅ Detected: bounce")
            return 'bounce'
        elif 'fade out' in prompt:
            print("✅ Detected: fade_out")
            return 'fade_out'
        elif 'fade in' in prompt:
            print("✅ Detected: fade_in")
            return 'fade_in'
        elif 'pulse' in prompt:
            print("✅ Detected: pulse")
            return 'pulse'
        elif 'glow' in prompt:
            print("✅ Detected: glow")
            return 'glow'
        else:
            print("✅ Detected: zoom_in (default)")
            return 'zoom_in'
    
    def create_frame(self, image, frame_num, total_frames, animation_type):
        """Create a single animated frame"""
        progress = frame_num / max(total_frames - 1, 1)
        width, height = image.size
        
        try:
            if animation_type == 'zoom_out':
                factor = 1.3 - (0.4 * progress)
                new_size = (int(width * factor), int(height * factor))
                img = image.resize(new_size, Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (width, height), 'black')
                paste_x = (width - img.width) // 2
                paste_y = (height - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
                return canvas
                
            elif animation_type == 'pan_right':
                shift = int(200 * progress)
                canvas = Image.new('RGB', (width + 200, height), 'black')
                canvas.paste(image, (shift, 0))
                return canvas.crop((shift, 0, shift + width, height))
                
            elif animation_type == 'pan_left':
                shift = 200 - int(200 * progress)
                canvas = Image.new('RGB', (width + 200, height), 'black')
                canvas.paste(image, (shift, 0))
                return canvas.crop((0, 0, width, height))
                
            elif animation_type == 'rotate_cw':
                angle = 360 * progress
                return image.rotate(-angle, expand=False, fillcolor='black')
                
            elif animation_type == 'rotate_ccw':
                angle = 360 * progress
                return image.rotate(angle, expand=False, fillcolor='black')
                
            elif animation_type == 'shake':
                shake_x = int(8 * math.sin(frame_num * 0.8))
                shake_y = int(6 * math.cos(frame_num * 0.6))
                canvas = Image.new('RGB', (width + 16, height + 12), 'black')
                canvas.paste(image, (8 + shake_x, 6 + shake_y))
                return canvas.crop((8, 6, 8 + width, 6 + height))
                
            elif animation_type == 'bounce':
                bounce_y = int(20 * abs(math.sin(progress * math.pi * 3)))
                canvas = Image.new('RGB', (width, height + 20), 'black')
                canvas.paste(image, (0, bounce_y))
                return canvas.crop((0, 0, width, height))
                
            elif animation_type == 'fade_in':
                black = Image.new('RGB', image.size, 'black')
                return Image.blend(black, image, progress)
                
            elif animation_type == 'fade_out':
                white = Image.new('RGB', image.size, 'white')
                return Image.blend(image, white, progress)
                
            elif animation_type == 'pulse':
                factor = 1.0 + 0.2 * math.sin(progress * math.pi * 4)
                new_size = (int(width * factor), int(height * factor))
                img = image.resize(new_size, Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (width, height), 'black')
                paste_x = (width - img.width) // 2
                paste_y = (height - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
                return canvas
                
            elif animation_type == 'glow':
                enhancer = ImageEnhance.Brightness(image)
                brightness = 1.0 + 0.3 * math.sin(progress * math.pi * 2)
                return enhancer.enhance(brightness)
                
            else:  # zoom_in
                factor = 1.0 + (0.5 * progress)
                new_size = (int(width * factor), int(height * factor))
                img = image.resize(new_size, Image.Resampling.LANCZOS)
                left = (img.width - width) // 2
                top = (img.height - height) // 2
                return img.crop((left, top, left + width, top + height))
                
        except Exception as e:
            print(f"Error in animation {animation_type}: {e}")
            return image
    
    def _generate_video_sync(self, prompt, output_path, image_path, num_frames=80):
        """Generate video synchronously"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            animation_type = self.analyze_prompt(prompt)
            
            frames = []
            for i in range(num_frames):
                frame = self.create_frame(image, i, num_frames, animation_type)
                frames.append(np.array(frame))
            
            imageio.mimwrite(output_path, frames, fps=8, codec='libx264')
            return output_path
            
        except Exception as e:
            print(f"Error generating video: {e}")
            raise
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None):
        """Generate video with progress tracking"""
        if not image_path:
            raise ValueError("Image required")
        
        loop = asyncio.get_event_loop()
        num_frames = min(duration_seconds * 8, 160)
        
        try:
            if progress_callback:
                await progress_callback(10, "Starting animation...")
            
            result = await loop.run_in_executor(
                self.executor,
                self._generate_video_sync,
                prompt,
                output_path,
                image_path,
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
            await progress_manager.update_progress(job_id, 1, "Initializing...")
            
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