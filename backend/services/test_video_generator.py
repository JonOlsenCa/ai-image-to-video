import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVideoGenerator:
    """Simple test video generator that creates animated text videos"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def analyze_prompt(self, prompt):
        """Analyze the prompt to determine what animation to apply"""
        if not prompt:
            print("No prompt provided, defaulting to zoom_in")
            return 'zoom_in'
        
        prompt_lower = prompt.lower().strip()
        print(f"Analyzing prompt: '{prompt_lower}'")
        
        # Check for specific animations with better pattern matching
        if 'pan left to right' in prompt_lower or 'left to right' in prompt_lower:
            print("Detected: pan_right")
            return 'pan_right'
        elif 'pan right to left' in prompt_lower or 'right to left' in prompt_lower:
            print("Detected: pan_left")
            return 'pan_left'
        elif 'zoom out' in prompt_lower:
            print("Detected: zoom_out")
            return 'zoom_out'
        elif 'zoom in' in prompt_lower:
            print("Detected: zoom_in")
            return 'zoom_in'
        elif 'rotate clockwise' in prompt_lower:
            print("Detected: rotate_cw")
            return 'rotate_cw'
        elif 'rotate counterclockwise' in prompt_lower:
            print("Detected: rotate_ccw")
            return 'rotate_ccw'
        elif 'rotate' in prompt_lower or 'spin' in prompt_lower:
            print("Detected: rotate")
            return 'rotate'
        elif 'shake' in prompt_lower:
            print("Detected: shake")
            return 'shake'
        elif 'tilt' in prompt_lower:
            print("Detected: tilt")
            return 'tilt'
        elif 'fade in' in prompt_lower:
            print("Detected: fade_in")
            return 'fade_in'
        elif 'fade out' in prompt_lower:
            print("Detected: fade_out")
            return 'fade_out'
        elif 'pulse' in prompt_lower:
            print("Detected: pulse")
            return 'pulse'
        elif 'blur' in prompt_lower:
            print("Detected: blur")
            return 'blur'
        elif 'orbit' in prompt_lower:
            print("Detected: orbit")
            return 'orbit'
        elif 'spiral' in prompt_lower:
            print("Detected: spiral")
            return 'spiral'
        elif 'wave' in prompt_lower:
            print("Detected: wave")
            return 'wave'
        elif 'bounce' in prompt_lower:
            print("Detected: bounce")
            return 'bounce'
        else:
            print(f"No specific animation detected in '{prompt_lower}', defaulting to zoom_in")
            return 'zoom_in'

    def _create_frame(self, text, frame_num, total_frames, base_image=None, size=(512, 512)):
        """Create a single frame with proper animation based on prompt"""
        if base_image is None:
            img = Image.new('RGB', size, color='black')
        else:
            # Determine animation type from prompt
            animation_type = self.analyze_prompt(text)
            progress = frame_num / max(total_frames - 1, 1)
            
            img = base_image.copy()
            width, height = img.size
            
            if animation_type == 'pan_left':
                # Pan from right to left
                canvas = Image.new('RGB', (width + 200, height), 'black')
                start_pos = 200 - int(200 * progress)
                canvas.paste(img, (start_pos, 0))
                img = canvas.crop((0, 0, width, height))
                
            elif animation_type == 'pan_right':
                # Pan from left to right
                canvas = Image.new('RGB', (width + 200, height), 'black')
                start_pos = int(200 * progress)
                canvas.paste(img, (start_pos, 0))
                img = canvas.crop((start_pos, 0, start_pos + width, height))
                
            elif animation_type == 'zoom_out':
                # Zoom out effect
                zoom_factor = 1.3 - (0.4 * progress)
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center on black background
                canvas = Image.new('RGB', size, 'black')
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                canvas.paste(img, (paste_x, paste_y))
                img = canvas
                
            elif animation_type == 'rotate':
                # Rotation effect
                angle = 360 * progress
                img = img.rotate(angle, expand=False, fillcolor='black')
                
            elif animation_type == 'shake':
                # Shake effect
                shake_x = int(10 * np.sin(frame_num * 0.8))
                shake_y = int(8 * np.cos(frame_num * 0.6))
                canvas = Image.new('RGB', (width + 20, height + 20), 'black')
                canvas.paste(img, (10 + shake_x, 10 + shake_y))
                img = canvas.crop((10, 10, 10 + width, 10 + height))
                
            elif animation_type == 'tilt':
                # Tilting effect
                angle = 15 * np.sin(progress * np.pi * 2)
                img = img.rotate(angle, expand=False, fillcolor='black')
                
            elif animation_type == 'fade':
                # Fade effect
                if progress < 0.5:
                    # Fade in
                    alpha = progress * 2
                    black = Image.new('RGB', img.size, 'black')
                    img = Image.blend(black, img, alpha)
                else:
                    # Fade out
                    alpha = (1.0 - progress) * 2
                    white = Image.new('RGB', img.size, 'white')
                    img = Image.blend(white, img, alpha)
                    
            elif animation_type == 'zoom_in':
                # Zoom in effect
                zoom_factor = 1.0 + (0.5 * progress)
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Crop from center
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                img = img.crop((left, top, left + width, top + height))
                
            elif animation_type == 'rotate_cw':
                # Clockwise rotation
                angle = 360 * progress
                img = img.rotate(-angle, expand=False, fillcolor='black')
                
            elif animation_type == 'rotate_ccw':
                # Counter-clockwise rotation
                angle = 360 * progress
                img = img.rotate(angle, expand=False, fillcolor='black')
                
            elif animation_type == 'fade_in':
                # Fade in from black
                black = Image.new('RGB', img.size, 'black')
                img = Image.blend(black, img, progress)
                
            elif animation_type == 'fade_out':
                # Fade out to white
                white = Image.new('RGB', img.size, 'white')
                img = Image.blend(img, white, progress)
                
            elif animation_type == 'pulse':
                # Pulsing zoom effect
                pulse_factor = 1.0 + 0.2 * np.sin(progress * np.pi * 4)
                new_width = int(width * pulse_factor)
                new_height = int(height * pulse_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center the pulsed image
                canvas = Image.new('RGB', (width, height), 'black')
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                canvas.paste(img, (paste_x, paste_y))
                img = canvas
                
            elif animation_type == 'blur':
                # Blur effect that goes in and out
                from PIL import ImageFilter
                blur_amount = 5 * np.sin(progress * np.pi)
                if blur_amount > 0.1:
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))
                    
            elif animation_type == 'orbit':
                # Orbital camera movement
                radius = 50
                angle = progress * np.pi * 2
                offset_x = int(radius * np.cos(angle))
                offset_y = int(radius * np.sin(angle))
                
                canvas = Image.new('RGB', (width + 100, height + 100), 'black')
                canvas.paste(img, (50 + offset_x, 50 + offset_y))
                img = canvas.crop((50, 50, 50 + width, 50 + height))
                
            elif animation_type == 'spiral':
                # Spiral zoom with rotation
                zoom_factor = 1.0 + (0.3 * progress)
                angle = progress * 720  # Two full rotations
                
                # Apply zoom
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Apply rotation
                img = img.rotate(angle, expand=False, fillcolor='black')
                
                # Crop from center
                if new_width > width:
                    left = (new_width - width) // 2
                    top = (new_height - height) // 2
                    img = img.crop((left, top, left + width, top + height))
                else:
                    canvas = Image.new('RGB', (width, height), 'black')
                    paste_x = (width - new_width) // 2
                    paste_y = (height - new_height) // 2
                    canvas.paste(img, (paste_x, paste_y))
                    img = canvas
                    
            elif animation_type == 'wave':
                # Wave distortion effect
                wave_pixels = np.array(img)
                wave_amplitude = 20
                wave_frequency = 0.1
                
                for y in range(height):
                    shift = int(wave_amplitude * np.sin(y * wave_frequency + progress * np.pi * 2))
                    if shift != 0:
                        wave_pixels[y] = np.roll(wave_pixels[y], shift, axis=0)
                
                img = Image.fromarray(wave_pixels)
                
            elif animation_type == 'bounce':
                # Bouncing effect
                bounce_height = 30
                bounce_y = int(bounce_height * abs(np.sin(progress * np.pi * 3)))
                
                canvas = Image.new('RGB', (width, height + bounce_height), 'black')
                canvas.paste(img, (0, bounce_y))
                img = canvas.crop((0, 0, width, height))
        
        # Don't add text overlay - let the animation speak for itself
        return np.array(img)
    
    def _generate_video_sync(self, prompt, output_path, num_frames=16, base_image=None):
        """Generate a test video with animated text and optional base image"""
        frames = []
        
        # Create frames
        for i in range(num_frames):
            frame = self._create_frame(prompt[:30], i, num_frames, base_image)
            frames.append(frame)
        
        # Save as video
        imageio.mimwrite(output_path, frames, fps=8, codec='libx264')
        
        return output_path
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None):
        """Generate a test video"""
        loop = asyncio.get_event_loop()
        
        try:
            if progress_callback:
                await progress_callback(10, "Starting test video generation...")
            
            # Load base image if provided
            base_image = None
            if image_path and os.path.exists(image_path):
                logger.info(f"Loading image: {image_path}")
                base_image = Image.open(image_path).convert('RGB')
                # Resize to standard size
                base_image = base_image.resize((512, 512), Image.Resampling.LANCZOS)
                if progress_callback:
                    await progress_callback(20, "Image loaded and processed...")
            
            # Calculate number of frames (8 fps)
            num_frames = min(duration_seconds * 8, 80)  # Cap at 80 frames
            
            logger.info(f"Generating test video: '{prompt}', {num_frames} frames, has_image={base_image is not None}")
            
            if progress_callback:
                await progress_callback(30, f"Creating {num_frames} frames...")
            
            # Generate video
            await loop.run_in_executor(
                self.executor,
                self._generate_video_sync,
                prompt,
                output_path,
                num_frames,
                base_image
            )
            
            if progress_callback:
                await progress_callback(90, "Finalizing video...")
            
            if progress_callback:
                await progress_callback(100, "Test video complete!")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating test video: {str(e)}")
            raise
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None):
        """Generate video with progress tracking"""
        try:
            await progress_manager.update_progress(job_id, 1, "Starting test video generation...")
            
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