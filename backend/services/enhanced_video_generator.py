import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
try:
    from PIL import ImageChops
except ImportError:
    ImageChops = None
import imageio
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVideoGenerator:
    """Enhanced video generator that understands text prompts and creates realistic animations"""

    def __init__(self, use_nsfw_model=False):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.use_nsfw_model = use_nsfw_model
        self.nsfw_generator = None

        # Initialize NSFW generator if requested
        if use_nsfw_model:
            self._initialize_nsfw_generator()

    def _initialize_nsfw_generator(self):
        """Initialize the NSFW text-to-video generator"""
        try:
            from .nsfw_text_to_video_generator import NSFWTextToVideoGenerator
            self.nsfw_generator = NSFWTextToVideoGenerator()
            logger.info("NSFW generator initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import NSFW generator: {e}")
            self.use_nsfw_model = False
            self.nsfw_generator = None
        except Exception as e:
            logger.error(f"Failed to initialize NSFW generator: {e}")
            self.use_nsfw_model = False
            self.nsfw_generator = None

    def switch_model(self, use_nsfw=False):
        """Switch between NSFW and standard models"""
        if use_nsfw != self.use_nsfw_model:
            self.use_nsfw_model = use_nsfw
            if use_nsfw and self.nsfw_generator is None:
                self._initialize_nsfw_generator()
            elif not use_nsfw and self.nsfw_generator is not None:
                if hasattr(self.nsfw_generator, 'cleanup'):
                    self.nsfw_generator.cleanup()
                self.nsfw_generator = None
            logger.info(f"Switched to {'NSFW' if use_nsfw else 'standard'} model")

    def analyze_prompt(self, prompt):
        """Analyze prompt to understand requested effects"""
        if not prompt:
            return {'zoom_in': True, 'speed': 'normal'}
            
        prompt_lower = prompt.lower().strip()
        print(f"Analyzing prompt: '{prompt_lower}'")
        
        effects = {
            # Movement effects - more specific patterns
            'zoom_out': any(phrase in prompt_lower for phrase in ['zoom out', 'pull back', 'pull away', 'move away', 'back away', 'wider', 'distant']),
            'zoom_in': any(phrase in prompt_lower for phrase in ['zoom in', 'close up', 'closer', 'approach', 'move in']) and not any(phrase in prompt_lower for phrase in ['zoom out', 'pull back', 'pull away', 'move away', 'back away', 'wider', 'distant']),
            'pan_left': any(phrase in prompt_lower for phrase in ['pan left', 'move left', 'slide left', 'sweep left', 'go left', 'right to left']),
            'pan_right': any(phrase in prompt_lower for phrase in ['pan right', 'move right', 'slide right', 'sweep right', 'go right', 'left to right', 'across']),
            'rotate_cw': any(phrase in prompt_lower for phrase in ['rotate right', 'clockwise', 'spin right', 'turn right']),
            'rotate_ccw': any(phrase in prompt_lower for phrase in ['rotate left', 'counterclockwise', 'spin left', 'turn left']),
            'rotate': any(phrase in prompt_lower for phrase in ['rotate', 'spin', 'turn around', 'revolve']),
            'tilt': any(phrase in prompt_lower for phrase in ['tilt', 'angle', 'lean', 'slant']),
            'shake': any(phrase in prompt_lower for phrase in ['shake', 'vibrate', 'tremor', 'wobble', 'jitter']),
            'drift': any(phrase in prompt_lower for phrase in ['drift', 'float', 'sway', 'gentle motion', 'subtle movement']),

            # Dance and dynamic movement effects
            'dance': any(phrase in prompt_lower for phrase in ['dancing', 'dance', 'dancing person', 'dancer', 'choreography', 'ballet', 'hip hop', 'salsa', 'tango', 'waltz']),
            'bounce': any(phrase in prompt_lower for phrase in ['bounce', 'bouncing', 'jump', 'jumping', 'hop', 'hopping', 'spring']),
            'wave': any(phrase in prompt_lower for phrase in ['wave', 'waving', 'flowing', 'undulate', 'ripple', 'fluid motion']),
            'pulse': any(phrase in prompt_lower for phrase in ['pulse', 'pulsing', 'beat', 'rhythm', 'throb', 'heartbeat']),
            'swing': any(phrase in prompt_lower for phrase in ['swing', 'swinging', 'pendulum', 'rock', 'rocking']),
            
            # Visual effects
            'fade_in': any(phrase in prompt_lower for phrase in ['fade in', 'appear', 'emerge', 'materialize']),
            'fade_out': any(phrase in prompt_lower for phrase in ['fade out', 'disappear', 'dissolve', 'vanish']),
            'blur': any(phrase in prompt_lower for phrase in ['blur', 'unfocus', 'defocus', 'soft focus']),
            'brighten': any(phrase in prompt_lower for phrase in ['brighten', 'brighter', 'illuminate', 'light up']),
            'darken': any(phrase in prompt_lower for phrase in ['darken', 'darker', 'shadow', 'dim']),
            'contrast': any(phrase in prompt_lower for phrase in ['contrast', 'bold', 'sharp']),
            'vintage': any(phrase in prompt_lower for phrase in ['vintage', 'old', 'sepia', 'aged', 'retro']),
            'glow': any(phrase in prompt_lower for phrase in ['glow', 'radiant', 'luminous', 'shine']),
        }
        
        # Speed detection
        if any(word in prompt_lower for word in ['slow', 'slowly', 'gentle', 'smooth', 'gradual', 'subtle']):
            effects['speed'] = 'slow'
        elif any(word in prompt_lower for word in ['fast', 'quick', 'rapid', 'swift', 'quickly']):
            effects['speed'] = 'fast'
        else:
            effects['speed'] = 'normal'
            
        # Style detection
        if any(word in prompt_lower for word in ['cinematic', 'movie', 'film']):
            effects['style'] = 'cinematic'
        elif any(word in prompt_lower for word in ['dreamy', 'ethereal', 'soft', 'peaceful']):
            effects['style'] = 'dreamy'
        elif any(word in prompt_lower for word in ['dramatic', 'intense', 'powerful']):
            effects['style'] = 'dramatic'
        else:
            effects['style'] = 'normal'
        
        # Debug: print detected effects
        detected = [k for k, v in effects.items() if v and k not in ['speed', 'style']]
        print(f"Detected effects: {detected}")
        print(f"Speed: {effects.get('speed')}, Style: {effects.get('style')}")
            
        # If no specific movement effect detected, check for any keywords that suggest no zoom
        no_zoom_keywords = ['still', 'static', 'no movement', 'stationary', 'freeze']
        movement_keys = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'rotate', 'rotate_cw', 'rotate_ccw', 'tilt', 'shake', 'drift', 'dance', 'bounce', 'wave', 'pulse', 'swing']

        if any(keyword in prompt_lower for keyword in no_zoom_keywords):
            effects['drift'] = True  # Use drift for subtle motion
        elif not any(effects[key] for key in movement_keys):
            # Only default to zoom in if absolutely no other movement is detected
            print("No movement detected, defaulting to subtle drift")
            effects['drift'] = True
            
        return effects
    
    def apply_zoom_effect(self, image, progress, zoom_in=True, speed_mult=1.0):
        """Apply zoom effect to image"""
        if zoom_in:
            # Zoom in: start at 1.0, end at 1.5
            zoom_factor = 1.0 + (0.5 * progress * speed_mult)
        else:
            # Zoom out: start at 1.3, end at 1.0
            zoom_factor = 1.3 - (0.3 * progress * speed_mult)
            zoom_factor = max(0.7, zoom_factor)  # Don't zoom out too much
            
        width, height = image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if zoom_in:
            # Crop from center for zoom in
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return resized.crop((left, top, left + width, top + height))
        else:
            # Place on black background for zoom out
            result = Image.new('RGB', (width, height), 'black')
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            result.paste(resized, (paste_x, paste_y))
            return result
    
    def apply_pan_effect(self, image, progress, direction, speed_mult=1.0):
        """Apply panning effect"""
        width, height = image.size
        
        # Create a larger canvas for smooth panning
        canvas_width = int(width * 1.5)
        canvas_height = int(height * 1.5)
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'black')
        
        # Place original image in center
        start_x = (canvas_width - width) // 2
        start_y = (canvas_height - height) // 2
        canvas.paste(image, (start_x, start_y))
        
        # Calculate pan distance
        max_pan = width // 3
        pan_distance = int(max_pan * progress * speed_mult)
        
        if direction == 'left':
            crop_x = start_x - pan_distance
        elif direction == 'right':
            crop_x = start_x + pan_distance
        else:
            crop_x = start_x
            
        crop_x = max(0, min(crop_x, canvas_width - width))
        
        return canvas.crop((crop_x, start_y, crop_x + width, start_y + height))
    
    def apply_rotation_effect(self, image, progress, clockwise=True, speed_mult=1.0):
        """Apply rotation effect"""
        max_angle = 360 if 'spin' in str(self.current_prompt).lower() else 15
        angle = max_angle * progress * speed_mult
        if not clockwise:
            angle = -angle
            
        return image.rotate(angle, expand=False, fillcolor='black', resample=Image.Resampling.BICUBIC)
    
    def apply_shake_effect(self, image, frame_num, speed_mult=1.0):
        """Apply shake/vibrate effect"""
        width, height = image.size
        
        # Create larger canvas
        canvas = Image.new('RGB', (width + 20, height + 20), 'black')
        
        # Calculate shake offset
        shake_intensity = 8 * speed_mult
        shake_x = int(shake_intensity * math.sin(frame_num * 0.8))
        shake_y = int(shake_intensity * math.cos(frame_num * 0.6))
        
        # Paste with offset
        paste_x = 10 + shake_x
        paste_y = 10 + shake_y
        canvas.paste(image, (paste_x, paste_y))
        
        # Crop back to original size
        return canvas.crop((10, 10, 10 + width, 10 + height))
    
    def apply_visual_effects(self, image, progress, effects):
        """Apply visual effects like fade, blur, brightness"""
        result = image.copy()
        
        # Fade effects
        if effects.get('fade_in'):
            alpha = progress
            black = Image.new('RGB', image.size, 'black')
            result = Image.blend(black, result, alpha)
        elif effects.get('fade_out'):
            alpha = 1.0 - progress
            white = Image.new('RGB', image.size, 'white')
            result = Image.blend(result, white, 1.0 - alpha)
        
        # Blur effect
        if effects.get('blur'):
            blur_amount = 2 * math.sin(progress * math.pi)  # Blur in and out
            if blur_amount > 0.1:
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        # Brightness effects
        if effects.get('brighten'):
            enhancer = ImageEnhance.Brightness(result)
            brightness = 1.0 + (0.3 * progress)
            result = enhancer.enhance(brightness)
        elif effects.get('darken'):
            enhancer = ImageEnhance.Brightness(result)
            brightness = 1.0 - (0.3 * progress)
            result = enhancer.enhance(max(0.3, brightness))
        
        # Contrast effect
        if effects.get('contrast'):
            enhancer = ImageEnhance.Contrast(result)
            contrast = 1.0 + (0.5 * progress)
            result = enhancer.enhance(contrast)
        
        # Vintage effect
        if effects.get('vintage'):
            # Convert to sepia-like tones
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(0.6)  # Desaturate
            
            # Add sepia tint
            sepia = result.convert('RGB')
            pixels = sepia.load()
            for i in range(sepia.width):
                for j in range(sepia.height):
                    r, g, b = pixels[i, j]
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    pixels[i, j] = (min(255, tr), min(255, tg), min(255, tb))
            result = sepia
        
        # Glow effect
        if effects.get('glow') and ImageChops:
            # Create a soft glow
            glow = result.filter(ImageFilter.GaussianBlur(radius=3))
            enhancer = ImageEnhance.Brightness(glow)
            glow = enhancer.enhance(1.5)
            result = ImageChops.screen(result, glow)
        elif effects.get('glow'):
            # Simple brightness increase if ImageChops not available
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(1.2)
        
        return result
    
    def apply_style_effects(self, image, effects):
        """Apply style-specific effects"""
        result = image.copy()
        
        if effects.get('style') == 'cinematic':
            # Add letterbox bars
            width, height = result.size
            bar_height = int(height * 0.1)
            draw = ImageDraw.Draw(result)
            draw.rectangle([(0, 0), (width, bar_height)], fill='black')
            draw.rectangle([(0, height - bar_height), (width, height)], fill='black')
            
            # Increase contrast
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.2)
            
        elif effects.get('style') == 'dreamy':
            # Soft focus
            result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
            # Reduce saturation slightly
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(0.9)
            
        elif effects.get('style') == 'dramatic':
            # High contrast and saturation
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.2)
        
        return result
    
    def create_animated_frame(self, base_image, frame_num, total_frames, effects):
        """Create a single animated frame based on effects"""
        progress = frame_num / max(total_frames - 1, 1)
        speed_mult = {'slow': 0.6, 'normal': 1.0, 'fast': 1.6}[effects.get('speed', 'normal')]
        
        # Start with base image
        result = base_image.copy()
        
        # Apply movement effects
        if effects.get('zoom_in'):
            result = self.apply_zoom_effect(result, progress, zoom_in=True, speed_mult=speed_mult)
        elif effects.get('zoom_out'):
            result = self.apply_zoom_effect(result, progress, zoom_in=False, speed_mult=speed_mult)
        
        if effects.get('pan_left'):
            result = self.apply_pan_effect(result, progress, 'left', speed_mult)
        elif effects.get('pan_right'):
            result = self.apply_pan_effect(result, progress, 'right', speed_mult)
        
        if effects.get('rotate') or effects.get('rotate_cw'):
            result = self.apply_rotation_effect(result, progress, clockwise=True, speed_mult=speed_mult)
        elif effects.get('rotate_ccw'):
            result = self.apply_rotation_effect(result, progress, clockwise=False, speed_mult=speed_mult)
        
        if effects.get('tilt'):
            # Gentle tilt back and forth
            angle = 10 * math.sin(progress * math.pi * 2) * speed_mult
            result = result.rotate(angle, expand=False, fillcolor='black')
        
        if effects.get('shake'):
            result = self.apply_shake_effect(result, frame_num, speed_mult)
        
        if effects.get('drift'):
            # Gentle floating motion
            drift_x = int(5 * math.sin(progress * math.pi * 2) * speed_mult)
            drift_y = int(3 * math.cos(progress * math.pi * 3) * speed_mult)
            canvas = Image.new('RGB', (result.width + 10, result.height + 10), 'black')
            canvas.paste(result, (5 + drift_x, 5 + drift_y))
            result = canvas.crop((5, 5, 5 + result.width, 5 + result.height))

        # Dance and dynamic movement effects
        if effects.get('dance'):
            # Complex dance motion: combine rotation, scaling, and translation
            motion_intensity = effects.get('motion_intensity', 1.0)

            # Rhythmic rotation (like body swaying)
            dance_angle = 15 * math.sin(progress * math.pi * 4) * speed_mult * motion_intensity
            result = result.rotate(dance_angle, expand=False, fillcolor='black')

            # Bouncing motion (up and down)
            bounce_y = int(20 * abs(math.sin(progress * math.pi * 6)) * speed_mult * motion_intensity)

            # Side-to-side motion
            sway_x = int(15 * math.sin(progress * math.pi * 3) * speed_mult * motion_intensity)

            # Apply the movement
            canvas = Image.new('RGB', (result.width + 40, result.height + 40), 'black')
            canvas.paste(result, (20 + sway_x, 20 - bounce_y))
            result = canvas.crop((20, 20, 20 + result.width, 20 + result.height))

            # Slight scale pulsing (like breathing while dancing)
            scale_factor = 1.0 + 0.1 * math.sin(progress * math.pi * 8) * motion_intensity
            new_size = (int(result.width * scale_factor), int(result.height * scale_factor))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            if scale_factor > 1.0:
                # Crop to original size
                crop_x = (result.width - 512) // 2
                crop_y = (result.height - 512) // 2
                result = result.crop((crop_x, crop_y, crop_x + 512, crop_y + 512))
            else:
                # Pad to original size
                canvas = Image.new('RGB', (512, 512), 'black')
                paste_x = (512 - result.width) // 2
                paste_y = (512 - result.height) // 2
                canvas.paste(result, (paste_x, paste_y))
                result = canvas

        if effects.get('bounce'):
            # Bouncing motion
            motion_intensity = effects.get('motion_intensity', 1.0)
            bounce_y = int(30 * abs(math.sin(progress * math.pi * 4)) * speed_mult * motion_intensity)
            canvas = Image.new('RGB', (result.width, result.height + 30), 'black')
            canvas.paste(result, (0, 30 - bounce_y))
            result = canvas.crop((0, 0, result.width, result.height))

        if effects.get('wave'):
            # Wave-like motion
            motion_intensity = effects.get('motion_intensity', 1.0)
            wave_x = int(20 * math.sin(progress * math.pi * 3) * speed_mult * motion_intensity)
            wave_y = int(10 * math.sin(progress * math.pi * 5) * speed_mult * motion_intensity)
            canvas = Image.new('RGB', (result.width + 40, result.height + 20), 'black')
            canvas.paste(result, (20 + wave_x, 10 + wave_y))
            result = canvas.crop((20, 10, 20 + result.width, 10 + result.height))

        if effects.get('pulse'):
            # Pulsing scale effect
            motion_intensity = effects.get('motion_intensity', 1.0)
            scale_factor = 1.0 + 0.2 * math.sin(progress * math.pi * 6) * motion_intensity
            new_size = (int(result.width * scale_factor), int(result.height * scale_factor))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
            # Center crop/pad to maintain size
            if scale_factor > 1.0:
                crop_x = (result.width - 512) // 2
                crop_y = (result.height - 512) // 2
                result = result.crop((crop_x, crop_y, crop_x + 512, crop_y + 512))
            else:
                canvas = Image.new('RGB', (512, 512), 'black')
                paste_x = (512 - result.width) // 2
                paste_y = (512 - result.height) // 2
                canvas.paste(result, (paste_x, paste_y))
                result = canvas

        if effects.get('swing'):
            # Pendulum-like swinging motion
            motion_intensity = effects.get('motion_intensity', 1.0)
            swing_angle = 25 * math.sin(progress * math.pi * 2) * speed_mult * motion_intensity
            result = result.rotate(swing_angle, expand=False, fillcolor='black')

        # Apply visual effects
        result = self.apply_visual_effects(result, progress, effects)
        
        # Apply style effects
        result = self.apply_style_effects(result, effects)
        
        return result
    
    def _generate_video_sync(self, image_path, prompt, num_frames=80, motion_bucket_id=127):
        """Generate video frames with advanced prompt-based animation"""
        self.current_prompt = prompt  # Store for reference
        
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Analyze prompt to determine effects
        effects = self.analyze_prompt(prompt)
        # Apply motion intensity from motion_bucket_id (127 = normal, higher = more motion)
        effects['motion_intensity'] = motion_bucket_id / 127.0
        logger.info(f"Applied effects for prompt '{prompt}': {[k for k, v in effects.items() if v and k != 'speed' and k != 'style']}, motion_intensity: {effects['motion_intensity']:.2f}")
        
        frames = []
        for i in range(num_frames):
            frame = self.create_animated_frame(image, i, num_frames, effects)
            frames.append(np.array(frame))
        
        return frames
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None, motion_bucket_id=127):
        """Generate an animated video from image and prompt, or text-to-video if NSFW model is enabled"""
        loop = asyncio.get_event_loop()

        try:
            # If NSFW model is enabled and no image provided, use text-to-video
            if self.use_nsfw_model and self.nsfw_generator and not image_path:
                logger.info(f"Using NSFW text-to-video generation for: '{prompt}'")

                if progress_callback:
                    await progress_callback(5, "Using NSFW-gen-v2 for text-to-video...")

                # Calculate parameters for NSFW generator
                num_frames = min(duration_seconds * 7, 25)  # 7 FPS, max 25 frames

                return await self.nsfw_generator.generate_from_text(
                    prompt=prompt,
                    output_path=output_path,
                    num_frames=num_frames
                )

            # Standard image-to-video generation
            if not image_path:
                raise ValueError("Image is required for standard video generation. Enable NSFW model for text-to-video.")

            if progress_callback:
                await progress_callback(10, "Analyzing your animation request...")

            # Calculate frames (8 fps)
            num_frames = min(duration_seconds * 8, 160)  # Cap at 160 frames (20 seconds)

            logger.info(f"Generating video: '{prompt}', {num_frames} frames")

            if progress_callback:
                await progress_callback(30, "Creating animation frames...")

            # Generate frames
            frames = await loop.run_in_executor(
                self.executor,
                self._generate_video_sync,
                image_path,
                prompt,
                num_frames,
                motion_bucket_id
            )

            if progress_callback:
                await progress_callback(80, "Encoding video...")

            # Save video
            imageio.mimwrite(output_path, frames, fps=8, codec='libx264')

            if progress_callback:
                await progress_callback(100, "Animation complete!")

            return output_path

        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None, motion_bucket_id=127):
        """Generate video with progress tracking"""
        try:
            # If NSFW model is enabled and no image provided, use text-to-video with progress
            if self.use_nsfw_model and self.nsfw_generator and not image_path:
                logger.info(f"Using NSFW text-to-video generation with progress for: '{prompt}'")

                # Calculate parameters
                num_frames = min(duration_seconds * 7, 25)  # 7 FPS, max 25 frames
                # Use the motion_bucket_id parameter passed in
                noise_aug_strength = 0.02
                fps = 7

                result = await self.nsfw_generator.generate_from_text_with_progress(
                    job_id=job_id,
                    prompt=prompt,
                    output_path=output_path,
                    num_frames=num_frames,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    fps=fps,
                    progress_manager=progress_manager
                )

                await progress_manager.complete_job(job_id, success=True)
                return result

            # Standard image-to-video generation with progress
            await progress_manager.update_progress(job_id, 1, "Starting enhanced video generation...")

            async def progress_callback(progress, message):
                await progress_manager.update_progress(job_id, int(progress), message)

            await self.generate_video(
                prompt=prompt,
                output_path=output_path,
                duration_seconds=duration_seconds,
                progress_callback=progress_callback,
                image_path=image_path,
                motion_bucket_id=motion_bucket_id
            )

            await progress_manager.complete_job(job_id, success=True)
            return output_path

        except Exception as e:
            logger.error(f"Error in generate_with_progress: {str(e)}")
            await progress_manager.complete_job(job_id, success=False, error=str(e))
            raise

    async def generate_image_only(self, prompt, output_path, width=1024, height=576):
        """Generate only an image from text prompt (requires NSFW model)"""
        if not self.use_nsfw_model or not self.nsfw_generator:
            raise ValueError("NSFW model must be enabled for image-only generation")

        return await self.nsfw_generator.generate_image_only(prompt, output_path, width, height)

    def get_model_info(self):
        """Get information about the current model configuration"""
        return {
            "enhanced_generator": True,
            "nsfw_model_enabled": self.use_nsfw_model,
            "nsfw_model_loaded": self.nsfw_generator is not None,
            "supports_text_to_video": self.use_nsfw_model and self.nsfw_generator is not None,
            "supports_image_to_video": True,
            "supports_image_only": self.use_nsfw_model and self.nsfw_generator is not None
        }

    def cleanup(self):
        """Clean up resources"""
        if self.nsfw_generator and hasattr(self.nsfw_generator, 'cleanup'):
            self.nsfw_generator.cleanup()
        self.executor.shutdown(wait=True)
        logger.info("Enhanced video generator cleaned up")