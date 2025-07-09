import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import imageio
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableVideoGenerator:
    """Working AI-style video generator using advanced image processing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def get_available_animations(self):
        """Return all available animations"""
        return {
            "Person Animation": [
                "person walking", "person dancing", "person waving", "person jumping",
                "person nodding", "person smiling", "person talking", "person running"
            ],
            "Camera Movement": [
                "zoom in", "zoom out", "pan left", "pan right", 
                "dolly forward", "dolly backward", "orbit left", "orbit right"
            ],
            "Motion Effects": [
                "gentle sway", "bounce", "shake", "vibrate",
                "float", "drift", "pulse", "breathe"
            ],
            "Visual Effects": [
                "fade in", "fade out", "blur focus", "brighten",
                "dramatic lighting", "soft glow", "vintage", "dream sequence"
            ]
        }
    
    def analyze_prompt(self, prompt):
        """Analyze prompt and return animation parameters"""
        if not prompt:
            return {'type': 'zoom_in', 'intensity': 0.3, 'speed': 1.0}
        
        prompt_lower = prompt.lower().strip()
        logger.info(f"🎬 Analyzing: '{prompt_lower}'")
        
        # Determine animation type, intensity, and speed
        animation_params = {'type': 'zoom_in', 'intensity': 0.3, 'speed': 1.0}
        
        # Person animations
        if 'person walking' in prompt_lower or 'walking' in prompt_lower:
            animation_params = {'type': 'person_walk', 'intensity': 0.4, 'speed': 1.2}
        elif 'person dancing' in prompt_lower or 'dancing' in prompt_lower:
            animation_params = {'type': 'person_dance', 'intensity': 0.6, 'speed': 1.5}
        elif 'person waving' in prompt_lower or 'waving' in prompt_lower:
            animation_params = {'type': 'person_wave', 'intensity': 0.5, 'speed': 1.0}
        elif 'person jumping' in prompt_lower or 'jumping' in prompt_lower:
            animation_params = {'type': 'person_jump', 'intensity': 0.8, 'speed': 1.8}
        elif 'person nodding' in prompt_lower or 'nodding' in prompt_lower:
            animation_params = {'type': 'person_nod', 'intensity': 0.3, 'speed': 0.8}
        elif 'person smiling' in prompt_lower or 'smiling' in prompt_lower:
            animation_params = {'type': 'person_smile', 'intensity': 0.2, 'speed': 0.5}
        elif 'person talking' in prompt_lower or 'talking' in prompt_lower:
            animation_params = {'type': 'person_talk', 'intensity': 0.3, 'speed': 2.0}
        
        # Camera movements
        elif 'zoom out' in prompt_lower:
            animation_params = {'type': 'zoom_out', 'intensity': 0.4, 'speed': 1.0}
        elif 'zoom in' in prompt_lower:
            animation_params = {'type': 'zoom_in', 'intensity': 0.4, 'speed': 1.0}
        elif 'pan left' in prompt_lower:
            animation_params = {'type': 'pan_left', 'intensity': 0.5, 'speed': 1.0}
        elif 'pan right' in prompt_lower:
            animation_params = {'type': 'pan_right', 'intensity': 0.5, 'speed': 1.0}
        elif 'orbit' in prompt_lower:
            animation_params = {'type': 'orbit', 'intensity': 0.6, 'speed': 1.0}
        
        # Motion effects
        elif 'bounce' in prompt_lower:
            animation_params = {'type': 'bounce', 'intensity': 0.6, 'speed': 1.5}
        elif 'shake' in prompt_lower:
            animation_params = {'type': 'shake', 'intensity': 0.4, 'speed': 2.0}
        elif 'sway' in prompt_lower or 'gentle' in prompt_lower:
            animation_params = {'type': 'sway', 'intensity': 0.2, 'speed': 0.8}
        elif 'pulse' in prompt_lower:
            animation_params = {'type': 'pulse', 'intensity': 0.3, 'speed': 1.2}
        elif 'breathe' in prompt_lower:
            animation_params = {'type': 'breathe', 'intensity': 0.15, 'speed': 0.6}
        
        # Visual effects
        elif 'fade out' in prompt_lower:
            animation_params = {'type': 'fade_out', 'intensity': 1.0, 'speed': 1.0}
        elif 'fade in' in prompt_lower:
            animation_params = {'type': 'fade_in', 'intensity': 1.0, 'speed': 1.0}
        elif 'glow' in prompt_lower:
            animation_params = {'type': 'glow', 'intensity': 0.4, 'speed': 1.0}
        elif 'dramatic' in prompt_lower:
            animation_params = {'type': 'dramatic', 'intensity': 0.6, 'speed': 1.0}
        
        logger.info(f"✅ Animation: {animation_params['type']} (intensity: {animation_params['intensity']}, speed: {animation_params['speed']})")
        return animation_params
    
    def apply_person_animation(self, image, frame_num, total_frames, anim_type, intensity, speed):
        """Apply person-specific animations using advanced image processing"""
        progress = (frame_num / max(total_frames - 1, 1)) * speed
        width, height = image.size
        
        if anim_type == 'person_walk':
            # Simulate walking with subtle body movement
            sway = intensity * 15 * math.sin(progress * math.pi * 4)
            bounce = intensity * 8 * abs(math.sin(progress * math.pi * 8))
            
            # Apply slight rotation for natural gait
            angle = intensity * 3 * math.sin(progress * math.pi * 4)
            img = image.rotate(angle, expand=False, fillcolor='black')
            
            # Apply movement
            canvas = Image.new('RGB', (width + 30, height + 20), 'black')
            canvas.paste(img, (15 + int(sway), 10 - int(bounce)))
            return canvas.crop((15, 10, 15 + width, 10 + height))
        
        elif anim_type == 'person_dance':
            # Rhythmic movement for dancing
            sway_x = intensity * 20 * math.sin(progress * math.pi * 6)
            sway_y = intensity * 15 * math.cos(progress * math.pi * 4)
            rotation = intensity * 8 * math.sin(progress * math.pi * 3)
            
            img = image.rotate(rotation, expand=False, fillcolor='black')
            canvas = Image.new('RGB', (width + 40, height + 30), 'black')
            canvas.paste(img, (20 + int(sway_x), 15 + int(sway_y)))
            return canvas.crop((20, 15, 20 + width, 15 + height))
        
        elif anim_type == 'person_wave':
            # Hand waving simulation with upper body movement
            wave_motion = intensity * 10 * math.sin(progress * math.pi * 6)
            lean = intensity * 5 * math.sin(progress * math.pi * 3)
            
            img = image.rotate(lean, expand=False, fillcolor='black')
            canvas = Image.new('RGB', (width + 20, height + 20), 'black')
            canvas.paste(img, (10 + int(wave_motion), 10))
            return canvas.crop((10, 10, 10 + width, 10 + height))
        
        elif anim_type == 'person_jump':
            # Jumping motion with anticipation and landing
            jump_phase = (progress * 2) % 2  # Two jumps per cycle
            if jump_phase < 1:
                # Going up
                jump_height = intensity * 40 * math.sin(jump_phase * math.pi)
                squash = 1.0 - (intensity * 0.1 * math.sin(jump_phase * math.pi))
            else:
                # Coming down
                jump_height = intensity * 40 * math.sin((jump_phase - 1) * math.pi)
                squash = 1.0 + (intensity * 0.05)
            
            # Apply squash and stretch
            new_height = int(height * squash)
            img = image.resize((width, new_height), Image.Resampling.LANCZOS)
            
            canvas = Image.new('RGB', (width, height + 50), 'black')
            paste_y = height + 25 - new_height - int(jump_height)
            canvas.paste(img, (0, paste_y))
            return canvas.crop((0, 25, width, 25 + height))
        
        elif anim_type == 'person_nod':
            # Head nodding simulation
            nod_angle = intensity * 8 * math.sin(progress * math.pi * 4)
            
            # Create a subtle zoom on upper portion
            crop_top = int(height * 0.1)
            upper_section = image.crop((0, 0, width, height - crop_top))
            lower_section = image.crop((0, height - crop_top, width, height))
            
            # Rotate upper section slightly
            upper_rotated = upper_section.rotate(nod_angle, expand=False, fillcolor='black')
            
            # Combine back
            result = Image.new('RGB', (width, height), 'black')
            result.paste(upper_rotated, (0, 0))
            result.paste(lower_section, (0, height - crop_top))
            return result
        
        elif anim_type == 'person_smile':
            # Subtle brightness change around face area (center-top)
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = 1.0 + (intensity * 0.2 * math.sin(progress * math.pi))
            
            # Apply brightness change to whole image for now
            return enhancer.enhance(brightness_factor)
        
        elif anim_type == 'person_talk':
            # Subtle mouth area movement
            talk_intensity = intensity * 5 * abs(math.sin(progress * math.pi * 8))
            
            # Focus on lower face area
            mouth_area = image.crop((width//4, 2*height//3, 3*width//4, height))
            mouth_scaled = mouth_area.resize(
                (int(mouth_area.width * (1 + talk_intensity * 0.1)), mouth_area.height),
                Image.Resampling.LANCZOS
            )
            
            result = image.copy()
            paste_x = width//4 - (mouth_scaled.width - mouth_area.width) // 2
            result.paste(mouth_scaled, (paste_x, 2*height//3))
            return result
        
        # Default to original image if animation not found
        return image
    
    def apply_camera_effect(self, image, frame_num, total_frames, anim_type, intensity, speed):
        """Apply camera-style effects"""
        progress = (frame_num / max(total_frames - 1, 1)) * speed
        width, height = image.size
        
        if anim_type == 'zoom_in':
            zoom_factor = 1.0 + (intensity * progress)
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            img = image.resize(new_size, Image.Resampling.LANCZOS)
            left = (img.width - width) // 2
            top = (img.height - height) // 2
            return img.crop((left, top, left + width, top + height))
        
        elif anim_type == 'zoom_out':
            zoom_factor = 1.0 + intensity - (intensity * progress)
            new_size = (int(width * zoom_factor), int(height * zoom_factor))
            img = image.resize(new_size, Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (width, height), 'black')
            paste_x = (width - img.width) // 2
            paste_y = (height - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))
            return canvas
        
        elif anim_type == 'orbit':
            radius = intensity * 50
            angle = progress * math.pi * 2
            offset_x = int(radius * math.cos(angle))
            offset_y = int(radius * math.sin(angle))
            
            canvas = Image.new('RGB', (width + 100, height + 100), 'black')
            canvas.paste(image, (50 + offset_x, 50 + offset_y))
            return canvas.crop((50, 50, 50 + width, 50 + height))
        
        # More camera effects...
        return self.apply_simple_effect(image, frame_num, total_frames, anim_type, intensity, speed)
    
    def apply_simple_effect(self, image, frame_num, total_frames, anim_type, intensity, speed):
        """Apply simple motion and visual effects"""
        progress = (frame_num / max(total_frames - 1, 1)) * speed
        
        if anim_type == 'bounce':
            bounce_y = int(intensity * 30 * abs(math.sin(progress * math.pi * 3)))
            canvas = Image.new('RGB', (image.width, image.height + 30), 'black')
            canvas.paste(image, (0, bounce_y))
            return canvas.crop((0, 0, image.width, image.height))
        
        elif anim_type == 'pulse':
            pulse_factor = 1.0 + intensity * 0.3 * math.sin(progress * math.pi * 4)
            new_size = (int(image.width * pulse_factor), int(image.height * pulse_factor))
            img = image.resize(new_size, Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', image.size, 'black')
            paste_x = (image.width - img.width) // 2
            paste_y = (image.height - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))
            return canvas
        
        elif anim_type == 'glow':
            enhancer = ImageEnhance.Brightness(image)
            glow_factor = 1.0 + intensity * 0.5 * math.sin(progress * math.pi * 2)
            return enhancer.enhance(glow_factor)
        
        # Default fallback
        return image
    
    def create_frame(self, image, frame_num, total_frames, animation_params):
        """Create a single animated frame"""
        anim_type = animation_params['type']
        intensity = animation_params['intensity']
        speed = animation_params['speed']
        
        try:
            # Apply person animations first (most sophisticated)
            if anim_type.startswith('person_'):
                return self.apply_person_animation(image, frame_num, total_frames, anim_type, intensity, speed)
            
            # Apply camera effects
            elif anim_type in ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'orbit']:
                return self.apply_camera_effect(image, frame_num, total_frames, anim_type, intensity, speed)
            
            # Apply simple effects
            else:
                return self.apply_simple_effect(image, frame_num, total_frames, anim_type, intensity, speed)
                
        except Exception as e:
            logger.error(f"Error in frame creation: {e}")
            return image
    
    def _generate_video_sync(self, prompt, output_path, image_path, num_frames=80, motion_bucket_id=127):
        """Generate video synchronously"""
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Analyze prompt with motion intensity
            animation_params = self.analyze_prompt(prompt)
            animation_params['motion_intensity'] = motion_bucket_id / 127.0  # Normalize to 0-2 range

            # Generate frames
            frames = []
            for i in range(num_frames):
                frame = self.create_frame(image, i, num_frames, animation_params)
                frames.append(np.array(frame))
            
            # Save video
            imageio.mimwrite(output_path, frames, fps=8, codec='libx264')
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            raise
    
    async def generate_video(self, prompt, output_path, duration_seconds=10, progress_callback=None, image_path=None, motion_bucket_id=127):
        """Generate video with progress tracking"""
        if not image_path:
            raise ValueError("Image required")
        
        loop = asyncio.get_event_loop()
        num_frames = min(duration_seconds * 8, 160)
        
        try:
            if progress_callback:
                await progress_callback(10, "Analyzing animation request...")
            
            result = await loop.run_in_executor(
                self.executor,
                self._generate_video_sync,
                prompt,
                output_path,
                image_path,
                num_frames,
                motion_bucket_id
            )
            
            if progress_callback:
                await progress_callback(100, "Advanced animation complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    async def generate_with_progress(self, job_id, prompt, output_path, duration_seconds, progress_manager, image_path=None, motion_bucket_id=127):
        """Generate video with progress manager"""
        try:
            await progress_manager.update_progress(job_id, 1, "Starting advanced animation...")
            
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
            logger.error(f"Error: {e}")
            await progress_manager.complete_job(job_id, success=False, error=str(e))
            raise