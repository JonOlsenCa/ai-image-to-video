import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import sys

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create a simple test image
if len(sys.argv) > 1:
    image = Image.open(sys.argv[1])
else:
    # Create a test image
    image = Image.new('RGB', (1024, 576), color='red')
    image.save('test_input.png')

print("Loading model...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

print("Generating video...")
generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, num_frames=5).frames[0]

print("Saving video...")
export_to_video(frames, "test_output.mp4", fps=7)
print("Done! Video saved as test_output.mp4")