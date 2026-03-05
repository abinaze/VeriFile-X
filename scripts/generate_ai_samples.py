"""
Generate AI reference images for CLIP embedding database.

Since we don't have access to paid AI generation APIs in this script,
we'll create synthetic test images that simulate AI characteristics.

For production, you would:
1. Use Stable Diffusion to generate images
2. Download samples from Midjourney showcases
3. Use DALL-E API to generate diverse images

This script creates test images with AI-like patterns.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/reference/ai")
NUM_SAMPLES = 500


def generate_smooth_gradient(size=(512, 512)):
    """Generate smooth gradient image (common in AI)."""
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    
    for i in range(size[0]):
        r = int(255 * i / size[0])
        g = int(255 * (1 - i / size[0]))
        b = 128
        draw.line([(i, 0), (i, size[1])], fill=(r, g, b))
    
    return img.filter(ImageFilter.GaussianBlur(radius=5))


def generate_symmetric_pattern(size=(512, 512)):
    """Generate symmetric pattern (common in AI art)."""
    img = Image.new('RGB', size)
    pixels = np.random.randint(0, 256, (size[0]//2, size[1], 3), dtype=np.uint8)
    
    # Mirror for symmetry
    full_pixels = np.concatenate([pixels, pixels[::-1]], axis=0)
    
    img = Image.fromarray(full_pixels)
    return img.filter(ImageFilter.SMOOTH)


def generate_low_frequency_noise(size=(512, 512)):
    """Generate low-frequency noise pattern (AI characteristic)."""
    # AI images often have smoother frequency spectrum
    noise = np.random.randn(size[0], size[1], 3) * 30 + 128
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(noise)
    # Heavy blur = low frequency
    return img.filter(ImageFilter.GaussianBlur(radius=10))


def generate_unrealistic_colors(size=(512, 512)):
    """Generate images with unrealistic color combinations."""
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    
    # Create bands of unusual colors
    band_height = size[1] // 5
    colors = [
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 255, 0),   # Yellow
        (128, 0, 255),   # Purple
        (255, 128, 0),   # Orange
    ]
    
    for i, color in enumerate(colors):
        y = i * band_height
        draw.rectangle([(0, y), (size[0], y + band_height)], fill=color)
    
    return img.filter(ImageFilter.SMOOTH_MORE)


def main():
    """Generate AI reference samples."""
    print("=" * 60)
    print("VeriFile-X: AI Reference Sample Generation")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generators = [
        generate_smooth_gradient,
        generate_symmetric_pattern,
        generate_low_frequency_noise,
        generate_unrealistic_colors,
    ]
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generating AI samples"):
        # Use different generators
        generator = generators[i % len(generators)]
        img = generator()
        
        filepath = OUTPUT_DIR / f"ai_sample_{i:04d}.png"
        img.save(filepath)
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {NUM_SAMPLES} AI samples")
    print(f"📁 Location: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n⚠️  Note: For production, replace these with real AI-generated images")
    print("   - Use Stable Diffusion API")
    print("   - Download Midjourney samples")
    print("   - Use DALL-E API")


if __name__ == "__main__":
    main()
