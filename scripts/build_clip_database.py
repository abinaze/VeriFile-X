"""
Build CLIP embedding database from reference images.

Computes CLIP embeddings for all real and AI images,
then calculates centroids to use in clip_detector.py
"""
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pickle


def load_clip_model():
    """Load CLIP model."""
    print("📦 Loading CLIP ViT-B/32 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Model loaded on {device}")
    return model, preprocess, device


def compute_embeddings(image_dir, model, preprocess, device):
    """Compute CLIP embeddings for all images in directory."""
    embeddings = []
    image_files = list(Path(image_dir).glob("*.jpg")) + \
                  list(Path(image_dir).glob("*.png"))
    
    print(f"📸 Processing {len(image_files)} images from {image_dir}")
    
    for img_path in tqdm(image_files, desc="Computing embeddings"):
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Compute embedding
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            embeddings.append(embedding.cpu().numpy())
        
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    return np.vstack(embeddings) if embeddings else np.array([])


def main():
    """Build CLIP reference database."""
    print("=" * 70)
    print("VeriFile-X: CLIP Reference Database Builder")
    print("=" * 70)
    
    # Load model
    model, preprocess, device = load_clip_model()
    
    # Compute embeddings for real images
    print("\n🌍 Computing embeddings for REAL images...")
    real_embeddings = compute_embeddings(
        "data/reference/real",
        model, preprocess, device
    )
    
    # Compute embeddings for AI images
    print("\n🤖 Computing embeddings for AI images...")
    ai_embeddings = compute_embeddings(
        "data/reference/ai",
        model, preprocess, device
    )
    
    # Compute centroids
    print("\n📊 Computing centroids...")
    real_centroid = real_embeddings.mean(axis=0)
    ai_centroid = ai_embeddings.mean(axis=0)
    
    # Normalize centroids
    real_centroid = real_centroid / np.linalg.norm(real_centroid)
    ai_centroid = ai_centroid / np.linalg.norm(ai_centroid)
    
    # Compute separation (cosine distance)
    separation = 1 - np.dot(real_centroid, ai_centroid)
    
    # Save database
    database = {
        'real_centroid': real_centroid,
        'ai_centroid': ai_centroid,
        'real_count': len(real_embeddings),
        'ai_count': len(ai_embeddings),
        'separation': float(separation),
        'embedding_dim': len(real_centroid),
    }
    
    output_path = Path("data/reference/clip_database.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("CLIP Database Built Successfully!")
    print("=" * 70)
    print(f"📊 Statistics:")
    print(f"   Real images: {database['real_count']}")
    print(f"   AI images: {database['ai_count']}")
    print(f"   Embedding dimension: {database['embedding_dim']}")
    print(f"   Centroid separation: {database['separation']:.4f}")
    print(f"   (Higher is better, >0.1 is good)")
    print(f"\n💾 Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
