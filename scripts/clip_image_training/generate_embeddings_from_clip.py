"""
Extract Fashion-CLIP embeddings on GCP VM and save to GCS.

Run this on your GCP VM where the model and dataset are located.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
from datetime import datetime
import pickle


class EnhancedFashionCLIPModel(nn.Module):
    """Fashion-CLIP model architecture (matches training)"""
    
    def __init__(self, 
                 base_model_name: str = "openai/clip-vit-base-patch32",
                 embed_dim: int = 512,
                 use_densepose: bool = True):
        
        super().__init__()
        
        self.base_clip = CLIPModel.from_pretrained(base_model_name)
        
        self.vision_embed_dim = self.base_clip.vision_model.config.hidden_size
        self.text_embed_dim = self.base_clip.text_model.config.hidden_size
        self.use_densepose = use_densepose
        
        # Fashion projections
        self.fashion_vision_proj = nn.Linear(self.vision_embed_dim, embed_dim)
        self.fashion_text_proj = nn.Linear(self.text_embed_dim, embed_dim)
        
        # DensePose encoder
        if use_densepose:
            self.densepose_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, embed_dim)
            )
            
            self.multimodal_fusion = nn.MultiheadAttention(
                embed_dim, num_heads=8, dropout=0.1, batch_first=True
            )
        
        self.shape_classifier = nn.Linear(embed_dim, 3 * 6)
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def encode_image(self, images, densepose=None):
        """Encode images to fashion embeddings."""
        vision_outputs = self.base_clip.vision_model(images)
        pooled_output = vision_outputs.pooler_output
        
        fashion_vision_embed = self.fashion_vision_proj(pooled_output)
        fashion_vision_embed = F.normalize(fashion_vision_embed, dim=-1)
        
        if self.use_densepose and densepose is not None and densepose.abs().sum() > 0:
            densepose_embed = self.densepose_encoder(densepose)
            densepose_embed = F.normalize(densepose_embed, dim=-1)
            
            vision_seq = fashion_vision_embed.unsqueeze(1)
            densepose_seq = densepose_embed.unsqueeze(1)
            
            fused_features, _ = self.multimodal_fusion(vision_seq, densepose_seq, densepose_seq)
            fashion_vision_embed = fused_features.squeeze(1)
        
        return fashion_vision_embed


class EmbeddingExtractor:
    """Extract embeddings and save to GCS."""
    
    def __init__(self, 
                 model_path: str,
                 dataset_path: str,
                 output_dir: str,
                 gcs_bucket: str,
                 device: str = None,
                 batch_size: int = 32):
        
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.gcs_bucket = gcs_bucket
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print(f"[Setup] Device: {self.device}")
        print(f"[Setup] Batch size: {batch_size}")
        print(f"[Setup] Output dir: {self.output_dir}")
        print(f"[Setup] GCS bucket: gs://{gcs_bucket}")
    
    def load_model(self) -> EnhancedFashionCLIPModel:
        """Load trained Fashion-CLIP model."""
        print(f"\n[Model] Loading from {self.model_path}")
        
        model = EnhancedFashionCLIPModel()
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[Model] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print(f"[Model] ✅ Loaded successfully on {self.device}")
        return model
    
    def get_image_files(self):
        """Get all image files from dataset."""
        images_dir = self.dataset_path / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files.sort()
        
        print(f"\n[Dataset] Found {len(image_files)} images")
        return image_files
    
    def extract_embeddings(self, model: EnhancedFashionCLIPModel, image_files: list):
        """Extract embeddings for all images."""
        print(f"\n[Extraction] Processing {len(image_files)} images...")
        
        embeddings = {}
        failed_images = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_files), self.batch_size)):
            batch_files = image_files[i:i + self.batch_size]
            batch_images = []
            batch_names = []
            
            # Load batch
            for img_path in batch_files:
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                    batch_names.append(img_path.stem)
                except Exception as e:
                    print(f"[Warning] Failed to load {img_path.name}: {e}")
                    failed_images.append(img_path.name)
                    continue
            
            if not batch_images:
                continue
            
            # Process batch
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = model.encode_image(pixel_values)
                
                # Store embeddings
                for name, embedding in zip(batch_names, batch_embeddings):
                    embeddings[name] = embedding.cpu().numpy()
            
            except Exception as e:
                print(f"[Error] Batch processing failed: {e}")
                continue
        
        print(f"\n[Extraction] Successfully extracted {len(embeddings)} embeddings")
        if failed_images:
            print(f"[Extraction] Failed: {len(failed_images)} images")
        
        return embeddings, failed_images
    
    def save_embeddings(self, embeddings: dict, failed_images: list):
        """Save embeddings locally and to GCS."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy arrays (efficient format)
        embeddings_file = self.output_dir / f"fashion_clip_embeddings_{timestamp}.npz"
        
        print(f"\n[Saving] Saving {len(embeddings)} embeddings...")
        
        # Convert to numpy format
        item_ids = list(embeddings.keys())
        embedding_matrix = np.stack([embeddings[k] for k in item_ids])
        
        # Save locally
        np.savez_compressed(
            embeddings_file,
            item_ids=item_ids,
            embeddings=embedding_matrix,
            embedding_dim=embedding_matrix.shape[1],
            num_items=len(item_ids)
        )
        
        print(f"[Saving] ✅ Saved locally: {embeddings_file}")
        print(f"[Saving] File size: {embeddings_file.stat().st_size / (1024**2):.2f} MB")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_items': len(item_ids),
            'embedding_dim': embedding_matrix.shape[1],
            'failed_images': failed_images,
            'model_path': str(self.model_path),
            'dataset_path': str(self.dataset_path)
        }
        
        metadata_file = self.output_dir / f"embeddings_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Saving] ✅ Saved metadata: {metadata_file}")
        
        # Upload to GCS
        if self.gcs_bucket:
            self.upload_to_gcs(embeddings_file)
            self.upload_to_gcs(metadata_file)
        
        return embeddings_file, metadata_file
    
    def upload_to_gcs(self, local_file: Path):
        """Upload file to Google Cloud Storage."""
        gcs_path = f"gs://{self.gcs_bucket}/embeddings/{local_file.name}"
        
        print(f"[GCS] Uploading to {gcs_path}...")
        
        try:
            cmd = f"gsutil cp {local_file} {gcs_path}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"[GCS] ✅ Uploaded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[GCS] ❌ Upload failed: {e.stderr}")
            print(f"[GCS] You can manually upload later:")
            print(f"      gsutil cp {local_file} {gcs_path}")
            return False
    
    def generate_item_mapping(self, image_files: list):
        """Generate item ID to filename mapping."""
        item_mapping = {}
        
        for idx, img_path in enumerate(image_files):
            item_mapping[f"item_{idx:06d}"] = {
                'filename': img_path.name,
                'stem': img_path.stem,
                'path': str(img_path)
            }
        
        mapping_file = self.output_dir / "item_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(item_mapping, f, indent=2)
        
        print(f"[Mapping] ✅ Saved item mapping: {mapping_file}")
        
        if self.gcs_bucket:
            self.upload_to_gcs(mapping_file)
        
        return mapping_file


def main():
    """Main execution."""
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE
    # ========================================================================
    
    MODEL_PATH = "/tmp/fashion_clip_best.pth"  # Your trained model
    DATASET_PATH = "dataset"   # DeepFashion dataset location
    OUTPUT_DIR = "/tmp/embeddings"              # Temporary output directory
    GCS_BUCKET = "fashion-dataset-as-113"       # Your GCS bucket
    BATCH_SIZE = 32                             # Adjust based on GPU memory
    
    # ========================================================================
    
    print("="*70)
    print("  FASHION-CLIP EMBEDDING EXTRACTION (GCP)")
    print("="*70)
    
    try:
        # Initialize extractor
        extractor = EmbeddingExtractor(
            model_path=MODEL_PATH,
            dataset_path=DATASET_PATH,
            output_dir=OUTPUT_DIR,
            gcs_bucket=GCS_BUCKET,
            batch_size=BATCH_SIZE
        )
        
        # Load model
        model = extractor.load_model()
        
        # Get image files
        image_files = extractor.get_image_files()
        
        # Generate item mapping
        extractor.generate_item_mapping(image_files)
        
        # Extract embeddings
        embeddings, failed_images = extractor.extract_embeddings(model, image_files)
        
        # Save embeddings
        embeddings_file, metadata_file = extractor.save_embeddings(embeddings, failed_images)
        
        # Summary
        print("\n" + "="*70)
        print("  EXTRACTION COMPLETE")
        print("="*70)
        print(f"\n✅ Extracted embeddings: {len(embeddings)}")
        print(f"✅ Embedding dimension: 512")
        print(f"✅ Saved to: {embeddings_file}")
        print(f"✅ Metadata: {metadata_file}")
        
        print("\n Files in GCS:")
        print(f"   gs://{GCS_BUCKET}/embeddings/{embeddings_file.name}")
        print(f"   gs://{GCS_BUCKET}/embeddings/{metadata_file.name}")
        
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
