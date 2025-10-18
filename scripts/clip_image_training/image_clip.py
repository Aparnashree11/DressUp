"""
GCP-Optimized Fashion-CLIP Training Script
Designed for L4 GPU with automatic cloud storage backup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from datetime import datetime
import subprocess
import time
import os


bucket_name = os.environ.get("GCS_BUCKET")
# ============================================================================
# Dataset Class
# ============================================================================

class DeepFashionMultiModalDataset(Dataset):
    """Dataset optimized for DeepFashion-MultiModal with GCP"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 use_parsing: bool = True,
                 use_densepose: bool = True,
                 augment_text: bool = True):
        
        self.data_root = Path(data_root)
        self.split = split
        self.use_parsing = use_parsing
        self.use_densepose = use_densepose
        self.augment_text = augment_text
        
        print(f"[Dataset] Initializing {split} dataset from: {self.data_root}")
        
        # Load captions
        captions_path = self.data_root / "captions.json"
        with open(captions_path, 'r', encoding='utf-8') as f:
            self.descriptions = json.load(f)
        print(f"[Dataset] Loaded {len(self.descriptions)} captions")
        
        # Load shape labels
        self.shape_labels = {}
        shape_file = self.data_root / "labels" / "shape_anno_all.txt"
        if shape_file.exists():
            with open(shape_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        img_name = parts[0].replace('.jpg', '')
                        self.shape_labels[img_name] = [int(x) for x in parts[1:]]
        print(f"[Dataset] Loaded {len(self.shape_labels)} shape labels")
        
        # Get image files
        images_dir = self.data_root / "images"
        self.image_files = list(images_dir.glob("*.jpg"))
        self.image_files.sort()
        
        # Train/val split (80/20)
        random.seed(42)
        random.shuffle(self.image_files)
        split_idx = int(0.8 * len(self.image_files))
        
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
        
        # Transforms
        self.transform = self.get_transforms(split == 'train')
        
        print(f"[Dataset] Loaded {len(self.image_files)} images for {split}")
        
        # Check data availability
        self.check_data_availability()
    
    def check_data_availability(self):
        """Check what data is available"""
        sample_imgs = self.image_files[:5]
        segm_count = sum(1 for img in sample_imgs if (self.data_root / "segm" / f"{img.stem}_segm.png").exists())
        dense_count = sum(1 for img in sample_imgs if (self.data_root / "densepose" / f"{img.stem}_densepose.png").exists())
        
        print(f"[Dataset] Segmentation: {segm_count}/5 samples available")
        print(f"[Dataset] DensePose: {dense_count}/5 samples available")
    
    def get_transforms(self, is_training: bool):
        """Get image transforms"""
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def load_densepose(self, img_name: str) -> torch.Tensor:
        """Load DensePose data"""
        if not self.use_densepose:
            return torch.zeros(3, 224, 224)
        
        dense_path = self.data_root / "densepose" / f"{img_name}_densepose.png"
        if dense_path.exists():
            try:
                densepose = Image.open(dense_path).convert('RGB')
                densepose = self.transform(densepose)
                return densepose
            except:
                pass
        return torch.zeros(3, 224, 224)
    
    def find_caption(self, img_name: str) -> str:
        """Find caption with flexible matching"""
        # Try exact match
        if img_name in self.descriptions:
            return self.descriptions[img_name]
        
        # Try with extensions
        for ext in ['.jpg', '.png']:
            if f"{img_name}{ext}" in self.descriptions:
                return self.descriptions[f"{img_name}{ext}"]
        
        # Try partial match
        for key in self.descriptions.keys():
            if key.replace('.jpg', '').replace('.png', '') == img_name:
                return self.descriptions[key]
        
        return "A fashion image showing clothing"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        # Load caption
        caption = self.find_caption(img_name)
        if not isinstance(caption, str):
            caption = str(caption)
        
        # Load DensePose
        densepose = self.load_densepose(img_name)
        
        # Load shape attributes
        attributes = {}
        attr_key = img_name
        if attr_key not in self.shape_labels:
            attr_key = f"{img_name}.jpg"
        if attr_key in self.shape_labels:
            attributes['shape'] = self.shape_labels[attr_key]
        
        return {
            'image': image,
            'text': caption,
            'img_name': img_name,
            'densepose': densepose,
            'attributes': attributes
        }

# ============================================================================
# Model Architecture
# ============================================================================

class EnhancedFashionCLIPModel(nn.Module):
    """Fashion-CLIP with DensePose integration"""
    
    def __init__(self, 
                 base_model_name: str = "openai/clip-vit-base-patch32",
                 embed_dim: int = 512,
                 use_densepose: bool = True):
        
        super().__init__()
        
        print(f"[Model] Loading base CLIP model: {base_model_name}")
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
        
        # Shape classifier (3 main attributes)
        self.shape_classifier = nn.Linear(embed_dim, 3 * 6)  # 3 attributes, max 6 classes each
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        self.init_weights()
        print(f"[Model] Model initialized with embed_dim={embed_dim}")
    
    def init_weights(self):
        """Initialize custom layers"""
        nn.init.xavier_uniform_(self.fashion_vision_proj.weight)
        nn.init.zeros_(self.fashion_vision_proj.bias)
        nn.init.xavier_uniform_(self.fashion_text_proj.weight)
        nn.init.zeros_(self.fashion_text_proj.bias)
    
    def encode_image(self, images, densepose=None):
        """Encode images with optional DensePose"""
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
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text"""
        text_outputs = self.base_clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = text_outputs.pooler_output
        
        fashion_text_embed = self.fashion_text_proj(pooled_output)
        fashion_text_embed = F.normalize(fashion_text_embed, dim=-1)
        
        return fashion_text_embed
    
    def forward(self, images, input_ids, attention_mask, densepose=None):
        """Forward pass"""
        vision_embed = self.encode_image(images, densepose)
        text_embed = self.encode_text(input_ids, attention_mask)
        
        # Shape predictions
        shape_preds = self.shape_classifier(vision_embed)
        
        return {
            'vision_embeds': vision_embed,
            'text_embeds': text_embed,
            'shape_preds': shape_preds,
            'temperature': self.temperature
        }

# ============================================================================
# Custom Collate Function
# ============================================================================

def custom_collate_fn(batch):
    """Custom collate to handle missing data"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'densepose': torch.stack([item['densepose'] for item in batch]),
        'text': [item['text'] for item in batch],
        'img_name': [item['img_name'] for item in batch],
        'attributes': [item['attributes'] for item in batch]
    }

# ============================================================================
# Trainer Class
# ============================================================================

class FashionCLIPTrainer:
    """Trainer with GCP integration"""
    
    def __init__(self, 
                 model: EnhancedFashionCLIPModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: str,
                 gcs_bucket: str = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gcs_bucket = gcs_bucket
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.best_val_loss = float('inf')
        
        print(f"[Trainer] Initialized on device: {device}")
        if gcs_bucket:
            print(f"[Trainer] Will backup to GCS bucket: {gcs_bucket}")
    
    def compute_contrastive_loss(self, vision_embeds, text_embeds, temperature):
        """CLIP-style contrastive loss"""
        logits = torch.matmul(vision_embeds, text_embeds.t()) / temperature
        batch_size = vision_embeds.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def compute_shape_loss(self, shape_preds, batch):
        """Shape attribute loss"""
        losses = []
        
        # Split predictions for 3 attributes
        preds_split = shape_preds.view(-1, 3, 6)  # [batch, 3_attrs, 6_classes]
        
        for attr_idx in range(3):
            targets = []
            for sample in batch['attributes']:
                if 'shape' in sample and len(sample['shape']) > attr_idx:
                    targets.append(sample['shape'][attr_idx])
                else:
                    targets.append(-1)
            
            targets = torch.tensor(targets, device=self.device)
            valid_mask = (targets >= 0) & (targets < 6)
            
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    preds_split[valid_mask, attr_idx, :],
                    targets[valid_mask]
                )
                losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        images = batch['image'].to(self.device)
        densepose = batch['densepose'].to(self.device)
        
        tokenized = self.tokenizer(
            batch['text'],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        outputs = self.model(images, input_ids, attention_mask, densepose)
        
        # Compute losses
        contrastive_loss = self.compute_contrastive_loss(
            outputs['vision_embeds'],
            outputs['text_embeds'],
            outputs['temperature']
        )
        
        shape_loss = self.compute_shape_loss(outputs['shape_preds'], batch)
        
        total_loss = contrastive_loss + 0.3 * shape_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'shape_loss': shape_loss.item()
        }
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                
                images = batch['image'].to(self.device)
                densepose = batch['densepose'].to(self.device)
                
                tokenized = self.tokenizer(
                    batch['text'],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors='pt'
                )
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask, densepose)
                
                contrastive_loss = self.compute_contrastive_loss(
                    outputs['vision_embeds'],
                    outputs['text_embeds'],
                    outputs['temperature']
                )
                
                shape_loss = self.compute_shape_loss(outputs['shape_preds'], batch)
                batch_loss = contrastive_loss + 0.3 * shape_loss
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {
                'embed_dim': 512,
                'use_densepose': True
            }
        }
        
        # Save locally
        local_path = f'/tmp/fashion_clip_epoch{epoch}_{timestamp}.pth'
        torch.save(checkpoint, local_path)
        print(f"[Checkpoint] Saved to {local_path}")
        
        # Upload to GCS if bucket specified
        if self.gcs_bucket:
            gcs_path = f"gs://{self.gcs_bucket}/models/fashion_clip_epoch{epoch}_{timestamp}.pth"
            try:
                subprocess.run(f"gsutil cp {local_path} {gcs_path}", shell=True, check=True)
                print(f"[GCS] Uploaded to {gcs_path}")
            except:
                print(f"[GCS] Upload failed, model saved locally only")
        
        if is_best:
            best_path = '/tmp/fashion_clip_best.pth'
            torch.save(checkpoint, best_path)
            if self.gcs_bucket:
                try:
                    subprocess.run(
                        f"gsutil cp {best_path} gs://{self.gcs_bucket}/models/fashion_clip_best.pth",
                        shell=True, check=True
                    )
                    print(f"[GCS] Uploaded best model")
                except:
                    pass
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            # Training
            self.model.train()
            train_metrics = {
                'total_loss': 0,
                'contrastive_loss': 0,
                'shape_loss': 0
            }
            
            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                
                losses = self.train_step(batch)
                
                for key in train_metrics:
                    train_metrics[key] += losses[key]
                
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': f"{losses['total_loss']:.4f}",
                        'temp': f"{self.model.temperature.item():.4f}"
                    })
            
            # Average training metrics
            num_batches = len(self.train_loader)
            for key in train_metrics:
                train_metrics[key] /= num_batches
            
            # Validation
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Print metrics
            print(f"\n[Metrics]")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Contrastive: {train_metrics['contrastive_loss']:.4f}")
            print(f"  Shape: {train_metrics['shape_loss']:.4f}")
            print(f"  Temperature: {self.model.temperature.item():.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✅ New best model!")
            
            if (epoch + 1) % 2 == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*60)
    print("Fashion-CLIP Training on GCP")
    print("="*60 + "\n")
    
    # Configuration
    DATA_ROOT = os.environ.get("DATASET_PATH")  # Update this path
    GCS_BUCKET = bucket_name  # Update with your GCS bucket name
    BATCH_SIZE = 32  # Optimized for L4 GPU with 24GB
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 8
    NUM_WORKERS = 4
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Setup] Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Datasets
    print("\n[Setup] Creating datasets...")
    train_dataset = DeepFashionMultiModalDataset(
        data_root=DATA_ROOT,
        split='train',
        use_parsing=True,
        use_densepose=True,
        augment_text=True
    )
    
    val_dataset = DeepFashionMultiModalDataset(
        data_root=DATA_ROOT,
        split='val',
        use_parsing=True,
        use_densepose=True,
        augment_text=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"[Setup] Train batches: {len(train_loader)}")
    print(f"[Setup] Val batches: {len(val_loader)}")
    
    # Model
    print("\n[Setup] Creating model...")
    model = EnhancedFashionCLIPModel(
        base_model_name="openai/clip-vit-base-patch32",
        embed_dim=512,
        use_densepose=True
    )
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    
    # Trainer
    trainer = FashionCLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gcs_bucket=GCS_BUCKET
    )
    
    # Train
    try:
        trainer.train(num_epochs=NUM_EPOCHS)
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(0, 0, is_best=False)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
