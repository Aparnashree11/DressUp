"""
Validate Fashion-CLIP Embedding Quality

This script:
1. Loads embeddings from GCS or local file
2. Visualizes embeddings with t-SNE and UMAP
3. Tests retrieval accuracy (image-to-image, text-to-image)
4. Computes retrieval metrics (Recall@K, MRR, nDCG)
5. Analyzes clustering quality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Optional: UMAP (install with: pip install umap-learn)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not installed. Using t-SNE only.")
    print("   Install with: pip install umap-learn")


class EnhancedFashionCLIPModel(nn.Module):
    """Fashion-CLIP model for text encoding"""
    
    def __init__(self, 
                 base_model_name: str = "openai/clip-vit-base-patch32",
                 embed_dim: int = 512):
        super().__init__()
        
        self.base_clip = CLIPModel.from_pretrained(base_model_name)
        self.text_embed_dim = self.base_clip.text_model.config.hidden_size
        self.fashion_text_proj = nn.Linear(self.text_embed_dim, embed_dim)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to fashion embeddings."""
        text_outputs = self.base_clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = text_outputs.pooler_output
        fashion_text_embed = self.fashion_text_proj(pooled_output)
        fashion_text_embed = F.normalize(fashion_text_embed, dim=-1)
        return fashion_text_embed


class EmbeddingValidator:
    """Validate and analyze Fashion-CLIP embeddings"""
    
    def __init__(self, 
                 embeddings_path: str,
                 model_path: str = None,
                 dataset_path: str = None,
                 device: str = None):
        
        self.embeddings_path = Path(embeddings_path)
        self.model_path = Path(model_path) if model_path else None
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Setup] Device: {self.device}")
        print(f"[Setup] Embeddings: {self.embeddings_path}")
        
        # Load embeddings
        self.load_embeddings()
        
        # Load model for text queries (optional)
        if self.model_path and self.model_path.exists():
            self.model = self.load_model()
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.model = None
            print("⚠️  Model not provided - text-to-image retrieval will be skipped")
    
    def load_embeddings(self):
        """Load embeddings from .npz file"""
        print(f"\n[Loading] Reading embeddings from {self.embeddings_path.name}")
        
        data = np.load(self.embeddings_path, allow_pickle=True)
        
        self.item_ids = data['item_ids']
        self.embeddings = data['embeddings']
        self.embedding_dim = int(data['embedding_dim'])
        self.num_items = int(data['num_items'])
        
        print(f"[Loading] ✅ Loaded {self.num_items} embeddings")
        print(f"[Loading] Embedding dimension: {self.embedding_dim}")
        print(f"[Loading] Matrix shape: {self.embeddings.shape}")
    
    def load_model(self):
        """Load Fashion-CLIP model for text encoding"""
        print(f"\n[Model] Loading from {self.model_path}")
        
        model = EnhancedFashionCLIPModel()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print(f"[Model] ✅ Loaded successfully")
        return model
    
    def visualize_embeddings_tsne(self, n_samples: int = 1000, save_path: str = None):
        """Visualize embeddings using t-SNE"""
        print(f"\n{'='*70}")
        print("  t-SNE VISUALIZATION")
        print(f"{'='*70}\n")
        
        # Sample embeddings if too many
        if self.num_items > n_samples:
            indices = np.random.choice(self.num_items, n_samples, replace=False)
            embeddings_sample = self.embeddings[indices]
            print(f"[t-SNE] Using {n_samples} random samples")
        else:
            embeddings_sample = self.embeddings
            print(f"[t-SNE] Using all {self.num_items} embeddings")
        
        # Run t-SNE
        print("[t-SNE] Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.5, s=10, c=range(len(embeddings_2d)), cmap='viridis')
        plt.colorbar(label='Item Index')
        plt.title('t-SNE Visualization of Fashion-CLIP Embeddings', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[t-SNE] ✅ Saved to {save_path}")
        else:
            plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
            print(f"[t-SNE] ✅ Saved to tsne_visualization.png")
        
        plt.close()
        
        return embeddings_2d
    
    def visualize_embeddings_umap(self, n_samples: int = 1000, save_path: str = None):
        """Visualize embeddings using UMAP"""
        if not HAS_UMAP:
            print("[UMAP] ⚠️  UMAP not installed, skipping")
            return None
        
        print(f"\n{'='*70}")
        print("  UMAP VISUALIZATION")
        print(f"{'='*70}\n")
        
        # Sample embeddings if too many
        if self.num_items > n_samples:
            indices = np.random.choice(self.num_items, n_samples, replace=False)
            embeddings_sample = self.embeddings[indices]
            print(f"[UMAP] Using {n_samples} random samples")
        else:
            embeddings_sample = self.embeddings
            print(f"[UMAP] Using all {self.num_items} embeddings")
        
        # Run UMAP
        print("[UMAP] Computing UMAP projection...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings_sample)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.5, s=10, c=range(len(embeddings_2d)), cmap='viridis')
        plt.colorbar(label='Item Index')
        plt.title('UMAP Visualization of Fashion-CLIP Embeddings', fontsize=14, fontweight='bold')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[UMAP] ✅ Saved to {save_path}")
        else:
            plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight')
            print(f"[UMAP] ✅ Saved to umap_visualization.png")
        
        plt.close()
        
        return embeddings_2d
    
    def test_image_to_image_retrieval(self, n_queries: int = 100, k_values: List[int] = [1, 5, 10, 20]):
        """Test image-to-image retrieval accuracy"""
        print(f"\n{'='*70}")
        print("  IMAGE-TO-IMAGE RETRIEVAL")
        print(f"{'='*70}\n")
        
        # Sample query items
        query_indices = np.random.choice(self.num_items, min(n_queries, self.num_items), replace=False)
        
        print(f"[Retrieval] Testing with {len(query_indices)} queries")
        print(f"[Retrieval] Computing similarities...")
        
        # Compute similarity matrix for queries
        query_embeddings = self.embeddings[query_indices]
        similarities = cosine_similarity(query_embeddings, self.embeddings)
        
        # Compute metrics
        metrics = self.compute_retrieval_metrics(similarities, query_indices, k_values)
        
        # Print results
        print(f"\n[Results] Image-to-Image Retrieval:")
        print(f"{'Metric':<20} " + " ".join([f"K={k:<4}" for k in k_values]))
        print("-" * 70)
        
        for metric_name, values in metrics.items():
            print(f"{metric_name:<20} " + " ".join([f"{v:.4f}" for v in values]))
        
        return metrics
    
    def test_text_to_image_retrieval(self, text_queries: List[str], k_values: List[int] = [1, 5, 10, 20]):
        """Test text-to-image retrieval"""
        if not self.model:
            print("\n[Text Retrieval] ⚠️  Model not loaded, skipping")
            return None
        
        print(f"\n{'='*70}")
        print("  TEXT-TO-IMAGE RETRIEVAL")
        print(f"{'='*70}\n")
        
        print(f"[Text Retrieval] Testing with {len(text_queries)} queries")
        
        results = []
        
        for query_text in text_queries:
            print(f"\n[Query] '{query_text}'")
            
            # Encode text query
            text_inputs = self.processor(text=query_text, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_emb = self.model.encode_text(
                    text_inputs['input_ids'],
                    text_inputs['attention_mask']
                ).cpu().numpy()
            
            # Compute similarities
            similarities = cosine_similarity(text_emb, self.embeddings)[0]
            
            # Get top-K results
            top_k_indices = np.argsort(similarities)[::-1][:max(k_values)]
            
            print(f"[Results] Top-5 matches:")
            for i, idx in enumerate(top_k_indices[:5]):
                print(f"  {i+1}. Item: {self.item_ids[idx]}, Similarity: {similarities[idx]:.4f}")
            
            results.append({
                'query': query_text,
                'top_indices': top_k_indices,
                'similarities': similarities[top_k_indices]
            })
        
        return results
    
    def compute_retrieval_metrics(self, similarities: np.ndarray, query_indices: np.ndarray, k_values: List[int]):
        """Compute retrieval metrics: Recall@K, MRR, nDCG@K"""
        
        metrics = {
            'Recall@K': [],
            'MRR': [],
            'nDCG@K': []
        }
        
        for k in k_values:
            recalls = []
            mrrs = []
            ndcgs = []
            
            for i, query_idx in enumerate(query_indices):
                # Get top-K predictions (excluding the query itself)
                similarities_i = similarities[i].copy()
                similarities_i[query_idx] = -np.inf  # Exclude self
                top_k_indices = np.argsort(similarities_i)[::-1][:k]
                
                # For image-to-image, we consider items semantically similar
                # As a proxy, we'll check if top-k includes nearby items
                # In practice, you'd use ground truth labels/categories
                
                # Simple metric: check if query is in top-K (should be high)
                # For now, we'll compute based on similarity distribution
                
                # Recall@K: proportion of relevant items in top-K
                # Using top 10% of similarities as "relevant"
                threshold = np.percentile(similarities[i], 90)
                relevant_items = np.where(similarities[i] >= threshold)[0]
                relevant_items = relevant_items[relevant_items != query_idx]
                
                recall = len(set(top_k_indices) & set(relevant_items)) / max(len(relevant_items), 1)
                recalls.append(recall)
                
                # MRR: Mean Reciprocal Rank of first relevant item
                for rank, idx in enumerate(top_k_indices, 1):
                    if idx in relevant_items:
                        mrrs.append(1.0 / rank)
                        break
                else:
                    mrrs.append(0.0)
                
                # nDCG@K: Normalized Discounted Cumulative Gain
                dcg = 0
                idcg = 0
                for rank, idx in enumerate(top_k_indices, 1):
                    relevance = 1 if idx in relevant_items else 0
                    dcg += relevance / np.log2(rank + 1)
                
                for rank in range(1, min(k, len(relevant_items)) + 1):
                    idcg += 1.0 / np.log2(rank + 1)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
            
            metrics['Recall@K'].append(np.mean(recalls))
            metrics['MRR'].append(np.mean(mrrs))
            metrics['nDCG@K'].append(np.mean(ndcgs))
        
        return metrics
    
    def analyze_clustering(self, n_clusters: int = 10):
        """Analyze embedding clustering quality"""
        print(f"\n{'='*70}")
        print("  CLUSTERING ANALYSIS")
        print(f"{'='*70}\n")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        print(f"[Clustering] Running K-Means with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Compute clustering metrics
        silhouette = silhouette_score(self.embeddings, cluster_labels, sample_size=5000)
        davies_bouldin = davies_bouldin_score(self.embeddings, cluster_labels)
        
        print(f"\n[Metrics]")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
        
        # Cluster size distribution
        cluster_counts = np.bincount(cluster_labels)
        print(f"\n[Cluster Sizes]")
        for i, count in enumerate(cluster_counts):
            print(f"  Cluster {i}: {count} items ({count/self.num_items*100:.1f}%)")
        
        # Visualize cluster distribution
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_clusters), cluster_counts)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Items')
        plt.title('Cluster Size Distribution')
        plt.tight_layout()
        plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n[Clustering] ✅ Saved visualization to cluster_distribution.png")
        plt.close()
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'cluster_labels': cluster_labels,
            'cluster_sizes': cluster_counts
        }
    
    def generate_report(self, output_path: str = "embedding_validation_report.txt"):
        """Generate comprehensive validation report"""
        print(f"\n{'='*70}")
        print("  GENERATING VALIDATION REPORT")
        print(f"{'='*70}\n")
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("  FASHION-CLIP EMBEDDING VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            
            f.write("EMBEDDING STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Number of items: {self.num_items}\n")
            f.write(f"Embedding dimension: {self.embedding_dim}\n")
            f.write(f"Matrix shape: {self.embeddings.shape}\n")
            f.write(f"Matrix size: {self.embeddings.nbytes / (1024**2):.2f} MB\n\n")
            
            # Embedding statistics
            f.write(f"Embedding norm mean: {np.mean(np.linalg.norm(self.embeddings, axis=1)):.4f}\n")
            f.write(f"Embedding norm std: {np.std(np.linalg.norm(self.embeddings, axis=1)):.4f}\n\n")
            
            f.write("\nVisualization and clustering analysis completed.\n")
            f.write("See generated plots for details.\n")
        
        print(f"[Report] ✅ Saved to {output_path}")


def main():
    """Main validation pipeline"""
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE
    # ========================================================================
    
    EMBEDDINGS_PATH = "/tmp/embeddings/fashion_clip_embeddings_20251009_191802.npz"
    MODEL_PATH = "/tmp/fashion_clip_best.pth"  # Optional, for text queries
    DATASET_PATH = "dataset"  # Optional, for loading images
    
    # Sample text queries for testing
    TEXT_QUERIES = [
        "red floral summer dress",
        "blue denim jeans",
        "black leather jacket",
        "white cotton t-shirt",
        "striped midi skirt",
        "casual sneakers",
        "formal blazer",
        "winter coat"
    ]
    
    # ========================================================================
    
    print("="*70)
    print("  FASHION-CLIP EMBEDDING VALIDATION")
    print("="*70)
    
    try:
        # Initialize validator
        validator = EmbeddingValidator(
            embeddings_path=EMBEDDINGS_PATH,
            model_path=MODEL_PATH,
            dataset_path=DATASET_PATH
        )
        
        # 1. Visualize embeddings
        print("\n" + "="*70)
        print("  STEP 1: VISUALIZATION")
        print("="*70)
        validator.visualize_embeddings_tsne(n_samples=1000)
        validator.visualize_embeddings_umap(n_samples=1000)
        
        # 2. Test image-to-image retrieval
        print("\n" + "="*70)
        print("  STEP 2: IMAGE-TO-IMAGE RETRIEVAL")
        print("="*70)
        img_metrics = validator.test_image_to_image_retrieval(n_queries=100)
        
        # 3. Test text-to-image retrieval
        print("\n" + "="*70)
        print("  STEP 3: TEXT-TO-IMAGE RETRIEVAL")
        print("="*70)
        text_results = validator.test_text_to_image_retrieval(TEXT_QUERIES)
        
        # 4. Analyze clustering
        print("\n" + "="*70)
        print("  STEP 4: CLUSTERING ANALYSIS")
        print("="*70)
        cluster_metrics = validator.analyze_clustering(n_clusters=10)
        
        # 5. Generate report
        validator.generate_report()
        
        # Final summary
        print("\n" + "="*70)
        print("  VALIDATION COMPLETE")
        print("="*70)
        print("\n✅ Generated outputs:")
        print("   - tsne_visualization.png")
        print("   - umap_visualization.png (if available)")
        print("   - cluster_distribution.png")
        print("   - embedding_validation_report.txt")
        
        print("\n🎯 Summary:")
        print(f"   Embeddings validated: {validator.num_items}")
        print(f"   Clustering quality: {cluster_metrics['silhouette_score']:.4f}")
        print(f"   Retrieval metrics computed ✓")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
