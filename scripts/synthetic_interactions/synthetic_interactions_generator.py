"""
Batch Interaction Generator - Generate interactions in chunks

This allows you to:
1. Generate interactions for a subset of users
2. Resume from where you left off
3. Combine multiple batches into final file
"""

import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import argparse


class BatchInteractionGenerator:
    """Generate interactions in batches"""
    
    def __init__(self,
                 users_path: str,
                 embeddings_path: str,
                 cluster_info_path: str,
                 simulation_months: int = 6):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Device] Using {self.device}")
        
        # Load users
        print(f"[Loading] Users from {users_path}")
        with open(users_path, 'r') as f:
            self.all_users = json.load(f)
        
        # Load embeddings
        print(f"[Loading] Embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings_np = data['embeddings']
        self.item_ids = data['item_ids']
        
        # Move to GPU
        self.embeddings_gpu = torch.from_numpy(embeddings_np).float().to(self.device)
        
        # Load cluster info
        print(f"[Loading] Cluster info from {cluster_info_path}")
        with open(cluster_info_path, 'r') as f:
            cluster_data = json.load(f)
        
        # Compute cluster labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cluster_data['n_clusters'], random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(embeddings_np)
        
        self.simulation_months = simulation_months
        
        # Create cluster mapping
        self.cluster_to_items = defaultdict(list)
        for idx, cluster_id in enumerate(self.cluster_labels):
            self.cluster_to_items[cluster_id].append(idx)
        
        print(f"[Ready] {len(self.all_users)} total users")
    
    def precompute_similarities_batch(self, batch_size=1000, k=200):
        """Precompute top-K similar items"""
        print(f"[GPU] Precomputing top-{k} similarities...")
        
        self.topk_similar_items = {}
        self.topk_similarities = {}
        
        embeddings_norm = torch.nn.functional.normalize(self.embeddings_gpu, p=2, dim=1)
        n_items = len(self.embeddings_gpu)
        
        for i in tqdm(range(0, n_items, batch_size), desc="Computing similarities"):
            batch_end = min(i + batch_size, n_items)
            batch_embeddings = embeddings_norm[i:batch_end]
            
            similarities = torch.mm(batch_embeddings, embeddings_norm.t())
            topk_sim, topk_idx = torch.topk(similarities, k=k, dim=1)
            
            topk_sim_cpu = topk_sim.cpu().numpy()
            topk_idx_cpu = topk_idx.cpu().numpy()
            
            for j in range(batch_end - i):
                item_idx = i + j
                self.topk_similar_items[item_idx] = topk_idx_cpu[j]
                self.topk_similarities[item_idx] = topk_sim_cpu[j]
        
        print(f"[GPU] ✅ Precomputed similarities")
    
    def generate_batch(self, start_user: int, end_user: int):
        """Generate interactions for a batch of users"""
        batch_users = self.all_users[start_user:end_user]
        interactions = []
        
        start_date = datetime.now() - timedelta(days=30 * self.simulation_months)
        
        print(f"\n[Batch] Processing users {start_user} to {end_user} ({len(batch_users)} users)")
        
        for user in tqdm(batch_users, desc=f"Users {start_user}-{end_user}"):
            # Determine active days
            total_days = 30 * self.simulation_months
            active_days = []
            
            for month in range(self.simulation_months):
                month_active = int(np.random.poisson(user['active_days_per_month']))
                if month_active > 0:
                    days_in_month = min(month_active, 30)
                    month_days = np.random.choice(30, size=days_in_month, replace=False)
                    active_days.extend(month_days + (month * 30))
            
            # Generate sessions
            for day_offset in active_days[:int(total_days * 0.3)]:
                session_date = start_date + timedelta(days=int(day_offset))
                session = self._simulate_session(user, session_date)
                interactions.extend(session)
        
        return interactions
    
    def _simulate_session(self, user, session_date):
        """Simulate single session"""
        session = []
        session_id = f"{user['user_id']}_{session_date.strftime('%Y%m%d_%H%M%S')}"
        
        session_length = max(1, int(np.random.poisson(user['avg_session_length'])))
        
        # Start item
        preferred_clusters = [int(k) for k in user['preferred_clusters'].keys()]
        cluster_weights = list(user['preferred_clusters'].values())
        
        if np.random.random() < user['exploration_rate']:
            current_cluster = np.random.randint(0, len(self.cluster_to_items))
        else:
            current_cluster = np.random.choice(preferred_clusters, p=cluster_weights)
        
        if len(self.cluster_to_items[current_cluster]) == 0:
            current_cluster = np.random.randint(0, len(self.cluster_to_items))
        
        current_item_idx = np.random.choice(self.cluster_to_items[current_cluster])
        
        # Generate interactions
        for step in range(session_length):
            timestamp = session_date + timedelta(seconds=step * 30)
            
            # View
            session.append({
                'user_id': user['user_id'],
                'item_id': str(self.item_ids[current_item_idx]),
                'item_index': int(current_item_idx),
                'cluster_id': int(self.cluster_labels[current_item_idx]),
                'interaction_type': 'view',
                'timestamp': timestamp.isoformat(),
                'session_id': session_id
            })
            
            # Click (70%)
            if np.random.random() < 0.7:
                session.append({
                    'user_id': user['user_id'],
                    'item_id': str(self.item_ids[current_item_idx]),
                    'item_index': int(current_item_idx),
                    'cluster_id': int(self.cluster_labels[current_item_idx]),
                    'interaction_type': 'click',
                    'timestamp': (timestamp + timedelta(seconds=5)).isoformat(),
                    'session_id': session_id
                })
                
                # Add to cart (20%)
                if np.random.random() < 0.2:
                    session.append({
                        'user_id': user['user_id'],
                        'item_id': str(self.item_ids[current_item_idx]),
                        'item_index': int(current_item_idx),
                        'cluster_id': int(self.cluster_labels[current_item_idx]),
                        'interaction_type': 'add_to_cart',
                        'timestamp': (timestamp + timedelta(seconds=10)).isoformat(),
                        'session_id': session_id
                    })
                    
                    # Purchase
                    if np.random.random() < user['conversion_rate']:
                        session.append({
                            'user_id': user['user_id'],
                            'item_id': str(self.item_ids[current_item_idx]),
                            'item_index': int(current_item_idx),
                            'cluster_id': int(self.cluster_labels[current_item_idx]),
                            'interaction_type': 'purchase',
                            'timestamp': (timestamp + timedelta(seconds=20)).isoformat(),
                            'session_id': session_id
                        })
            
            # Navigate to next item
            if step < session_length - 1:
                if current_item_idx in self.topk_similar_items:
                    similar_indices = self.topk_similar_items[current_item_idx][:100]
                    similarities = self.topk_similarities[current_item_idx][:100]
                else:
                    similar_indices = np.random.choice(len(self.item_ids), 100, replace=False)
                    similarities = np.random.random(100)
                
                # Stay in cluster (80%)
                if np.random.random() < 0.8:
                    same_cluster_mask = np.array([self.cluster_labels[idx] == self.cluster_labels[current_item_idx] 
                                                   for idx in similar_indices])
                    if same_cluster_mask.sum() > 0:
                        same_cluster_indices = similar_indices[same_cluster_mask]
                        same_cluster_sims = similarities[same_cluster_mask]
                        weights = np.exp(same_cluster_sims * 5)
                        weights /= weights.sum()
                        current_item_idx = int(np.random.choice(same_cluster_indices, p=weights))
                    else:
                        current_item_idx = int(np.random.choice(similar_indices[:50]))
                else:
                    current_item_idx = int(np.random.choice(similar_indices[:50]))
        
        return session
    
    def save_batch(self, interactions, batch_num, output_dir):
        """Save batch to file"""
        output_path = Path(output_dir) / f"interactions_batch_{batch_num:04d}.json"
        with open(output_path, 'w') as f:
            json.dump(interactions, f)
        print(f"[Saved] Batch {batch_num}: {len(interactions)} interactions -> {output_path}")
        return output_path


def combine_batches(batch_dir, output_path):
    """Combine all batch files into one"""
    print("\n[Combining] Merging all batches...")
    
    batch_files = sorted(Path(batch_dir).glob("interactions_batch_*.json"))
    all_interactions = []
    
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_interactions.extend(batch_data)
    
    print(f"[Combining] Total interactions: {len(all_interactions)}")
    
    with open(output_path, 'w') as f:
        json.dump(all_interactions, f, indent=2)
    
    print(f"[Saved] Combined file: {output_path}")
    
    # Print statistics
    df = pd.DataFrame(all_interactions)
    print(f"\n[Stats] Total interactions: {len(df):,}")
    print(f"[Stats] Unique users: {df['user_id'].nunique():,}")
    print(f"[Stats] Unique items: {df['item_id'].nunique():,}")
    print(f"\n[Stats] Interaction types:")
    print(df['interaction_type'].value_counts())


def main():
    parser = argparse.ArgumentParser(description='Generate interactions in batches')
    parser.add_argument('--users', required=True, help='Path to synthetic_users.json')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings .npz')
    parser.add_argument('--clusters', required=True, help='Path to cluster info JSON')
    parser.add_argument('--output-dir', default='./batches', help='Output directory for batches')
    parser.add_argument('--batch-size', type=int, default=5000, help='Users per batch')
    parser.add_argument('--start-batch', type=int, default=0, help='Starting batch number (for resume)')
    parser.add_argument('--combine-only', action='store_true', help='Only combine existing batches')
    parser.add_argument('--months', type=int, default=6, help='Simulation months')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.combine_only:
        combine_batches(output_dir, output_dir / 'synthetic_interactions_combined.json')
        return
    
    print("="*70)
    print("  BATCH INTERACTION GENERATOR")
    print("="*70)
    
    # Initialize generator
    generator = BatchInteractionGenerator(
        users_path=args.users,
        embeddings_path=args.embeddings,
        cluster_info_path=args.clusters,
        simulation_months=args.months
    )
    
    # Precompute similarities once
    generator.precompute_similarities_batch()
    
    # Generate batches
    total_users = len(generator.all_users)
    num_batches = (total_users + args.batch_size - 1) // args.batch_size
    
    print(f"\n[Plan] Total users: {total_users}")
    print(f"[Plan] Batch size: {args.batch_size}")
    print(f"[Plan] Total batches: {num_batches}")
    print(f"[Plan] Starting from batch: {args.start_batch}")
    
    for batch_num in range(args.start_batch, num_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_users)
        
        print(f"\n{'='*70}")
        print(f"  BATCH {batch_num + 1}/{num_batches}")
        print(f"{'='*70}")
        
        interactions = generator.generate_batch(start_idx, end_idx)
        generator.save_batch(interactions, batch_num, output_dir)
        
        print(f"✅ Batch {batch_num + 1} complete")
    
    # Combine all batches
    print(f"\n{'='*70}")
    print("  COMBINING BATCHES")
    print(f"{'='*70}")
    
    combine_batches(output_dir, output_dir / 'synthetic_interactions_combined.json')
    
    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
