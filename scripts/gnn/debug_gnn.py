"""
Working GraphSAGE Training - Fixed backward error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time


class GraphSAGEModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        
        self.conv1 = SAGEConv(embed_dim, 128, aggr='mean')
        self.conv2 = SAGEConv(128, embed_dim, aggr='mean')
        
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
    
    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x[:self.num_users], x[self.num_users:]


def evaluate(model, edge_index, test_df, user_to_idx, item_to_idx, user_history, num_items, device):
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
    
    test_df = test_df.copy()
    test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
    test_df['item_idx'] = test_df['item_id'].map(item_to_idx)
    test_df = test_df.dropna()
    test_df = test_df[test_df['user_idx'].apply(lambda x: int(x) in user_history)]
    
    if len(test_df) == 0:
        return 0.0, 0.0
    
    test_by_user = test_df.groupby('user_idx')['item_idx'].apply(lambda x: set([int(i) for i in x])).to_dict()
    
    hits = []
    times = []
    
    for user_idx, true_items in list(test_by_user.items())[:500]:
        user_idx = int(user_idx)
        
        start = time.time()
        
        with torch.no_grad():
            user_vec = user_emb[user_idx].unsqueeze(0)
            scores = (user_vec * item_emb).sum(dim=1)
            
            for train_item in user_history[user_idx]:
                scores[train_item] = -1e10
            
            _, topk = torch.topk(scores, 10)
            topk = set(topk.cpu().numpy())
        
        times.append((time.time() - start) * 1000)
        hits.append(len(topk & true_items) > 0)
    
    return np.mean(hits) * 100, np.mean(times)


def main():
    print("="*70)
    print("  GraphSAGE Training (Fixed)")
    print("="*70)
    
    CSV_PATH = "interactions.csv"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load
    print("\n[Loading]...")
    df = pd.read_csv(CSV_PATH)
    print(f"[Loaded] {len(df):,} interactions")
    
    users = sorted(df['user_id'].unique())
    items = sorted(df['item_id'].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['item_id'].map(item_to_idx)
    df = df.dropna()
    
    print(f"[Data] {len(users):,} users, {len(items):,} items")
    
    # Split
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    n = len(df)
    train_df = df.iloc[:int(n*0.7)]
    val_df = df.iloc[int(n*0.7):int(n*0.85)]
    test_df = df.iloc[int(n*0.85):]
    
    print(f"[Split] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Graph
    user_idx_arr = train_df['user_idx'].values.astype(int)
    item_idx_arr = (train_df['item_idx'].values + len(users)).astype(int)
    
    edges = np.vstack([
        np.concatenate([user_idx_arr, item_idx_arr]),
        np.concatenate([item_idx_arr, user_idx_arr])
    ])
    
    edge_index = torch.LongTensor(edges).to(DEVICE)
    
    print(f"[Graph] {edge_index.shape[1]:,} edges")
    
    # User history
    user_history = defaultdict(set)
    for u, i in zip(user_idx_arr, item_idx_arr - len(users)):
        user_history[u].add(i)
    
    # Model
    model = GraphSAGEModel(len(users), len(items), embed_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"[Model] {sum(p.numel() for p in model.parameters()):,} params\n")
    
    best_hit = 0
    
    for epoch in range(50):
        model.train()
        epoch_losses = []
        
        # Process users in batches
        all_users = list(user_history.keys())
        np.random.shuffle(all_users)
        
        n_batches = len(all_users) // 256 + 1
        
        for batch_idx in range(n_batches):
            batch_users = all_users[batch_idx*256:(batch_idx+1)*256]
            
            batch_u, batch_pos, batch_neg = [], [], []
            
            for u_idx in batch_users:
                pos_items = list(user_history[u_idx])
                
                for _ in range(3):
                    pos = np.random.choice(pos_items)
                    neg = np.random.randint(0, len(items))
                    
                    attempts = 0
                    while neg in pos_items and attempts < 20:
                        neg = np.random.randint(0, len(items))
                        attempts += 1
                    
                    batch_u.append(u_idx)
                    batch_pos.append(pos)
                    batch_neg.append(neg)
            
            if len(batch_u) == 0:
                continue
            
            batch_u = torch.LongTensor(batch_u).to(DEVICE)
            batch_pos = torch.LongTensor(batch_pos).to(DEVICE)
            batch_neg = torch.LongTensor(batch_neg).to(DEVICE)
            
            # CRITICAL: Call forward inside the batch loop
            user_emb, item_emb = model(edge_index)
            
            pos_scores = (user_emb[batch_u] * item_emb[batch_pos]).sum(dim=1)
            neg_scores = (user_emb[batch_u] * item_emb[batch_neg]).sum(dim=1)
            
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        
        if (epoch + 1) % 5 == 0:
            hit10, inf_time = evaluate(model, edge_index, val_df, user_to_idx, item_to_idx, user_history, len(items), DEVICE)
            print(f"Epoch {epoch+1}/50 | Loss: {avg_loss:.4f} | Hit@10: {hit10:.2f}% | Time: {inf_time:.1f}ms")
            
            if hit10 > best_hit:
                best_hit = hit10
                torch.save(model.state_dict(), 'best_model.pt')
        else:
            print(f"Epoch {epoch+1}/50 | Loss: {avg_loss:.4f}")
    
    print(f"\nBest Hit@10: {best_hit:.2f}%")


if __name__ == "__main__":
    main()
