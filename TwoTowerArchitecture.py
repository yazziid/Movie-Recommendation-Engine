import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class TwoTowerWithItemFeatures(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        item_feat_dim,
        n_actors,
        K,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.1,
        normalize=True
    ):
        super().__init__()

        self.normalize = normalize
        self.K = K

        # USER TOWER
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )

        # ITEM COMPONENTS
        self.item_id_emb = nn.Embedding(n_items, emb_dim)
        self.actor_emb = nn.Embedding(n_actors, emb_dim)
        self.item_feat_proj = nn.Linear(item_feat_dim, emb_dim)
        fusion_dim = emb_dim * 3
        
        self.item_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )

    def user_tower(self, u):
        u = self.user_emb(u)
        u = self.user_mlp(u)
        if self.normalize:
            u = F.normalize(u, dim=-1)
        return u

    def compute_all_item_vectors(self, X_item, A_item):
        id_emb = self.item_id_emb.weight
        feat_emb = self.item_feat_proj(X_item)

        actors = self.actor_emb(A_item)
        actor_emb = actors.mean(dim=1)

        concat = torch.cat([id_emb, feat_emb, actor_emb], dim=1)

        item_vec = self.item_mlp(concat)

        if self.normalize:
            item_vec = F.normalize(item_vec, dim=1)

        return item_vec

def inbatch_bpr_loss(user_vec, item_vec):
    scores = torch.matmul(user_vec, item_vec.T)
    pos_scores = scores.diag()
    loss = -torch.log(
        torch.sigmoid(pos_scores.unsqueeze(1) - scores) + 1e-8).mean()
    return loss

def train_model(
    model,
    train_dataset,
    X_item,
    A_item,
    device,
    epochs=25,
    batch_size=4096 # decrease to 2048 then 1024 if too much
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-6
    )

    model.to(device)
    X_item = X_item.to(device)
    A_item = A_item.to(device)

    epoch_history = []

    for epoch in range(epochs):
        model.train()
        losses = []

        with torch.no_grad():
            item_vectors = model.compute_all_item_vectors(X_item, A_item)

        for u, pos_i in train_loader:
            u = u.to(device, non_blocking=True)
            pos_i = pos_i.to(device, non_blocking=True)
            optimizer.zero_grad()
            user_vec = model.user_tower(u)
            pos_item_vec = item_vectors[pos_i]
            loss = inbatch_bpr_loss(user_vec, pos_item_vec)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch {epoch}: {avg_loss:.4f}")
        
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })
        
    return epoch_history