import numpy as np
import torch

def calculate_metrics(model, test_df, item_vectors, userid2idx, movieid2idx, device, global_item_counts, total_train_interactions, topk=10):
    """Calculates comprehensive ranking and diversity metrics."""
    model.eval()
    hits = 0
    ndcg = 0
    mrr = 0
    novelty_score = 0
    
    total_users = 0
    recommended_unique_items = set()

    item_novelty = {}
    for item_ml_id, idx in movieid2idx.items():
        count = global_item_counts.get(item_ml_id, 1) 
        prob = count / total_train_interactions
        item_novelty[idx] = -np.log2(prob)

    test_user_groups = test_df.groupby('userid')['movielensid'].apply(list).to_dict()

    with torch.no_grad():
        for user_id, true_item_ids in test_user_groups.items():
            if user_id not in userid2idx: continue
                
            true_item_idxs = [movieid2idx[m] for m in true_item_ids if m in movieid2idx]
            if not true_item_idxs: continue

            uidx = torch.tensor([userid2idx[user_id]], device=device)
            uvec = model.user_tower(uidx)
            
            scores = torch.matmul(uvec, item_vectors.T).squeeze()
            top_items = torch.topk(scores, topk).indices.cpu().numpy()

            # 1. Recall & Coverage
            if any(item in top_items for item in true_item_idxs): hits += 1
            for item in top_items: recommended_unique_items.add(item)
            
            # 2. NDCG
            dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top_items) if item in true_item_idxs)
            idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(true_item_idxs), topk)))
            if idcg > 0: ndcg += dcg / idcg

            # 3. MRR
            for rank, item in enumerate(top_items):
                if item in true_item_idxs:
                    mrr += 1 / (rank + 1)
                    break 

            # 4. Novelty
            novelty_score += sum(item_novelty.get(item, 0) for item in top_items) / topk
            total_users += 1

    return {
        "Recall@10": hits / total_users if total_users > 0 else 0,
        "NDCG@10": ndcg / total_users if total_users > 0 else 0,
        "MRR@10": mrr / total_users if total_users > 0 else 0,
        "Catalog_Coverage": len(recommended_unique_items) / len(movieid2idx),
        "Novelty": novelty_score / total_users if total_users > 0 else 0
    }