import pandas as pd
import torch
import pickle

from model_prep import train_df, TwoTowerDataset, index_mapping, top_k_actor_per_movie, X_item, movies_df
from TwoTowerArchitecture import TwoTowerWithItemFeatures, train_model

n_users, n_items, idx2movieid, userid2idx, movieid2idx = index_mapping(train_df)
A_item, n_actors, K = top_k_actor_per_movie()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_train_dataset(subset_size = 0):
    if subset_size > 1:
        train_df_small = train_df.sample(
            n=min(subset_size, len(train_df)),
            random_state=42)

        implicit_train = train_df_small[["userid", "movielensid"]].drop_duplicates()

    else: implicit_train = train_df[["userid","movielensid"]].drop_duplicates()

    train_dataset = TwoTowerDataset(implicit_train, userid2idx, movieid2idx)
    return train_dataset

train_dataset = generate_train_dataset()

def create_model(n_users=n_users, n_items=n_items, item_feat_dim=X_item.shape[1], n_actors=n_actors, K=K, emb_dim=64, hidden_dim=128, dropout=0.1, normalize=True):
    return TwoTowerWithItemFeatures(
        n_users=n_users,
        n_items=n_items,
        item_feat_dim=X_item.shape[1],
        n_actors=n_actors,
        K=K,
        emb_dim=128,
        hidden_dim=264,
        dropout=0.1,
        normalize=True
    )
    
model = create_model()

print("starting training")
train_model(
    model=model,
    train_dataset=train_dataset,
    X_item=X_item,
    A_item=A_item,
    device=device,
    epochs=10,
    batch_size=3072
)

print("evaluating model")
model.eval()

X_item = X_item.to(device)
A_item = A_item.to(device)

with torch.no_grad():
    item_vectors = model.compute_all_item_vectors(X_item,A_item)

print("test recommendation")

def recommend(idx2movieid, userid2idx, model, user_id, topk = 10):
    if user_id not in userid2idx: return []
    uidx = torch.tensor([userid2idx[user_id]],device=device)
    with torch.no_grad():
        uvec = model.user_tower(uidx)
        scores = torch.matmul(uvec,item_vectors.T).squeeze()
        top_items = torch.topk(scores,topk).indices.cpu().numpy()
    return [idx2movieid[i] for i in top_items]

rec_ids = recommend(idx2movieid, userid2idx, model, user_id=1, topk=10)

def extract_recommendation(rec_ids, movies_df):
    """Your exact function from train_eval.py to format the output."""
    rec_df = pd.DataFrame({"movielensid": rec_ids})
    rec_table = rec_df.merge(
        movies_df,
        left_on="movielensid",
        right_on="movieid",
        how="left"
    )
    return rec_table[["movielensid", "movieid", "original_title", "release_date", "overview"]]

print("top 10 recommendation for user 1:\n")
recommendation_table = extract_recommendation(rec_ids, movies_df)
print(recommendation_table)


torch.save(model.state_dict(), "two_tower_model.pth")

model_artifacts = {
    "userid2idx": userid2idx,
    "movieid2idx": movieid2idx,
    "idx2movieid": idx2movieid,
    "item_vectors": item_vectors.cpu(),
    "n_actors": n_actors,
    "K": K                
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

print("Model and artifacts saved successfully.")
