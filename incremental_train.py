import pandas as pd
import torch
import pickle
from sqlalchemy import create_engine, text
from torch.utils.data import Dataset, DataLoader
from TwoTowerArchitecture import TwoTowerWithItemFeatures, inbatch_bpr_loss

class NewRatingsDataset(Dataset):
    def __init__(self, user_idx, item_indices):
        self.user_idx = user_idx
        self.item_indices = item_indices

    def __len__(self):
        return len(self.item_indices)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_idx, dtype=torch.long),
            torch.tensor(self.item_indices[idx], dtype=torch.long)
        )

def run_incremental_training(csv_path="user_movie_ratings.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with open("model_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
    except FileNotFoundError:
        return False, "Model artifacts not found. Run base training first."

    try:
        #engine = create_engine("postgresql+psycopg2://db_user:db_password@localhost:5402/movieRec")
        engine = create_engine("postgresql+psycopg2://db_user:db_password@movieRec_postgres:5432/movieRec")
        movies_df = pd.read_sql_query(text("SELECT movieId AS movieid FROM movies"), engine)
        movies_df["movieid"] = pd.to_numeric(movies_df["movieid"], errors="coerce")
        user_df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Data loading failed: {e}"

    liked_movies = user_df[user_df['rating'] >= 3]
    if liked_movies.empty:
        return False, "No highly rated movies to train on."

    item_indices = []
    for tmdb_id in liked_movies['movie_id']:
        match = movies_df[movies_df['movieid'] == tmdb_id]
        if not match.empty:
            ml_id = match.iloc[0]['movieid']
            if ml_id in artifacts["movieid2idx"]:
                item_indices.append(artifacts["movieid2idx"][ml_id])

    if not item_indices:
        return False, "None of the rated movies exist in the model's vocabulary."

    # Initialize Model
    model = TwoTowerWithItemFeatures(
        n_users=len(artifacts["userid2idx"]),
        n_items=len(artifacts["movieid2idx"]),
        item_feat_dim=3, 
        n_actors=artifacts["n_actors"],
        K=artifacts.get("K", 5),
        emb_dim=128,   
        hidden_dim=264
    )
    model.load_state_dict(torch.load("two_tower_model.pth", map_location=device))
    model.to(device)
    model.train()

    user_idx = artifacts["userid2idx"].get(1, 1) 
    
    dataset = NewRatingsDataset(user_idx, item_indices)
    loader = DataLoader(dataset, batch_size=16, shuffle=True) 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    if "X_item" not in artifacts or "A_item" not in artifacts:
        return False, "Critical Error: Please update train_eval.py to save X_item and A_item into model_artifacts.pkl so we can run the forward pass during incremental training."

    X_item = artifacts["X_item"].to(device)
    A_item = artifacts["A_item"].to(device)

    # Here we train for just 2 Micro-Epochs
    for epoch in range(2):
        item_vectors = model.compute_all_item_vectors(X_item, A_item)
        for u, pos_i in loader:
            u, pos_i = u.to(device), pos_i.to(device)
            optimizer.zero_grad()
            user_vec = model.user_tower(u)
            pos_item_vec = item_vectors[pos_i]
            
            loss = inbatch_bpr_loss(user_vec, pos_item_vec)
            loss.backward(retain_graph=True)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        updated_item_vectors = model.compute_all_item_vectors(X_item, A_item).cpu()

    torch.save(model.state_dict(), "two_tower_model.pth")
    artifacts["item_vectors"] = updated_item_vectors
    
    with open("model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    return True, "Incremental training completed successfully!"