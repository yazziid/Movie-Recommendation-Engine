
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import pickle
from TwoTowerArchitecture import TwoTowerWithItemFeatures

app = FastAPI(title="Movie Recommendation API")

model = None
artifacts = None

@app.on_event("startup")
def load_model():
    global model, artifacts
    try:
        with open("model_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
        
        model = TwoTowerWithItemFeatures(
            n_users=len(artifacts["userid2idx"]),
            n_items=len(artifacts["movieid2idx"]),
            item_feat_dim=3, 
            n_actors=artifacts["n_actors"],
            K=artifacts.get("K", 5),
            emb_dim=128,      
            hidden_dim=264
        )
        model.load_state_dict(torch.load("two_tower_model.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

class RecommendRequest(BaseModel):
    liked_ml_ids: List[int]
    top_k: int = 10

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if not model or not artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    liked_indices = [
        artifacts["movieid2idx"][ml_id] 
        for ml_id in req.liked_ml_ids 
        if ml_id in artifacts["movieid2idx"]
    ]
    
    if not liked_indices:
        return {"recommendations": []}

    # Compute dynamic user vector
    item_vectors = artifacts["item_vectors"]
    liked_vectors = item_vectors[liked_indices]
    dynamic_user_vec = liked_vectors.mean(dim=0).unsqueeze(0)

    # Score all items and retrieve top K
    with torch.no_grad():
        scores = torch.matmul(dynamic_user_vec, item_vectors.T).squeeze()
        for idx in liked_indices:
            scores[idx] = -float('inf')
            
        top_indices = torch.topk(scores, k=req.top_k).indices.numpy()
    
    rec_ml_ids = [int(artifacts["idx2movieid"][i]) for i in top_indices]
    return {"recommendations": rec_ml_ids}