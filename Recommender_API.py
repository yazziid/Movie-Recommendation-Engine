from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import faiss
import os

app = FastAPI(title="Movie Recommendation API")

# Global cache to keep multiple models loaded in memory to prevent slow disk reads
MODEL_CACHE = {}

def get_model_assets(version: str):
    """Loads and caches model components dynamically based on version folder."""
    if version in MODEL_CACHE:
        return MODEL_CACHE[version]
    
    model_dir = f"models/{version}"
    artifacts_path = f"{model_dir}/model_artifacts.pkl"
    index_path = f"{model_dir}/movie_vectors.index"
    
    if not os.path.exists(artifacts_path) or not os.path.exists(index_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model version '{version}' not found on disk. Run training first."
        )
        
    try:
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        index = faiss.read_index(index_path)
        
        MODEL_CACHE[version] = {"artifacts": artifacts, "index": index}
        return MODEL_CACHE[version]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {version}: {e}")

class RecommendRequest(BaseModel):
    liked_ml_ids: List[int]
    model_version: str = "v1"
    top_k: int = 10

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    # Fetch components dynamically out of our model version folder
    assets = get_model_assets(req.model_version)
    artifacts = assets["artifacts"]
    faiss_index = assets["index"]

    liked_indices = [
        artifacts["movieid2idx"][ml_id] 
        for ml_id in req.liked_ml_ids 
        if ml_id in artifacts["movieid2idx"]
    ]
    
    if not liked_indices:
        return {"recommendations": []}

    item_vectors = artifacts["item_vectors"]
    liked_vectors = item_vectors[liked_indices]
    dynamic_user_vec = liked_vectors.mean(dim=0).numpy().astype('float32').reshape(1, -1)

    search_k = req.top_k + len(liked_indices)
    scores, indices = faiss_index.search(dynamic_user_vec, search_k)
    
    recommended_indices = []
    for idx in indices[0]:
        if idx not in liked_indices:
            recommended_indices.append(idx)
        if len(recommended_indices) == req.top_k:
            break
            
    rec_ml_ids = [int(artifacts["idx2movieid"][i]) for i in recommended_indices]
    return {"recommendations": rec_ml_ids}