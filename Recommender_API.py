from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import faiss

app = FastAPI(title="Movie Recommendation API")

artifacts = None
faiss_index = None

@app.on_event("startup")
def load_artifacts_and_index():
    global artifacts, faiss_index
    try:
        # Load mappings and precomputed vectors
        with open("model_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
        
        # Load the FAISS vector index
        faiss_index = faiss.read_index("movie_vectors.index")
        print("Artifacts and FAISS Index loaded successfully.")
    except Exception as e:
        print(f"Failed to load artifacts or index: {e}")

class RecommendRequest(BaseModel):
    liked_ml_ids: List[int]
    top_k: int = 10

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if not artifacts or not faiss_index:
        raise HTTPException(status_code=503, detail="Model artifacts or FAISS index are not loaded.")

    # 1. Map requested IDs to internal tensor indices
    liked_indices = [
        artifacts["movieid2idx"][ml_id] 
        for ml_id in req.liked_ml_ids 
        if ml_id in artifacts["movieid2idx"]
    ]
    
    if not liked_indices:
        return {"recommendations": []}

    # 2. Compute dynamic user vector from artifacts
    item_vectors = artifacts["item_vectors"]
    liked_vectors = item_vectors[liked_indices]
    
    # 3. Format the vector for FAISS (numpy float32, shape: [1, dimension])
    dynamic_user_vec = liked_vectors.mean(dim=0).numpy().astype('float32').reshape(1, -1)

    # 4. Query FAISS
    # We ask for extra items (top_k + len) so we have backups if the top results 
    # are movies the user has already liked.
    search_k = req.top_k + len(liked_indices)
    scores, indices = faiss_index.search(dynamic_user_vec, search_k)
    
    # 5. Filter out items the user has already liked
    recommended_indices = []
    for idx in indices[0]:
        if idx not in liked_indices:
            recommended_indices.append(idx)
        if len(recommended_indices) == req.top_k:
            break
            
    # 6. Map back to original MovieLens IDs
    rec_ml_ids = [int(artifacts["idx2movieid"][i]) for i in recommended_indices]
    return {"recommendations": rec_ml_ids}