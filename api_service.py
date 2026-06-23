import requests
import pandas as pd
import streamlit as st

def get_api_recommendations(user_ratings_df, movies_df, top_k=10):
    """Calls the FastAPI backend to generate movie recommendations."""
    liked_movies = user_ratings_df[user_ratings_df['rating'] >= 3]
    if liked_movies.empty:
        return []

    liked_ml_ids = []
    for tmdb_id in liked_movies['movie_id']:
        match = movies_df[movies_df['movieid'] == tmdb_id]
        if not match.empty:
            liked_ml_ids.append(int(match.iloc[0]['movieid']))
    
    try:
        url = "http://movie_app:8001/recommend" 
        payload = {"liked_ml_ids": liked_ml_ids, "top_k": top_k}
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("recommendations", [])
        else:
            st.error(f"API Error: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to recommendation engine: {e}")
        return []

def extract_recommendation(rec_ids, movies_df):
    """Merges the returned MovieLens IDs with the movie metadata."""
    rec_df = pd.DataFrame({"movielensid": rec_ids})
    rec_table = rec_df.merge(
        movies_df,
        left_on="movielensid",
        right_on="movieid",
        how="left"
    )
    return rec_table[["movielensid", "movieid", "original_title", "release_date", "overview"]]