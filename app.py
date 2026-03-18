import streamlit as st
import pandas as pd
import requests
import os
import torch
import pickle
from streamlit_searchbox import st_searchbox
from info import TMDB_API_KEY  # Ensure this is your Bearer token
from TwoTowerArchitecture import TwoTowerWithItemFeatures
from sqlalchemy import create_engine, text

@st.cache_data
def load_ui_movies():
    #engine = create_engine("postgresql+psycopg2://db_user:db_password@localhost:5402/movieRec")
    engine = create_engine("postgresql+psycopg2://db_user:db_password@movieRec_postgres:5432/movieRec")
    query = text("""
        SELECT movieId AS movieid, original_title, release_date, overview 
        FROM movies
        WHERE original_title IS NOT NULL 
          AND original_title != 'NaN'
          AND revenue > 0 
          AND lang_en = 1
    """)
    
    with engine.connect() as conn: df = pd.read_sql_query(query, conn)
        
    df["movieid"] = pd.to_numeric(df["movieid"], errors="coerce")
    return df

movies_df = load_ui_movies()

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
CSV_FILE = "user_movie_ratings.csv"

def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "poster_path" not in df.columns:
            df["poster_path"] = ""
        return df
    return pd.DataFrame(columns=["movie_id", "title", "year", "rating", "poster_path"])

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

def save_rating(movie, rating):
    df = load_data()
    poster = movie.get("poster_path") or ""
    
    if movie["id"] in df["movie_id"].values:
        df.loc[df["movie_id"] == movie["id"], "rating"] = rating
        if poster:
            df.loc[df["movie_id"] == movie["id"], "poster_path"] = poster
    else:
        new_row = {
            "movie_id": movie["id"],
            "title": movie["title"],
            "year": movie["release_date"][:4] if movie.get("release_date") else "N/A",
            "rating": rating,
            "poster_path": poster
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    save_data(df)
    st.success(f"Updated {movie['title']}!")

def get_poster_url(poster_path, tmdb_id):
    """Generates the image URL, or fetches it dynamically if missing from old CSVs."""
    if pd.notna(poster_path) and poster_path:
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}", headers=headers, timeout=5)
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w300{path}"
    except Exception:
        pass
    
    return "https://via.placeholder.com/300x450?text=No+Poster"

# helper function for Search
def search_tmdb_suggestions(query):
    """The function searchbox calls as you type."""
    if not query or len(query) < 2:
        return []

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }
    params = {"query": query, "language": "en-US"}
    try:
        response = requests.get(SEARCH_URL, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            # Format: [(Label to show, Data to return)]
            return [(f"{m['title']} ({m.get('release_date', '')[:4]})", m) for m in results if m.get("title")]
    except Exception:
        return []
    return []

# Recommendation Helpers
@st.cache_resource
def load_recommender_model():
    """Loads the pre-trained Two-Tower model and mapping artifacts."""
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
        return model, artifacts
    except FileNotFoundError:
        st.error("Model files not found. Please run your train_eval.py script first.")
        return None, None


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

def get_dynamic_recommendations(user_ratings_df, model, artifacts, movies_df, top_k=10):
    """Generates recommendations by averaging the vectors of movies you liked."""
    # 1. Filter for movies you actually liked (Rating >= 3)
    liked_movies = user_ratings_df[user_ratings_df['rating'] >= 3]
    if liked_movies.empty:
        return []

    liked_indices = []
    for tmdb_id in liked_movies['movie_id']:
        match = movies_df[movies_df['movieid'] == tmdb_id]
        if not match.empty:
            ml_id = match.iloc[0]['movieid'] 
            if ml_id in artifacts["movieid2idx"]:
                liked_indices.append(artifacts["movieid2idx"][ml_id])
    
    if not liked_indices:
        return []

    item_vectors = artifacts["item_vectors"]
    liked_vectors = item_vectors[liked_indices]
    dynamic_user_vec = liked_vectors.mean(dim=0).unsqueeze(0) # Shape: [1, emb_dim]

    # Compute scores against all items
    with torch.no_grad():
        scores = torch.matmul(dynamic_user_vec, item_vectors.T).squeeze()
        
        for idx in liked_indices:
            scores[idx] = -float('inf')
            
        top_indices = torch.topk(scores, k=top_k).indices.numpy()
    
    return [artifacts["idx2movieid"][i] for i in top_indices]

# Streamlit UI
st.title(" Movie Rating App")

# Single search widget for suggestions
selected_movie = st_searchbox(
    search_tmdb_suggestions,
    placeholder="Search and select a movie...",
    key="movie_search_box"
)

if selected_movie:
    st.divider()
    col1, col2 = st.columns([1, 2]) # Creates a 1/3 and 2/3 column layout
    
    with col1:
        # Display the poster if it exists in TMDB
        poster_path = selected_movie.get('poster_path')
        if poster_path:
            image_url = f"https://image.tmdb.org/t/p/w300{poster_path}"
            st.image(image_url, use_container_width=True)
        else:
            st.info("No poster available for this movie.")
            
    with col2:
        # Display details and rating slider
        st.write(f"### Rate: {selected_movie['title']}")
        st.caption(f"Released: {selected_movie.get('release_date', 'N/A')}")
        st.write(selected_movie.get('overview', 'No overview available.'))
        
        rating = st.slider("Rating (1-5)", 1, 5, 3)
        if st.button("Save/Update Selection", use_container_width=True):
            save_rating(selected_movie, rating)
            st.rerun()

# Display History
st.divider()
st.subheader("Your Ratings")
df = load_data()

if not df.empty:
    # Create a grid of 4 columns per row
    cols = st.columns(4)
    
    # Loop through the dataframe and distribute items across the columns
    for idx, (original_idx, row) in enumerate(df.sort_index(ascending=False).iterrows()):
        col = cols[idx % 4]
        
        with col:
            # 1. Show the Poster
            poster_url = get_poster_url(row.get('poster_path'), row['movie_id'])
            st.image(poster_url, use_container_width=True)
            
            # 2. Show the Title under the poster
            st.markdown(f"**{row['title']}**")
            
            # 3. The Expander (Slidebar) for details and actions
            with st.expander("Details & Edit"):
                st.write(f"**Year:** {row['year']}")
                st.write(f"**Current Rating:** {row['rating']}/5")
                
                # Edit and Delete Controls
                new_val = st.number_input("Change Rating", 1, 5, int(row["rating"]), key=f"edit_{row['movie_id']}")
                
                c1, c2 = st.columns(2)
                if c1.button("Update", key=f"upd_{row['movie_id']}"):
                    # Pass the poster_path so it doesn't get lost on update
                    save_rating({
                        "id": row["movie_id"], 
                        "title": row["title"], 
                        "poster_path": row.get("poster_path", "")
                    }, new_val)
                    st.rerun()
                    
                if c2.button("Delete", key=f"del_{row['movie_id']}"):
                    df_new = df[df["movie_id"] != row["movie_id"]]
                    save_data(df_new)
                    st.rerun()
else:
    st.info("Your watchlist is currently empty.")

# Recommendations Section (UI)
st.divider()
st.subheader(" Recommended for You")

num_ratings = len(df)
needs_more = num_ratings < 5

if needs_more:
    st.info(f"You have rated {num_ratings}/5 movies. Rate at least {5 - num_ratings} more to unlock AI recommendations!")

if st.button("Generate AI Recommendations", disabled=needs_more):
    model, artifacts = load_recommender_model()
    
    if model and artifacts and not movies_df.empty:
        with st.spinner("Analyzing your tastes and fetching posters..."):
            rec_ids = get_dynamic_recommendations(df, model, artifacts, movies_df)
            
            if not rec_ids:
                st.warning("Our database is too small! We couldn't match your rated movies to our training data. Try rating older or more popular movies.")
            else:
                final_table = extract_recommendation(rec_ids, movies_df)
                
                st.write("### Here are some movies you might like:")
                
                # Create the 4-column grid for recommendations
                rec_cols = st.columns(4)
                
                for idx, row in final_table.iterrows():
                    col = rec_cols[idx % 4]
                    
                    with col:
                        # Fetch the poster dynamically using the TMDB ID
                        tmdb_id = row['movieid']
                        poster_url = get_poster_url("", tmdb_id)
                        
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{row['original_title']}**")
                        
                        # Hidden details in the expander
                        with st.expander("More Details"):
                            st.write(f"**Released:** {row['release_date']}")
                            st.caption(row['overview'])
