import pandas as pd
import os
from sqlalchemy import create_engine, text
import streamlit as st

CSV_FILE = "user_movie_ratings.csv"

@st.cache_data
def load_ui_movies():
    """Loads and caches the movie metadata from PostgreSQL."""
    engine = create_engine("postgresql+psycopg2://db_user:db_password@movieRec_postgres:5432/movieRec")
    query = text("""
        SELECT movieId AS movieid, original_title, release_date, overview 
        FROM movies
        WHERE original_title IS NOT NULL 
          AND original_title != 'NaN'
          AND revenue > 0 
          AND lang_en = TRUE
    """)
    with engine.connect() as conn: 
        df = pd.read_sql_query(query, conn)
        
    df["movieid"] = pd.to_numeric(df["movieid"], errors="coerce")
    return df

def load_ratings_data():
    """Loads the user's ratings from the local CSV."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "poster_path" not in df.columns:
            df["poster_path"] = ""
        return df
    return pd.DataFrame(columns=["movie_id", "title", "year", "rating", "poster_path"])

def save_ratings_data(df):
    """Saves the ratings dataframe to CSV."""
    df.to_csv(CSV_FILE, index=False)

def save_movie_rating(movie, rating):
    """Adds or updates a movie rating in the CSV."""
    df = load_ratings_data()
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
        
    save_ratings_data(df)
    return movie['title']