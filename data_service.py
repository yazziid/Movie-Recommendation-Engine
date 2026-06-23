import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

# Centralized engine pointing to your Docker network database
engine = create_engine("postgresql+psycopg2://db_user:db_password@movieRec_postgres:5432/movieRec")

@st.cache_data
def load_ui_movies():
    """Loads and caches the movie metadata from PostgreSQL."""
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
    """Loads the user's watchlist directly from PostgreSQL instead of CSV."""
    query = text("SELECT movie_id, title, year, rating, poster_path FROM user_watchlist")
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df

def save_movie_rating(movie, rating):
    """Inserts or updates a movie rating directly in the SQL database."""
    poster = movie.get("poster_path") or ""
    year = movie["release_date"][:4] if movie.get("release_date") else "N/A"
    
    # PostgreSQL UPSERT statement (ON CONFLICT DO UPDATE)
    upsert_query = text("""
        INSERT INTO user_watchlist (movie_id, title, year, rating, poster_path)
        VALUES (:movie_id, :title, :year, :rating, :poster_path)
        ON CONFLICT (movie_id) 
        DO UPDATE SET rating = EXCLUDED.rating, poster_path = EXCLUDED.poster_path;
    """)
    
    with engine.connect() as conn:
        conn.execute(upsert_query, {
            "movie_id": movie["id"],
            "title": movie["title"],
            "year": year,
            "rating": rating,
            "poster_path": poster
        })
        conn.commit()
    return movie['title']

def delete_movie_rating(movie_id):
    """Deletes a movie from the user's database watchlist."""
    delete_query = text("DELETE FROM user_watchlist WHERE movie_id = :movie_id")
    with engine.connect() as conn:
        conn.execute(delete_query, {"movie_id": movie_id})
        conn.commit()