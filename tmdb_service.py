import requests
import pandas as pd
from info import TMDB_API_KEY

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

def get_poster_url(poster_path, tmdb_id):
    """Generates the TMDB image URL, or fetches it dynamically."""
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

def search_tmdb_suggestions(query):
    """Fetches movie suggestions for the Streamlit search box."""
    if not query or len(query) < 2:
        return []

    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    params = {"query": query, "language": "en-US"}
    try:
        response = requests.get(SEARCH_URL, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return [(f"{m['title']} ({m.get('release_date', '')[:4]})", m) for m in results if m.get("title")]
    except Exception:
        return []
    return []