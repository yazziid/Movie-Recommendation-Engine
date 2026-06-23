import streamlit as st
from streamlit_searchbox import st_searchbox

from data_service import load_ui_movies, load_ratings_data, save_movie_rating, delete_movie_rating
from tmdb_service import get_poster_url, search_tmdb_suggestions
from api_service import get_api_recommendations, extract_recommendation

st.set_page_config(page_title="Movie Recommender", layout="wide")

movies_df = load_ui_movies()
df_ratings = load_ratings_data()

st.title(" Movie Rating App")

# --- Search Section ---
selected_movie = st_searchbox(
    search_tmdb_suggestions,
    placeholder="Search and select a movie...",
    key="movie_search_box"
)

if selected_movie:
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster_path = selected_movie.get('poster_path')
        if poster_path:
            st.image(f"https://image.tmdb.org/t/p/w300{poster_path}", use_container_width=True)
        else:
            st.info("No poster available.")
            
    with col2:
        st.write(f"### Rate: {selected_movie['title']}")
        st.caption(f"Released: {selected_movie.get('release_date', 'N/A')}")
        st.write(selected_movie.get('overview', 'No overview available.'))
        
        rating = st.slider("Rating (1-5)", 1, 5, 3)
        if st.button("Save/Update Selection", use_container_width=True):
            title = save_movie_rating(selected_movie, rating)
            st.success(f"Updated {title}!")
            st.rerun()

# --- History Section ---
st.divider()
st.subheader("Your Ratings")

if not df_ratings.empty:
    cols = st.columns(4)
    for idx, (original_idx, row) in enumerate(df_ratings.sort_index(ascending=False).iterrows()):
        col = cols[idx % 4]
        with col:
            poster_url = get_poster_url(row.get('poster_path'), row['movie_id'])
            st.image(poster_url, use_container_width=True)
            st.markdown(f"**{row['title']}**")
            
            with st.expander("Details & Edit"):
                st.write(f"**Year:** {row['year']}")
                st.write(f"**Current Rating:** {row['rating']}/5")
                
                new_val = st.number_input("Change Rating", 1, 5, int(row["rating"]), key=f"edit_{row['movie_id']}")
                c1, c2 = st.columns(2)
                
                if c1.button("Update", key=f"upd_{row['movie_id']}"):
                    save_movie_rating({
                        "id": row["movie_id"], 
                        "title": row["title"], 
                        "poster_path": row.get("poster_path", "")
                    }, new_val)
                    st.rerun()
                    
                if c2.button("Delete", key=f"del_{row['movie_id']}"):
                    delete_movie_rating(row["movie_id"])
                    st.rerun()
else:
    st.info("Your watchlist is currently empty.")

# --- AI Recommendations Section ---
st.divider()
st.subheader(" Recommended for You")

num_ratings = len(df_ratings)
needs_more = num_ratings < 5

if needs_more:
    st.info(f"You have rated {num_ratings}/5 movies. Rate at least {5 - num_ratings} more to unlock AI recommendations!")

if st.button("Generate AI Recommendations", disabled=needs_more):
    if not movies_df.empty:
        with st.spinner("Analyzing your tastes via API..."):
            rec_ids = get_api_recommendations(df_ratings, movies_df)
            
            if not rec_ids:
                st.warning("No recommendations returned.")
            else:
                final_table = extract_recommendation(rec_ids, movies_df)
                st.write("### Here are some movies you might like:")
                
                rec_cols = st.columns(4)
                for idx, row in final_table.iterrows():
                    with rec_cols[idx % 4]:
                        poster_url = get_poster_url("", row['movieid'])
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{row['original_title']}**")
                        
                        with st.expander("More Details"):
                            st.write(f"**Released:** {row['release_date']}")
                            st.caption(row['overview'])