import pandas as pd
from sqlalchemy import create_engine, text
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset

class TwoTowerDataset(Dataset):
    def __init__(self, interactions_df, userid2idx, movieid2idx):

        self.interactions = [
            (userid2idx[u], movieid2idx[m])
            for u, m in interactions_df.values
            if u in userid2idx and m in movieid2idx
        ]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, i = self.interactions[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
        )



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#engine = create_engine("postgresql+psycopg2://db_user:db_password@localhost:5402/movieRec")

engine = create_engine("postgresql+psycopg2://db_user:db_password@movieRec_postgres:5432/movieRec")

def load_table(table_name):
    q = text(f'SELECT * FROM "{table_name}"')
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn)

links_df   = load_table("links")
movies_df  = load_table("movies")
cast_df    = load_table("movie_cast")
ratings_df = load_table("ratings")
print("Database successfully loaded")

links_df = links_df.rename(columns={"tmdbId": "movieid"})
links_df["movieid"] = pd.to_numeric(links_df["movieid"], errors="coerce")

movies_df["movieid"] = pd.to_numeric(movies_df["movieid"], errors="coerce")

covered_tmdb = set(movies_df["movieid"].dropna().astype(int))
covered_movielensids = set(
    links_df.loc[links_df["movieid"].isin(covered_tmdb), "movielensid"]
)


# Train test split
ratings_cov = ratings_df[ratings_df["movielensid"].isin(covered_movielensids)].copy()

ratings_cov = ratings_cov.sample(frac=1, random_state=42).reset_index(drop=True)

msk = np.random.rand(len(ratings_cov)) < 0.8

train_df = ratings_cov[msk]
test_df  = ratings_cov[~msk]


def index_mapping(train_df):
    user_ids = train_df["userid"].unique()
    movie_ids = train_df["movielensid"].unique()

    userid2idx = {u:i for i,u in enumerate(user_ids)}
    movieid2idx = {m:i for i,m in enumerate(movie_ids)}

    idx2movieid = {i:m for m,i in movieid2idx.items()}

    n_users = len(userid2idx)
    n_items = len(movieid2idx)
    
    return n_users, n_items, idx2movieid, userid2idx, movieid2idx

n_users, n_items, idx2movieid, userid2idx, movieid2idx = index_mapping(train_df)

def build_metadata_features(movies_df, covered_tmdb, movieid2idx):
    movies_df = movies_df[movies_df["movieid"].isin(covered_tmdb)]

    movie_meta = links_df.merge(movies_df, on="movieid", how="inner")
    movie_meta = movie_meta[movie_meta["movielensid"].isin(movieid2idx)]
    movie_meta = movie_meta.drop_duplicates("movielensid")
    movie_meta = movie_meta.set_index("movielensid")
    movie_meta = movie_meta.reindex(movieid2idx.keys())

    feature_cols = ["runtime", "budget", "revenue"]

    X_item = movie_meta[feature_cols].fillna(0).values
    X_item = torch.tensor(X_item, dtype=torch.float32)
    return X_item

X_item = build_metadata_features(movies_df, covered_tmdb, movieid2idx)

def top_k_actor_per_movie(cast_df = cast_df, idx2movieid = idx2movieid, K = 5):
    cast_small = cast_df.sort_values(["movieid","cast_order"]).groupby("movieid").head(K)

    actor_ids = cast_small["actorid"].unique()
    actor2idx = {a:i+1 for i,a in enumerate(actor_ids)}
    n_actors = len(actor2idx)+1

    tmdb_to_actor_idxs = {}

    for mid,g in cast_small.groupby("movieid"):
        tmdb_to_actor_idxs[mid] = [actor2idx[a] for a in g["actorid"] if a in actor2idx]

    A_item = np.zeros((n_items, K),dtype=np.int64)
    movielens_to_tmdb = dict(zip(links_df.movielensid,links_df.movieid))

    for item_idx,movieid in idx2movieid.items():
        tmdb = movielens_to_tmdb.get(movieid)
        if tmdb is None: continue
        actors = tmdb_to_actor_idxs.get(tmdb,[])[:K]
        if actors: A_item[item_idx,:len(actors)] = actors
        
    A_item = torch.tensor(A_item, dtype=torch.long)
    return A_item, n_actors, K

if 'name' == '__main__':
    n_users, n_items, idx2movieid, userid2idx, movieid2idx = index_mapping(train_df)
    A_item, n_actors, K = top_k_actor_per_movie() 

