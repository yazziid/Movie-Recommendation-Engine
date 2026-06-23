import pandas as pd
import psycopg2
from sqlalchemy import create_engine

engine = create_engine("postgresql://db_user:db_password@localhost:5432/movieRec")

def makingConn():
    conn = psycopg2.connect(
        host="localhost",
        dbname="movieRec",
        user="db_user",
        password="db_password",
        port=5432
    )
    return conn

connection = makingConn()
cur = connection.cursor()


## Links Table
cur.execute("""
    CREATE TABLE IF NOT EXISTS links (
        movieId INT PRIMARY KEY,
        imdbId INT,
        movieLensId INT
    );
    """)
    
with open("data/links_cleaned_for_sql.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY links (movieId, imdbId, movieLensId)
        FROM STDIN WITH CSV HEADER
    """, f)


## Movies Table
cur.execute(""" CREATE TABLE IF NOT EXISTS movies (
    movieId INT PRIMARY KEY,
    original_title TEXT,
    overview TEXT,
    popularity FLOAT,
    production_companies TEXT,
    release_date DATE,
    revenue FLOAT,
    budget FLOAT,
    runtime FLOAT,
    vote_average FLOAT,
    vote_count FLOAT,
    country_CA INT,
    country_DE INT,
    country_ES INT,
    country_FR INT,
    country_GB INT,
    country_HK INT,
    country_IN INT,
    country_IT INT,
    country_JP INT,
    country_US INT,
    lang_de BOOLEAN,
    lang_en BOOLEAN,
    lang_es BOOLEAN,
    lang_fr BOOLEAN,
    lang_hi BOOLEAN,
    lang_it BOOLEAN,
    lang_ja BOOLEAN,
    lang_ko BOOLEAN,
    lang_other BOOLEAN,
    lang_ru BOOLEAN,
    lang_zh BOOLEAN,
    Action INT,
    Adventure INT,
    Animation INT,
    Comedy INT,
    Crime INT,
    Documentary INT,
    Drama INT,
    Family INT,
    Fantasy INT,
    History INT,
    Horror INT,
    Music INT,
    Mystery INT,
    Romance INT,
    Science_Fiction INT,
    TV_Movie INT,
    Thriller INT,
    War INT,
    Western INT,
    NumberLanguages INT,
    FOREIGN KEY (movieID) REFERENCES links(movieId)
    )"""
    )


with open("data/movies_cleaned_for_sql.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY movies (budget, movieId, original_title, overview, popularity, production_companies, release_date, revenue, runtime, vote_average, vote_count, country_CA,
        country_DE, country_ES, country_FR, country_GB, country_HK, country_IN, country_IT, country_JP, country_US, lang_de, lang_en, lang_es, lang_fr, lang_hi, lang_it,
        lang_ja, lang_ko, lang_other, lang_ru, lang_zh, Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery,
        Romance, Science_Fiction, TV_Movie, Thriller, War, Western, NumberLanguages)
        FROM STDIN WITH CSV HEADER
    """, f)

# Casts table
cur.execute("""
    CREATE TABLE IF NOT EXISTS movie_cast (
    actorId INT,
    movieId INT,
    name TEXT,
    gender TEXT,
    department TEXT,
    popularity FLOAT,
    cast_id INT,
    credit_id TEXT,
    cast_order INT,
    PRIMARY KEY (actorId, movieId),
    FOREIGN KEY (movieId) REFERENCES links(movieId)
    );         
        """)


with open("data/cast_data_for_sql.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY movie_cast (actorId, movieId, name, gender, department, popularity, cast_id , credit_id, cast_order)
        FROM STDIN WITH CSV HEADER
    """, f)


# Tags Table
cur.execute("""
    CREATE TABLE IF NOT EXISTS AggTags (
    movieLensId INT PRIMARY KEY,
    tag TEXT,
    FOREIGN KEY (movieLensId) REFERENCES links(movieLensId)
    );         
        """)


with open("data/tags_cleaned_for_sql.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY AggTags (movieLensId, tag)
        FROM STDIN WITH CSV HEADER
    """, f)



# Ratings
cur.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        userId INT,
        movieLensId INT,
        rating FLOAT,
        timestamp BIGINT,
        PRIMARY KEY (userId, movieLensId, timestamp),
        FOREIGN KEY (movieLensId) REFERENCES links(movieLensId)
    );
""")


with open("data/rating_cleaned_for_sql.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY ratings (userId, movieLensId, rating, timestamp)
        FROM STDIN WITH CSV HEADER
    """, f)
    

cur.execute('ALTER TABLE ratings ADD COLUMN IF NOT EXISTS "movieid" INT;')
cur.execute('ALTER TABLE "aggtags" ADD COLUMN IF NOT EXISTS "movieid" INT;')

cur.execute("""
    UPDATE ratings r
    SET "movieid" = l."movieid"
    FROM links l
    WHERE r."movielensid" = l."movielensid"
      AND r."movieid" IS NULL;
""")

cur.execute("""
    UPDATE "aggtags" a
    SET "movieid" = l."movieid"
    FROM links l
    WHERE a."movielensid" = l."movielensid"
      AND a."movieid" IS NULL;
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS user_watchlist (
        movie_id INT PRIMARY KEY,
        title TEXT,
        year TEXT,
        rating INT,
        poster_path TEXT,
        FOREIGN KEY (movie_id) REFERENCES links(movieId)
    );
""")



connection.commit()

cur.close()
connection.close()

