import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. CREATE / LOAD DATASET
# -------------------------------

data = {
    "title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avengers",
        "Iron Man",
        "Titanic",
        "The Notebook",
        "Joker",
        "Doctor Strange",
        "Batman Begins"
    ],
    "genre": [
        "Action Sci-Fi Thriller",
        "Sci-Fi Drama Adventure",
        "Action Crime Drama",
        "Action Sci-Fi Fantasy",
        "Action Sci-Fi",
        "Romance Drama",
        "Romance Drama",
        "Crime Drama Thriller",
        "Action Fantasy Sci-Fi",
        "Action Crime Drama"
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. FEATURE EXTRACTION (TF-IDF)
# -------------------------------

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["genre"])

# -------------------------------
# 3. COSINE SIMILARITY
# -------------------------------

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# 4. INDEX MAPPING
# -------------------------------

indices = pd.Series(df.index, index=df["title"])

# -------------------------------
# 5. RECOMMENDATION FUNCTION
# -------------------------------

def recommend_movies(movie_title, top_n=5):
    if movie_title not in indices:
        return "❌ Movie not found in database."

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices]

# -------------------------------
# 6. USER INPUT
# -------------------------------

print("🎬 Movie Recommendation System")
print("-------------------------------")
print("Available movies:")
print(df["title"].to_string(index=False))

movie_name = input("\nEnter a movie name: ")

recommendations = recommend_movies(movie_name)

print("\nRecommended movies:")
print(recommendations)
