import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

# Set up Streamlit app
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Get top movie recommendations based on your favorite movie!")

# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv(os.path.join('data', 'movies.csv'))
    ratings = pd.read_csv(os.path.join('data', 'ratings.csv'))
    return movies, ratings

movies, ratings = load_data()

# Merge movies and ratings
movie_data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table of user ratings (userId x movieId)
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Calculate cosine similarity between movies
similarity = cosine_similarity(user_movie_matrix.T)

# Helper: get similar movies
def recommend_movies(movie_name, top_n=5):
    if movie_name not in user_movie_matrix.columns:
        return []

    movie_idx = user_movie_matrix.columns.get_loc(movie_name)
    sim_scores = list(enumerate(similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    recommended = [(user_movie_matrix.columns[i], score) for i, score in sim_scores]
    return recommended

# UI: Select a movie
selected_movie = st.selectbox("Choose a movie you like:", sorted(user_movie_matrix.columns))

if st.button("Recommend Movies"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("📽️ Top Recommended Movies:")
    for title, score in recommendations:
        st.markdown(f"**🎞️ {title}** — Similarity Score: `{score:.2f}`")

# Show raw data (optional)
with st.expander("📄 See Raw Movie Data"):
    st.write(movies.head())

with st.expander("📊 See Ratings Data"):
    st.write(ratings.head())
