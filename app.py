import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import random
from collections import Counter

SAVED_ASSETS_PATH = "saved_assets"

class MusicRecommender:
    def __init__(self):
        self.scaled_features = np.load(os.path.join(SAVED_ASSETS_PATH, "scaled_features.npy"))
        self.filenames = joblib.load(os.path.join(SAVED_ASSETS_PATH, "filenames.pkl"))
        self.kmeans = joblib.load(os.path.join(SAVED_ASSETS_PATH, "kmeans_model.joblib"))
        self.scaler = joblib.load(os.path.join(SAVED_ASSETS_PATH, "scaler.joblib"))

    def get_recommendations_by_genre(self, genre, top_n=5):
        # Pick all songs from selected genre
        genre_files = [f for f in self.filenames if f.startswith(genre)]
        if not genre_files:
            raise ValueError(f"No songs found for genre '{genre}'")

        # Pick a random reference song from that genre
        ref_song = random.choice(genre_files)
        idx = self.filenames.index(ref_song)
        input_vector = self.scaled_features[idx].reshape(1, -1)

        # Compute similarity with ALL songs (not just genre)
        similarities = cosine_similarity(input_vector, self.scaled_features)[0]

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != idx]  # exclude self

        # Take top_n songs across genres
        recommendations = [(self.filenames[i], similarities[i]) for i in sorted_indices[:top_n]]
        return recommendations, ref_song, idx, sorted_indices[:top_n]


# ------------------- STREAMLIT DASHBOARD -------------------

st.set_page_config(layout="wide")
st.title("🎵 Music Recommendation Dashboard")

# Load recommender
recommender = MusicRecommender()

# Extract genres from filenames
genres = sorted(set([f.split(".")[0] for f in recommender.filenames]))

# Sidebar controls
st.sidebar.header("Controls")
selected_genre = st.sidebar.selectbox("Choose a genre:", genres)
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

def get_cluster_genre_labels(filenames, labels):
    cluster_genre_map = {}
    for cluster_id in sorted(set(labels)):
        cluster_files = [f for f, l in zip(filenames, labels) if l == cluster_id]
        genres = [f.split(".")[0] for f in cluster_files]
        most_common_genre, count = Counter(genres).most_common(1)[0]
        cluster_genre_map[cluster_id] = f"{most_common_genre} ({count} songs)"
    return cluster_genre_map


if st.sidebar.button("Generate Recommendations"):
    recs, ref_song, song_idx, rec_indices = recommender.get_recommendations_by_genre(selected_genre, top_n=top_n)

    # Show recommendations
    st.subheader(f"--- Recommendations for genre '{selected_genre}' ---")
    st.write(f"Reference song used: **{ref_song}**")
    for idx, (song, score) in enumerate(recs, 1):
        st.write(f"{idx}. {song} (similarity: {score:.4f})")

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(recommender.scaled_features)

    reduced_features = reduced_features - reduced_features.min(axis=0)

    df = pd.DataFrame({
        "x": reduced_features[:, 0],
        "y": reduced_features[:, 1],
        "filename": recommender.filenames,
        "cluster": recommender.kmeans.labels_
    })

    # Map cluster IDs → dominant genres
    cluster_genre_map = get_cluster_genre_labels(recommender.filenames, recommender.kmeans.labels_)
    df["cluster_name"] = df["cluster"].map(cluster_genre_map)

    # Mark selected + recommendations
    df["highlight"] = "Other"
    df.loc[song_idx, "highlight"] = "Reference Song"
    for i in rec_indices:
        df.loc[i, "highlight"] = "Recommendation"

    # Plot clusters with genres instead of numbers
    fig = px.scatter(
        df, x="x", y="y",
        color="cluster_name",  
        hover_data=["filename", "cluster_name"],
        symbol="highlight",
        size=df["highlight"].apply(lambda x: 12 if x != "Other" else 6),
        title="Pattern Recognition: Clusters of Songs by Dominant Genre"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution chart (with genre names)
    st.subheader("Cluster Distribution by Dominant Genre")
    cluster_counts = pd.Series(recommender.kmeans.labels_).value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.map(lambda x: cluster_genre_map[x])
    st.bar_chart(cluster_counts)
