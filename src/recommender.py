import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances
SAVED_ASSETS_PATH = "saved_assets"

class MusicRecommender:
    def __init__(self):
        # Load trained assets
        self.scaled_features = np.load(os.path.join(SAVED_ASSETS_PATH, "scaled_features.npy"))
        self.filenames = joblib.load(os.path.join(SAVED_ASSETS_PATH, "filenames.pkl"))
        self.kmeans = joblib.load(os.path.join(SAVED_ASSETS_PATH, "kmeans_model.joblib"))
        self.scaler = joblib.load(os.path.join(SAVED_ASSETS_PATH, "scaler.joblib"))
        print("Recommender initialized successfully.")

    def get_recommendations(self, filename, top_n=5):
        """Return top_n recommendations for the given filename, ranked by distance."""
        if filename not in self.filenames:
            raise ValueError(f"❌ File '{filename}' not found in dataset.")

        # Find index of the input song
        idx = self.filenames.index(filename)
        input_vector = self.scaled_features[idx].reshape(1, -1)

        # Predict its cluster
        song_cluster = self.kmeans.predict(input_vector)[0]

        # Get all indices from the same cluster (except itself)
        cluster_indices = [i for i, label in enumerate(self.kmeans.labels_) 
                        if label == song_cluster and i != idx]

        # Compute distances within the cluster
        cluster_vectors = self.scaled_features[cluster_indices]
        distances = euclidean_distances(input_vector, cluster_vectors)[0]

        # Sort by distance (closest first)
        sorted_indices = np.argsort(distances)

        # Pick top_n
        recommendations = [self.filenames[cluster_indices[i]] for i in sorted_indices[:top_n]]

        return recommendations

