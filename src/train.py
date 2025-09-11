# src/train.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

SAVED_ASSETS_PATH = "saved_assets"

def main():
    """Loads features, scales them, and trains a K-Means model."""
    # Load the extracted features
    features = np.load(os.path.join(SAVED_ASSETS_PATH, 'features.npy'))

    # 1. Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Save the scaler and the scaled features
    joblib.dump(scaler, os.path.join(SAVED_ASSETS_PATH, 'scaler.joblib'))
    np.save(os.path.join(SAVED_ASSETS_PATH, 'scaled_features.npy'), scaled_features)
    print("Features scaled and scaler saved.")

    # 2. K-Means Clustering
    num_clusters = 10  # As there are 10 genres
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Save the trained K-Means model
    joblib.dump(kmeans, os.path.join(SAVED_ASSETS_PATH, 'kmeans_model.joblib'))
    print(f"K-Means model with {num_clusters} clusters trained and saved.")

if __name__ == "__main__":
    main()
