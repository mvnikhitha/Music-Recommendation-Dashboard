# src/feature_extraction.py

import os
import librosa
import numpy as np
import pickle

DATASET_PATH = "Data/genres_original"
SAVED_ASSETS_PATH = "saved_assets"

def extract_features(file_path):
    """Extracts a feature vector from an audio file."""
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        # MFCCs (mean and variance)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        
        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # Spectral contrast
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Combine all features
        features = np.hstack([mfccs_mean, mfccs_var, chroma, spectral_contrast, tempo, zcr])
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Processes all audio files and saves the features."""
    all_features = []
    all_filenames = []

    if not os.path.exists(SAVED_ASSETS_PATH):
        os.makedirs(SAVED_ASSETS_PATH)

    for genre_folder in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path, filename)
                    features = extract_features(file_path)
                    if features is not None:
                        all_features.append(features)
                        all_filenames.append(filename)
                        print(f"Processed: {filename}")

    # Save the extracted features and filenames
    np.save(os.path.join(SAVED_ASSETS_PATH, 'features.npy'), np.array(all_features))
    with open(os.path.join(SAVED_ASSETS_PATH, 'filenames.pkl'), 'wb') as f:
        pickle.dump(all_filenames, f)
        
    print("\nFeature extraction complete.")
    print(f"Saved features and filenames to '{SAVED_ASSETS_PATH}' directory.")

if __name__ == "__main__":
    main()
