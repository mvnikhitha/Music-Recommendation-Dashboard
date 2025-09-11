import os
import sys
from src import feature_extraction, train
from src.recommender import MusicRecommender

SAVED_ASSETS_PATH = "saved_assets"

def run_feature_extraction():
    print("\n[1] Running feature extraction...")
    feature_extraction.main()

def run_training():
    print("\n[2] Running training...")
    if not os.path.exists(os.path.join(SAVED_ASSETS_PATH, "features.npy")):
        print("⚠️ No features found. Run feature extraction first!")
    else:
        train.main()

def run_recommender():
    print("\n[3] Running recommender...")
    if not os.path.exists(os.path.join(SAVED_ASSETS_PATH, "scaled_features.npy")):
        print("⚠️ No trained model found. Run training first!")
        return

    recommender = MusicRecommender()
    filename = input("Enter a song filename (e.g., rock.00050.wav): ").strip()
    try:
        recommendations = recommender.get_recommendations(filename)
        print(f"\n--- Recommendations for '{filename}' ---")
        for idx, rec in enumerate(recommendations, 1):
            print(f"{idx}. {rec}")
    except Exception as e:
        print(f"❌ Error while getting recommendations: {e}")

def main():
    while True:
        print("\n🎵 Music Recommendation System 🎵")
        print("--------------------------------")
        print("1. Extract Features")
        print("2. Train Model")
        print("3. Get Recommendations")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_feature_extraction()
        elif choice == "2":
            run_training()
        elif choice == "3":
            run_recommender()
        elif choice == "4":
            print("Goodbye 👋")
            sys.exit(0)
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
