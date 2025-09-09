# 🎵 Music Recommendation Dashboard  

This project is a **Music Recommendation System** built with **Python, scikit-learn, and Streamlit**.  
It uses **audio feature extraction, K-Means clustering, and cosine similarity** to recommend similar songs or songs from specific genres.  
The system also provides interactive visualizations with **Plotly**.

---

## 🚀 Features
- Extract features from songs and cluster them into genres using **K-Means**.
- Get **song-based recommendations** or **genre-based recommendations**.
- Interactive **Streamlit Dashboard** with:
  - Song/genre selection
  - Recommendation list with similarity scores
  - PCA-based 2D cluster visualization
  - Cluster distribution by **dominant genre**

---

## 🛠️ Tech Stack
- **Python 3.12**
- **Streamlit**
- **scikit-learn**
- **NumPy / Pandas**
- **Plotly**

---

## 📊 Dashboard Preview
- **Scatter Plot** → Visualizes clusters of songs (PCA reduced to 2D).
- **Bar Chart** → Shows how many songs belong to each cluster, labeled by dominant genre.

---

## ⚙️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/mvnikhitha/Music-Recommendation-Dashboard.git
   cd Music-Recommendation-Dashboard
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the app:
   ```bash
   streamlit run app.py

## 📂 **Project Structure**
```bash 
    ├── app.py # Streamlit dashboard
    ├── src/ # Feature extraction & training scripts
    ├── saved_assets/ # Saved models & scalers (ignored in git)
    ├── requirements.txt # Dependencies
    ├── README.md # Project documentation
    └── .gitignore
```
## 🔮 Future Improvements
- Add audio playback inside the dashboard.  
- Experiment with deep learning embeddings for improved recommendations.  
- Deploy on Streamlit Cloud / Hugging Face Spaces for easy access.

  Then push it to
  ```bash
  git add README.md
  git commit -m "Added README.md"
  git push origin main

  ```
## MIT License

Copyright (c) 2025 mvnikhitha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies


