<img width="1887" height="842" alt="Screenshot 2026-10-01 151434" src="https://github.com/user-attachments/assets/f3a0a4d8-71b9-4b14-bde1-02584586905f" />
<img width="1867" height="850" alt="Screenshot 2026-10-01 151820" src="https://github.com/user-attachments/assets/5014aca8-ea30-4d9a-8ec8-e4d48f430e3a" />
<img width="1866" height="836" alt="Screenshot 2026-10-01 151849" src="https://github.com/user-attachments/assets/3142a636-37fc-4365-bb46-d1a54a99120a" />
<img width="1797" height="833" alt="Screenshot 2026-10-01 151933" src="https://github.com/user-attachments/assets/329d4d63-2274-4d9e-b923-304eeda9582b" />


# 🎬 Personalised Movie Recommendation System

A personalised movie recommendation system built using **Collaborative Filtering** and deployed through an interactive **Streamlit** application.

The system analyses historical user–movie ratings to recommend movies that a user is likely to enjoy. It combines collaborative-filtering scores with popularity to improve the final Top-N recommendations.

---

## 📌 Project Overview

With thousands of movies available, users often find it difficult to discover movies relevant to their interests.

This project addresses this problem by learning from existing user ratings and identifying patterns between:

- Users with similar rating behaviour
- Movies with similar rating patterns
- Movie popularity

The final system generates **personalised Top-N movie recommendations** for a selected user.

---

## 🎯 Objectives

- Build a personalised movie recommendation system.
- Apply **user-based and item-based collaborative filtering**.
- Analyse the sparsity of the user–movie rating matrix.
- Predict ratings for unseen movies.
- Improve recommendations using popularity and score shrinkage.
- Handle the **cold-start problem** for users with very few ratings.
- Evaluate the recommendation system using **RMSE, MAE and Precision@10**.
- Provide an interactive interface using **Streamlit**.

---

## 🧠 Recommendation Approach

The project follows this pipeline:

```text
Movie & Rating Data
        ↓
Data Preprocessing
        ↓
User–Movie Rating Matrix
        ↓
Mean-Centred Ratings
        ↓
Cosine Similarity
        ↓
Collaborative Filtering
   ↙              ↘
Item-Based CF    User-Based CF
        ↓
Score Shrinkage / Damping
        ↓
Popularity Signal
        ↓
Hybrid Recommendation
        ↓
Top-N Movies
        ↓
Streamlit Application
