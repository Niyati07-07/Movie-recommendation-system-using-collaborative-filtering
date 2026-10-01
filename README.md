🎬 Personalized Movie Recommendation System (Collaborative Filtering)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

A movie recommendation system built on the MovieLens ratings dataset. It uses item-based and user-based collaborative filtering with cosine similarity to predict how a user would rate movies they have not watched, then recommends the top N titles. It includes a cold-start fallback for new users, a reason for each recommendation, and an interactive Gradio interface.

Objective

Streaming platforms have very large catalogues, and users often struggle to find movies that match their taste, which reduces engagement and satisfaction. This project uses historical ratings to find users with similar preferences and movies with similar rating patterns, predicts ratings for unseen movies, and recommends the most relevant titles to each user. Data sparsity and the cold-start problem are addressed, and the models are evaluated with RMSE, MAE and Precision@10.

Dataset
MovieLens (latest-small) from GroupLens Research, University of Minnesota.
105,339 ratings, 668 users and 10,325 rated movies (0.5 to 5 stars).
Files used: ratings.csv (userId, movieId, rating, timestamp) and movies.csv (movieId, title, genres).
Loaded in the notebook from a GitHub copy of the dataset. Original source: https://grouplens.org/datasets/movielens/
Approach
Step	What was done
1. Load and clean	Loaded ratings and movies; checked missing values, duplicates and rating range (no problems found); merged on movieId
2. EDA	Rating distribution, ratings per user and per movie (long tail), most and highest rated movies, ratings by genre
3. Sparsity	Matrix is 98.47% empty; filtered users with fewer than 20 ratings and movies with fewer than 5 ratings, reducing sparsity to 96.35%
4. Split	80/20 random split of ratings (75,297 train, 18,824 test)
5. Matrix	User × movie matrix (668 × 3,854) built from training data only; ratings mean-centred per user
6. Similarity	Cosine similarity between users and between movies
7. Prediction	Item-based and user-based predictors using the k nearest neighbours
8. Evaluation	RMSE and MAE against baselines; Precision@10 against a popularity baseline
9. Tuning	Chose k on one slice of the test set, reported on a different slice
10. Recommend	Top-N function using a blend of the personalised score and popularity
11. Cold start	Popular, well-rated movies for new or unknown users
12. Explain	Each recommendation names a movie the user liked that is most similar to it
13. UI	Gradio app with a recommendation tab and a similar-movies tab
Results

Rating prediction (3,000 held-out ratings, k = 20):

Model	RMSE	MAE
Global average (baseline)	1.018	0.824
User average (baseline)	0.912	0.708
Item-based CF	0.826	0.631
User-based CF	0.840	0.638

Top-10 ranking (Precision@10 on 297 users not used for tuning):

Method	Precision@10
Popularity	0.139
Item-based CF (min 50 ratings)	0.110
Hybrid (CF + popularity, alpha 0.6)	0.146

Findings:

Collaborative filtering predicts ratings better than both baselines.
For top-10 ranking, plain CF scored below a popularity baseline. A blend of the two matches popularity within statistical noise (difference +0.007, standard error 0.005) and clearly improves on CF alone.
On MovieLens, Precision@10 rewards popular movies, because only movies a user chose to rate can count as hits.
Requiring a minimum number of ratings per recommended movie and damping predictions based on few neighbours raised Precision@10 from 0.024 to 0.110.
Limitations
Sparsity: the rating matrix stays about 96% empty after filtering.
Cold start: new users get a popularity-based list until they have rated some movies.
Popularity bias: recommendations lean toward well-known, well-rated movies, which limits discovery of niche titles.
Filtered test set: only movies with at least 5 ratings are evaluated, so errors on rarely rated movies are not measured.
Next steps: matrix factorization (SVD/ALS), hybrid models using genres, and implicit feedback.
Tech stack

Python, pandas, NumPy, scikit-learn (cosine similarity), matplotlib, seaborn, Gradio, Google Colab.

How to run
Open the notebook in Google Colab (or Jupyter).
Run all cells in order (Runtime → Run all). The data loads from GitHub URLs, so no downloads are needed.
The last cell launches the Gradio interface:
Recommend for a user: enter a user ID (for example 1) or a new ID (for example 999999 to see the cold-start list).
Similar movies: pick a movie and see which movies the same users rated similarly.
Project structure
├── Movie_Recommendation_System.ipynb   # full workflow: EDA, model, evaluation, UI
├── ratings.csv                         # MovieLens ratings (if included)
├── movies.csv                          # MovieLens movies (if included)
└── README.md

Edit the file names above to match your repository.

Project plan

Delivered in six sprints:

Sprint	Focus
SID_1	Requirements, objective, success metrics, dataset selection
SID_2	Data loading, cleaning, EDA, sparsity check and filtering
SID_3	Train/test split, matrix, similarity, predictors
SID_4	RMSE, Precision@10, baseline comparison, tuning, final model
SID_5	Top-N recommendations, cold start, explanations
SID_6	Report, PDF and demo


