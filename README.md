# CineMatch — Streamlit Movie Recommendation System

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses the same GitHub-hosted `movies.csv` and `ratings.csv` sources as the project notebook.

## Pages

- Home
- Personalized Recommendations
- Model Evaluation
- Data Analysis
- About

The recommendation pipeline follows the project's notebook: collaborative filtering, minimum-rating filtering, shrinkage, hybrid scoring with alpha=0.6, and cold-start popularity fallback.
