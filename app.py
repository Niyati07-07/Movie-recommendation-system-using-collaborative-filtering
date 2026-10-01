
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Personalized Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main { background: #0b1020; }
    .block-container { padding-top: 2rem; max-width: 1250px; }
    .hero {
        padding: 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #111a33, #1d2850);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,.08);
    }
    .hero h1 { margin-bottom: .3rem; font-size: 3rem; }
    .hero p { color: #cbd5e1; font-size: 1.1rem; }
    .metric-card {
        padding: 1.2rem;
        border-radius: 18px;
        background: #111827;
        border: 1px solid rgba(255,255,255,.08);
        text-align: center;
    }
    .movie-card {
        padding: 1.1rem;
        border-radius: 18px;
        background: #111827;
        border: 1px solid rgba(255,255,255,.08);
        min-height: 170px;
        margin-bottom: 1rem;
    }
    .movie-rank { color: #93c5fd; font-weight: 700; }
    .movie-title { font-size: 1.08rem; font-weight: 700; margin: .4rem 0; }
    .movie-meta { color: #cbd5e1; font-size: .9rem; }
    .reason { color: #a5b4fc; font-size: .9rem; margin-top: .65rem; }
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


MOVIES_URL = "https://github.com/Sravyatogarla/Movie-recommendation-system/raw/refs/heads/main/movies.csv"
RATINGS_URL = "https://github.com/Sravyatogarla/Movie-recommendation-system/raw/refs/heads/main/ratings.csv"


@st.cache_data
def load_data():
    movies_df = pd.read_csv(MOVIES_URL)
    ratings_df = pd.read_csv(RATINGS_URL)
    merged_df = pd.merge(ratings_df, movies_df, on="movieId")
    return movies_df, ratings_df, merged_df


@st.cache_resource(show_spinner="Building the collaborative-filtering model...")
def build_model():
    movies_df, ratings_df, merged_df = load_data()

    # Same filtering used in the notebook
    min_user_ratings = 20
    min_movie_ratings = 5

    u = merged_df["userId"].value_counts()
    m = merged_df["movieId"].value_counts()

    filtered_df = merged_df[
        merged_df["userId"].isin(u[u >= min_user_ratings].index)
        & merged_df["movieId"].isin(m[m >= min_movie_ratings].index)
    ].copy()

    # Same 80/20 random split as the notebook
    test_df = filtered_df.sample(frac=0.2, random_state=42)
    train_df = filtered_df.drop(test_df.index)

    # Same user-item matrix
    user_item = train_df.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    user_mean = user_item.mean(axis=1)
    centred = user_item.sub(user_mean, axis=0).fillna(0)

    # Same similarities as the notebook
    user_sim = pd.DataFrame(
        cosine_similarity(centred),
        index=user_item.index,
        columns=user_item.index,
    )

    item_sim = pd.DataFrame(
        cosine_similarity(centred.T),
        index=user_item.columns,
        columns=user_item.columns,
    )

    movie_titles = movies_df.set_index("movieId")["title"]

    movie_stats = train_df.groupby("movieId")["rating"].agg(["mean", "count"])
    movie_counts = user_item.notna().sum()
    pop = train_df["movieId"].value_counts()

    return {
        "movies_df": movies_df,
        "ratings_df": ratings_df,
        "merged_df": merged_df,
        "filtered_df": filtered_df,
        "train_df": train_df,
        "test_df": test_df,
        "user_item": user_item,
        "user_mean": user_mean,
        "centred": centred,
        "user_sim": user_sim,
        "item_sim": item_sim,
        "movie_titles": movie_titles,
        "movie_stats": movie_stats,
        "movie_counts": movie_counts,
        "pop": pop,
    }


def get_model():
    return build_model()


def recommend_scores_v2(model, user, min_ratings=20, shrink=10):
    user_item = model["user_item"]
    item_sim = model["item_sim"]
    centred = model["centred"]
    user_mean = model["user_mean"]
    movie_counts = model["movie_counts"]

    rated_mask = user_item.loc[user].notna()
    S = item_sim.loc[:, rated_mask].values
    r = centred.loc[user, rated_mask].values

    num = S @ r
    den = np.abs(S).sum(axis=1)
    sc = user_mean[user] + np.divide(
        num, den + shrink, out=np.zeros_like(num), where=(den + shrink) > 0
    )

    scores = pd.Series(sc, index=item_sim.index)

    keep = (~rated_mask.values) & (movie_counts.values >= min_ratings)
    return scores[keep]


def recommend_hybrid(model, user, alpha=0.6, min_ratings=20, shrink=10):
    s = recommend_scores_v2(
        model, user, min_ratings=min_ratings, shrink=shrink
    )

    p = np.log1p(model["movie_counts"].reindex(s.index))

    def z(x):
        std = x.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    return alpha * z(s) + (1 - alpha) * z(p)


def popular_movies(model, n=10, min_ratings=50):
    movie_stats = model["movie_stats"]
    movie_titles = model["movie_titles"]

    p = (
        movie_stats[movie_stats["count"] >= min_ratings]
        .sort_values("mean", ascending=False)
        .head(n)
    )

    return pd.DataFrame({
        "movieId": p.index,
        "title": p.index.map(movie_titles),
        "avg_rating": p["mean"].round(2).values,
        "ratings": p["count"].values,
    })


def explain(model, user, movie_id):
    user_item = model["user_item"]
    item_sim = model["item_sim"]
    movie_titles = model["movie_titles"]

    rated = user_item.loc[user].dropna()
    liked = rated[rated >= 4].index

    if len(liked) == 0:
        return "Highly rated by other viewers"

    sims = item_sim.loc[movie_id, liked]
    best = sims.idxmax()

    if sims[best] <= 0:
        return "Highly rated by other viewers"

    return f"Because you rated '{movie_titles[best]}' {rated[best]:.1f} stars"


def recommend_explained(model, user, n=10, alpha=0.6, min_history=5):
    user_item = model["user_item"]
    movie_stats = model["movie_stats"]
    movie_titles = model["movie_titles"]

    if user not in user_item.index or user_item.loc[user].notna().sum() < min_history:
        out = popular_movies(model, n)
        out["reason"] = (
            "Popular with all viewers. Rate a few movies for personal picks."
        )
        return out

    top = recommend_hybrid(
        model, user, alpha=alpha, min_ratings=20, shrink=10
    ).nlargest(n)

    return pd.DataFrame({
        "movieId": top.index,
        "title": top.index.map(movie_titles),
        "score": top.values.round(2),
        "avg_rating": movie_stats.loc[top.index, "mean"].round(2).values,
        "ratings": movie_stats.loc[top.index, "count"].values,
        "reason": [explain(model, user, m) for m in top.index],
    })


@st.cache_data
def calculate_evaluation(_model):
    """
    Uses the same evaluation logic as the notebook:
    - RMSE/MAE on a 3000-row test sample
    - Precision@10 on the evaluation-user slice
    """
    model = _model
    train_df = model["train_df"]
    test_df = model["test_df"]
    user_mean = model["user_mean"]

    # RMSE/MAE
    eval_df = test_df.sample(min(3000, len(test_df)), random_state=42)
    actual = eval_df["rating"].values

    def predict_item(user, movie, k=30):
        user_item = model["user_item"]
        item_sim = model["item_sim"]
        centred = model["centred"]

        if user not in user_item.index or movie not in user_item.columns:
            return train_df["rating"].mean()

        rated = user_item.loc[user].dropna().index.drop(movie, errors="ignore")
        if len(rated) == 0:
            return user_mean[user]

        sims = item_sim.loc[movie, rated].nlargest(k)
        denom = sims.abs().sum()

        if denom == 0:
            return user_mean[user]

        pred = user_mean[user] + (
            sims * centred.loc[user, sims.index]
        ).sum() / denom

        return float(np.clip(pred, 0.5, 5))

    def predict_user(user, movie, k=30):
        user_item = model["user_item"]
        user_sim = model["user_sim"]
        centred = model["centred"]

        if user not in user_item.index or movie not in user_item.columns:
            return train_df["rating"].mean()

        raters = user_item[movie].dropna().index.drop(user, errors="ignore")
        if len(raters) == 0:
            return user_mean[user]

        sims = user_sim.loc[user, raters].nlargest(k)
        denom = sims.abs().sum()

        if denom == 0:
            return user_mean[user]

        pred = user_mean[user] + (
            sims * centred.loc[sims.index, movie]
        ).sum() / denom

        return float(np.clip(pred, 0.5, 5))

    item_preds = np.array([
        predict_item(r.userId, r.movieId) for r in eval_df.itertuples()
    ])

    user_preds = np.array([
        predict_user(r.userId, r.movieId) for r in eval_df.itertuples()
    ])

    global_avg = np.full(len(actual), train_df["rating"].mean())
    user_avg = (
        eval_df["userId"]
        .map(user_mean)
        .fillna(train_df["rating"].mean())
        .values
    )

    def rmse(p):
        return np.sqrt(np.mean((actual - p) ** 2))

    def mae(p):
        return np.mean(np.abs(actual - p))

    results = pd.DataFrame({
        "Model": [
            "Global average",
            "User average",
            "Item-based CF",
            "User-based CF",
        ],
        "RMSE": [
            rmse(global_avg),
            rmse(user_avg),
            rmse(item_preds),
            rmse(user_preds),
        ],
        "MAE": [
            mae(global_avg),
            mae(user_avg),
            mae(item_preds),
            mae(user_preds),
        ],
    }).round(3)

    # Precision@10 on the same evaluation-user slice used in the notebook
    all_users = test_df["userId"].unique()
    eval_users = all_users[100:400]

    def hits_for_user(user):
        liked = set(
            test_df[
                (test_df["userId"] == user) & (test_df["rating"] >= 4)
            ]["movieId"]
        )

        if not liked or user not in model["user_item"].index:
            return None

        top = set(
            recommend_hybrid(
                model, user, alpha=0.6, min_ratings=20, shrink=10
            ).nlargest(10).index
        )

        return len(liked & top) / 10

    hits = [h for u in eval_users if (h := hits_for_user(u)) is not None]
    precision10 = float(np.mean(hits)) if hits else np.nan

    # Sparsity after the project's filtering
    filtered = model["filtered_df"]
    n_users = filtered["userId"].nunique()
    n_movies = filtered["movieId"].nunique()
    sparsity_after = 1 - len(filtered) / (n_users * n_movies)

    return results, precision10, sparsity_after, len(hits)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 CineMatch")
st.sidebar.caption("Personalized Movie Recommendation System")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 Recommendations", "📊 Evaluation", "📈 Data Analysis", "ℹ️ About"],
)

try:
    model = get_model()
except Exception as e:
    st.error("Could not load the dataset/model.")
    st.exception(e)
    st.stop()

# -----------------------------
# Home
# -----------------------------
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>Personalized Movie Recommendation System using Collaborative Filtering</p>
        <p>Discover movies based on rating patterns, similar preferences, and popularity.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="metric-card"><h2>{model["merged_df"]["userId"].nunique():,}</h2><p>Users</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h2>{model["merged_df"]["movieId"].nunique():,}</h2><p>Movies</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h2>{len(model["merged_df"]):,}</h2><p>Ratings</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        sparsity = 1 - len(model["merged_df"]) / (
            model["merged_df"]["userId"].nunique()
            * model["merged_df"]["movieId"].nunique()
        )
        st.markdown(
            f'<div class="metric-card"><h2>{sparsity*100:.2f}%</h2><p>Sparsity</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### How it works")

    a, b, c, d = st.columns(4)
    a.info("**1. Ratings**\n\nHistorical user-movie ratings are used.")
    b.info("**2. Collaborative Filtering**\n\nUser/item rating patterns are compared.")
    c.info("**3. Hybrid Scoring**\n\nPersonalized CF scores are blended with popularity.")
    d.info("**4. Top-N**\n\nThe highest-scoring unseen movies are recommended.")

# -----------------------------
# Recommendations
# -----------------------------
elif page == "🎯 Recommendations":
    st.title("🎯 Personalized Recommendations")
    st.write("Select a user and generate their Top-N movie recommendations.")

    users = sorted(model["user_item"].index.tolist())

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_user = st.selectbox(
            "👤 Select User",
            users,
            format_func=lambda x: f"User {int(x)}",
        )

    with col2:
        n = st.slider("Number of recommendations", 5, 20, 10)

    history_count = int(model["user_item"].loc[selected_user].notna().sum())
    st.caption(f"User {int(selected_user)} has {history_count} ratings in the training data.")

    if history_count < 5:
        st.warning(
            "This user has fewer than 5 ratings. The system will use the "
            "cold-start popularity fallback."
        )

    if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating personalized recommendations..."):
            recs = recommend_explained(model, selected_user, n=n)

        st.subheader("Recommended Movies")

        for i, row in recs.reset_index(drop=True).iterrows():
            score_text = (
                f"Personalized score: {row['score']:.2f}"
                if "score" in row
                else "Popularity-based recommendation"
            )

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-rank">#{i+1}</div>
                    <div class="movie-title">🎬 {row['title']}</div>
                    <div class="movie-meta">
                        ⭐ Average rating: {row['avg_rating']:.2f}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        👥 Ratings: {int(row['ratings'])}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        {score_text}
                    </div>
                    <div class="reason">💡 {row['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------
# Evaluation
# -----------------------------
elif page == "📊 Evaluation":
    st.title("📊 Model Evaluation")
    st.write(
        "Evaluation follows the methodology used in the project notebook."
    )

    with st.spinner("Calculating evaluation metrics..."):
        results, precision10, sparsity_after, users_evaluated = calculate_evaluation(model)

    item_rmse = float(
        results.loc[results["Model"] == "Item-based CF", "RMSE"].iloc[0]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Item-based CF RMSE", f"{item_rmse:.3f}")
    with c2:
        st.metric("Precision@10", f"{precision10 * 100:.1f}%")
    with c3:
        st.metric("Filtered-data sparsity", f"{sparsity_after * 100:.2f}%")

    st.subheader("RMSE / MAE Comparison")
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.subheader("What the metrics mean")
    st.info(
        "**RMSE** measures the error between predicted and actual ratings. "
        "Lower values indicate smaller prediction errors.\n\n"
        "**Precision@10** measures the proportion of relevant movies among "
        "the first 10 recommendations. In this project, a test rating of "
        "4 or higher is treated as relevant."
    )

    st.caption(
        f"Precision@10 was evaluated on {users_evaluated} users in the "
        "evaluation slice used by the notebook."
    )

# -----------------------------
# Data Analysis
# -----------------------------
elif page == "📈 Data Analysis":
    st.title("📈 Dataset Analysis")

    merged = model["merged_df"]
    filtered = model["filtered_df"]

    c1, c2, c3 = st.columns(3)

    c1.metric("Original ratings", f"{len(merged):,}")
    c2.metric("Filtered ratings", f"{len(filtered):,}")
    c3.metric("Average rating", f"{merged['rating'].mean():.2f}")

    st.subheader("Rating Distribution")
    rating_counts = merged["rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

    st.subheader("Most Rated Movies")
    most_rated = (
        merged.groupby("title")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .rename("ratings")
    )
    st.bar_chart(most_rated)

    st.subheader("Top Rated Movies — minimum 50 ratings")
    stats = merged.groupby("title")["rating"].agg(["mean", "count"])
    top_rated = (
        stats[stats["count"] >= 50]
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
    )
    top_rated.columns = ["Movie", "Average Rating", "Number of Ratings"]
    st.dataframe(top_rated, use_container_width=True, hide_index=True)

    st.subheader("Filtering Used in the Project")
    f1, f2 = st.columns(2)
    f1.metric("Minimum user ratings", "20")
    f2.metric("Minimum movie ratings", "5")

    st.write(
        f"Users before filtering: **{merged['userId'].nunique():,}**  \n"
        f"Users after filtering: **{filtered['userId'].nunique():,}**  \n"
        f"Movies before filtering: **{merged['movieId'].nunique():,}**  \n"
        f"Movies after filtering: **{filtered['movieId'].nunique():,}**"
    )

# -----------------------------
# About
# -----------------------------
else:
    st.title("ℹ️ About the Project")

    st.markdown("""
    ### Personalized Movie Recommendation System

    This project uses **collaborative filtering** to recommend movies
    based on historical user ratings.

    #### Recommendation pipeline

    1. Load movie and rating data
    2. Check and merge the datasets
    3. Filter low-activity users and movies
    4. Split the ratings into training and testing data
    5. Build the user-movie rating matrix
    6. Calculate user-user and item-item cosine similarity
    7. Predict ratings using collaborative filtering
    8. Evaluate using RMSE, MAE and Precision@10
    9. Add minimum-rating filtering and shrinkage
    10. Blend personalized scores with popularity
    11. Handle cold-start users with a popularity fallback
    12. Generate explanations for recommendations

    #### Final recommendation logic

    The final notebook uses a **hybrid score** with:

    - `alpha = 0.6`
    - minimum movie ratings = `20` for the hybrid scorer
    - shrinkage = `10`
    - cold-start threshold = `5` user ratings

    For users with insufficient history, the system recommends popular
    movies with at least 50 ratings.
    """)

    st.success(
        "The Streamlit interface is a frontend for the collaborative-filtering "
        "pipeline developed in the project notebook."
    )
