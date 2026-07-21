import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
import json
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK data
for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger"]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Psychology Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0b0f1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
    border-right: 1px solid #1e2d3d;
  }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0f1c2e 0%, #162032 50%, #0f1c2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,183,255,0.08) 0%, transparent 70%);
  }
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e6edf3;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    color: #7d8fa3;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
  }
  .accent { color: #00b7ff; }

  /* Metric cards */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #111827;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #00b7ff44; }
  .metric-label {
    color: #6b7a8d;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    color: #e6edf3;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
  }
  .metric-delta {
    font-size: 0.75rem;
    margin-top: 0.3rem;
    font-weight: 500;
  }
  .delta-pos { color: #3fb950; }
  .delta-neg { color: #f85149; }
  .delta-neu { color: #7d8fa3; }

  /* Section headers */
  .section-header {
    color: #e6edf3;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d3d;
  }
  .section-tag {
    display: inline-block;
    background: #0d2137;
    color: #00b7ff;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.5rem;
  }

  /* Driver / barrier cards */
  .driver-card {
    background: #0d1220;
    border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
  }
  .barrier-card {
    background: #0d1220;
    border-left: 3px solid #f85149;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
  }
  .loyalty-card {
    background: #0d1220;
    border-left: 3px solid #d29922;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
  }
  .card-title { color: #e6edf3; font-size: 0.85rem; font-weight: 600; }
  .card-count { color: #7d8fa3; font-size: 0.75rem; }
  .card-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 0.4rem;
    background: #1e2d3d;
  }
  .card-fill-driver { height: 100%; border-radius: 2px; background: #3fb950; }
  .card-fill-barrier { height: 100%; border-radius: 2px; background: #f85149; }
  .card-fill-loyalty { height: 100%; border-radius: 2px; background: #d29922; }

  /* Executive summary */
  .exec-summary {
    background: linear-gradient(135deg, #0f1c2e 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.5rem;
  }
  .exec-block {
    background: #0d1220;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid #1e2d3d;
  }
  .exec-block-title {
    color: #00b7ff;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
  }
  .exec-block-text { color: #c9d1d9; font-size: 0.85rem; line-height: 1.6; }

  /* Topic pills */
  .topic-pill {
    display: inline-block;
    background: #0d2137;
    color: #58a6ff;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
  }

  /* Wow insight card */
  .wow-card {
    background: linear-gradient(135deg, #0a1628 0%, #0f1f35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .wow-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00b7ff, #3fb950, #d29922, #f85149);
  }
  .wow-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }
  .wow-block {
    background: #0d1220;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border: 1px solid #1e2d3d;
  }
  .wow-block-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
  }
  .wow-block-text {
    color: #c9d1d9;
    font-size: 0.88rem;
    line-height: 1.6;
  }
  .urgency-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    background: #1e2d3d;
    margin-left: 0.5rem;
  }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    background: #0d1220;
    border: 1px dashed #1e3a5f;
    border-radius: 8px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7d8fa3;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1e2d3d !important;
    color: #e6edf3 !important;
  }

  /* Streamlit overrides */
  .stButton>button {
    background: linear-gradient(135deg, #0061c8, #00b7ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    width: 100%;
  }
  .stButton>button:hover { opacity: 0.9; }
  div[data-testid="stMarkdownContainer"] p { color: #c9d1d9; }
  .stSelectbox label, .stSlider label, .stTextInput label { color: #7d8fa3 !important; font-size: 0.8rem !important; }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PURCHASE_DRIVERS = {
    "Quality": ["quality", "durable", "durability", "sturdy", "solid", "well-made", "premium", "reliable", "lasting", "robust", "excellent", "superior"],
    "Performance": ["fast", "speed", "performance", "efficient", "powerful", "works great", "responsive", "smooth", "effective", "accurate"],
    "Design": ["design", "beautiful", "stylish", "aesthetic", "look", "appearance", "sleek", "compact", "elegant", "attractive", "gorgeous"],
    "Convenience": ["easy", "convenient", "simple", "hassle-free", "user-friendly", "intuitive", "effortless", "quick", "straightforward"],
    "Value": ["value", "affordable", "worth", "price", "cheap", "budget", "cost-effective", "reasonable", "bang for buck"],
    "Features": ["feature", "functionality", "versatile", "capability", "option", "function", "ability", "support"],
    "Brand Trust": ["brand", "trust", "reputation", "reliable brand", "always buy", "loyal", "recommend"],
}

PURCHASE_BARRIERS = {
    "Price": ["expensive", "overpriced", "costly", "pricey", "too much", "not worth", "price"],
    "Quality Issues": ["defective", "broken", "poor quality", "cheaply made", "flimsy", "fragile", "fell apart", "broke", "defect", "faulty"],
    "Service Issues": ["customer service", "support", "rude", "unhelpful", "no response", "bad service", "terrible service"],
    "Delivery Issues": ["late delivery", "damaged", "packaging", "shipping", "delay", "not delivered", "wrong item"],
    "Usability Issues": ["confusing", "difficult", "hard to use", "complicated", "unclear instructions", "not intuitive"],
    "Performance Issues": ["slow", "laggy", "doesn't work", "stopped working", "not working", "malfunction", "bug", "glitch"],
}

LOYALTY_INDICATORS = ["repurchase", "buy again", "reorder", "loyal", "always buy", "lifetime customer", "recommend", "tell friends", "love this brand", "never switching", "best brand"]
VALUE_TERMS = ["value for money", "worth the price", "great deal", "affordable", "bang for buck", "cost effective", "good price", "reasonable price", "money well spent"]

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0d1220",
    font=dict(color="#c9d1d9", family="Inter"),
    title_font=dict(color="#e6edf3", size=14),
    xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", tickfont=dict(color="#7d8fa3")),
    yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", tickfont=dict(color="#7d8fa3")),
    colorway=["#00b7ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#ffa657", "#79c0ff"],
)

COLORS = {
    "positive": "#3fb950",
    "negative": "#f85149",
    "neutral": "#7d8fa3",
    "primary": "#00b7ff",
    "warning": "#d29922",
    "purple": "#a371f7",
}

# ── NLP helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp_tools():
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) | {
        "product", "item", "bought", "purchase", "buy", "ordered", "amazon",
        "get", "got", "one", "would", "also", "really", "very", "much",
        "us", "use", "used", "using", "like", "just", "even", "good",
    }
    return sia, lemmatizer, stop_words

def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def get_sentiment(text, sia):
    scores = sia.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return compound, label, scores["pos"], scores["neg"], scores["neu"]

def extract_ngrams(texts, n=2, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), max_features=500, min_df=2)
    try:
        X = vec.fit_transform(texts)
        sums = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        idx = np.argsort(sums)[::-1][:top_k]
        return [(vocab[i], int(sums[i])) for i in idx]
    except:
        return []

def extract_keywords(texts, top_k=30):
    try:
        tfidf = TfidfVectorizer(max_features=200, min_df=2, ngram_range=(1, 2))
        X = tfidf.fit_transform(texts)
        scores = X.sum(axis=0).A1
        vocab = tfidf.get_feature_names_out()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(vocab[i], float(scores[i])) for i in idx]
    except:
        return []

def run_lda(texts, n_topics=5):
    try:
        cv = CountVectorizer(max_features=500, min_df=2, max_df=0.9)
        dtm = cv.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15)
        lda.fit(dtm)
        vocab = cv.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lda.components_):
            top_words = [vocab[j] for j in comp.argsort()[-8:][::-1]]
            topics.append({"id": i + 1, "words": top_words, "label": f"Topic {i+1}: {top_words[0].title()}"})
        doc_topics = lda.transform(dtm)
        return topics, doc_topics
    except:
        return [], np.array([])

def score_drivers_barriers(texts_raw, patterns_dict):
    results = {}
    full_text = " ".join(str(t).lower() for t in texts_raw if isinstance(t, str))
    total = len(texts_raw)
    for category, keywords in patterns_dict.items():
        count = sum(full_text.count(kw.lower()) for kw in keywords)
        review_hits = sum(
            1 for t in texts_raw
            if isinstance(t, str) and any(kw.lower() in t.lower() for kw in keywords)
        )
        results[category] = {"mentions": count, "reviews": review_hits, "pct": round(review_hits / max(total, 1) * 100, 1)}
    return dict(sorted(results.items(), key=lambda x: x[1]["mentions"], reverse=True))

def generate_wow_insight(df, drivers, barriers, keywords, pct_positive, pct_negative, avg_rating):
    """Generate a data-driven top insight + recommendation card."""

    # --- Top insight: best driver in positive reviews vs best barrier in negative reviews ---
    pos_reviews = df[df["sentiment_label"] == "Positive"]["review_text"].tolist()
    neg_reviews = df[df["sentiment_label"] == "Negative"]["review_text"].tolist()
    n_pos = max(len(pos_reviews), 1)
    n_neg = max(len(neg_reviews), 1)

    # Score drivers against positive reviews only
    pos_driver_pcts = {}
    for cat, kws in PURCHASE_DRIVERS.items():
        hits = sum(1 for t in pos_reviews if isinstance(t, str) and any(k.lower() in t.lower() for k in kws))
        pos_driver_pcts[cat] = round(hits / n_pos * 100, 1)

    # Score barriers against negative reviews only
    neg_barrier_pcts = {}
    for cat, kws in PURCHASE_BARRIERS.items():
        hits = sum(1 for t in neg_reviews if isinstance(t, str) and any(k.lower() in t.lower() for k in kws))
        neg_barrier_pcts[cat] = round(hits / n_neg * 100, 1)

    top_pos_driver = max(pos_driver_pcts, key=pos_driver_pcts.get)
    top_pos_pct    = pos_driver_pcts[top_pos_driver]
    top_neg_barrier = max(neg_barrier_pcts, key=neg_barrier_pcts.get)
    top_neg_pct    = neg_barrier_pcts[top_neg_barrier]

    insight = (
        f"<b style='color:#3fb950'>{top_pos_driver}</b> is mentioned in "
        f"<b style='color:#3fb950'>{top_pos_pct:.0f}%</b> of positive reviews, "
        f"while <b style='color:#f85149'>{top_neg_barrier}</b> appears in "
        f"<b style='color:#f85149'>{top_neg_pct:.0f}%</b> of negative reviews."
    )

    # --- Recommendation: highest-impact action ---
    # Find the barrier with highest negative review penetration
    top_barrier_name = max(barriers, key=lambda k: barriers[k]["pct"]) if barriers else top_neg_barrier
    top_barrier_pct  = barriers[top_barrier_name]["pct"] if barriers else top_neg_pct

    # Find the driver with highest positive review penetration
    top_driver_name = max(drivers, key=lambda k: drivers[k]["pct"]) if drivers else top_pos_driver
    top_driver_pct  = drivers[top_driver_name]["pct"] if drivers else top_pos_pct

    # Sentiment gap signal
    if pct_negative > 30:
        urgency = "🚨 High urgency"
        urgency_color = "#f85149"
    elif pct_negative > 15:
        urgency = "⚠️ Medium urgency"
        urgency_color = "#d29922"
    else:
        urgency = "✅ Low urgency"
        urgency_color = "#3fb950"

    recommendation = (
        f"Customers love <b style='color:#3fb950'>{top_driver_name}</b> "
        f"({top_driver_pct:.0f}% of reviews) but repeatedly flag "
        f"<b style='color:#f85149'>{top_barrier_name}</b> "
        f"({top_barrier_pct:.0f}% of reviews). "
        f"Fixing <b style='color:#f85149'>{top_barrier_name}</b> has the highest potential to shift sentiment "
        f"and improve the current <b style='color:#e6edf3'>{avg_rating:.1f}★</b> rating."
    )

    return insight, recommendation, urgency, urgency_color, top_driver_name, top_barrier_name


def generate_sample_data(n=300):
    np.random.seed(42)
    positive_reviews = [
        "Amazing quality product, very durable and well-made. Worth every penny!",
        "Excellent performance and fast delivery. Will definitely buy again.",
        "Great value for money. The design is sleek and modern. Highly recommend!",
        "Best product I've ever bought. Customer service was fantastic too.",
        "Super easy to use, intuitive interface. Works perfectly as described.",
        "Premium quality, exactly what I expected. Very satisfied with my purchase.",
        "Incredible build quality. Very sturdy and reliable. Love the design!",
        "Outstanding performance. Battery life is exceptional. Great brand trust.",
        "Perfect product for the price. Fast shipping and well packaged.",
        "Highly recommend this product. Excellent features and top notch quality.",
        "Love the sleek design and premium feel. Works amazingly well.",
        "Value for money is unbeatable. Customer support was very helpful.",
        "Durable and long-lasting. The build quality is top notch.",
        "Smooth performance, no lag. Very user-friendly. Great buy!",
        "Best value product in this category. Beautiful design, works great.",
    ]
    negative_reviews = [
        "Very expensive for the quality offered. Broke after 2 weeks of use.",
        "Terrible customer service. Product stopped working after one month.",
        "Not worth the price at all. Very poor quality, flimsy material.",
        "Late delivery and damaged packaging. Very disappointed with this purchase.",
        "Stopped working after first use. Defective product. Avoid this brand.",
        "Overpriced and underperformed. The design looks cheap in person.",
        "Customer support was rude and unhelpful. Will not buy again.",
        "Hard to use, very confusing instructions. Waste of money.",
        "Quality issues from day one. The product fell apart quickly.",
        "Slow performance and constant glitches. Very frustrating experience.",
        "Not as described. Poor quality materials. Expensive for what it is.",
        "Delivery took 3 weeks. Product was damaged. Terrible experience.",
        "Defective out of the box. Customer service gave no refund.",
        "Very disappointing. Performance is nothing like advertised.",
        "Fragile and poorly made. Broke within a week. Bad value.",
    ]
    neutral_reviews = [
        "Product is okay. Nothing special but does the job.",
        "Average quality. Price is fair for what you get.",
        "Works as expected. Delivery was on time. Pretty standard.",
        "Not bad, not great. Some features could be improved.",
        "Decent product overall. Has some minor issues but functional.",
        "It's fine for the price. Don't expect premium quality.",
        "Middle of the road product. Gets the job done.",
        "Satisfactory purchase. Could have better packaging.",
        "Average experience. Product is functional but basic.",
        "Okay product. Meets basic requirements. Nothing more.",
    ]
    reviews = []
    for _ in range(n):
        r = np.random.random()
        if r < 0.55:
            text = np.random.choice(positive_reviews)
            rating = np.random.choice([4, 5], p=[0.35, 0.65])
        elif r < 0.80:
            text = np.random.choice(neutral_reviews)
            rating = np.random.choice([3, 4], p=[0.7, 0.3])
        else:
            text = np.random.choice(negative_reviews)
            rating = np.random.choice([1, 2], p=[0.5, 0.5])

        noise = [
            "I would", "In my experience,", "Overall,", "Honestly,",
            "After using for a month,", "Long term review:", "Update:", "Pro tip:",
        ]
        text = np.random.choice(noise) + " " + text

        days_back = np.random.randint(1, 730)
        date = (datetime.now() - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
        product_cat = np.random.choice(["Electronics", "Home & Kitchen", "Sports", "Fashion"])
        reviews.append({"review_text": text, "rating": rating, "date": date, "category": product_cat})

    return pd.DataFrame(reviews)

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    sia, lemmatizer, stop_words = load_nlp_tools()

    # Hero
    st.markdown("""
    <div class="hero-banner">
      <p style="margin:0 0 0.3rem 0; color:#00b7ff; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em;">Marketing Intelligence · NLP Powered</p>
      <h1 class="hero-title">Consumer Psychology <span class="accent">Engine</span></h1>
      <p class="hero-sub">Decode why customers buy — and why they don't. Upload reviews or use sample data to unlock purchase drivers, barriers, and brand perception.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        n_topics = st.slider("LDA Topics", 3, 8, 5)
        top_k_keywords = st.slider("Top Keywords", 10, 40, 20)
        min_rating_filter = st.selectbox("Filter Reviews by Rating", ["All Ratings", "1-2 (Negative)", "3 (Neutral)", "4-5 (Positive)"])
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        use_sample = st.checkbox("Use Sample Dataset", value=True)
        uploaded_file = None
        if not use_sample:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="CSV must have: review_text, rating, date columns")
        st.markdown("---")
        analyze_btn = st.button("🚀 Run Analysis")
        st.markdown("---")
        st.markdown("<p style='color:#7d8fa3;font-size:0.72rem;'>Built with VADER · LDA · TF-IDF<br>Consumer Psychology Engine v1.0</p>", unsafe_allow_html=True)

    # Load data
    if "df_results" not in st.session_state:
        st.session_state.df_results = None

    if analyze_btn or st.session_state.df_results is None:
        with st.spinner("Loading & preprocessing reviews…"):
            if use_sample or uploaded_file is None:
                df = generate_sample_data(300)
                data_source = "Sample Dataset (300 reviews)"
            else:
                df = pd.read_csv(uploaded_file)
                data_source = f"Uploaded: {uploaded_file.name}"

            # Validate columns
            required = ["review_text", "rating"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}. CSV must have: review_text, rating, date")
                return

            if "date" not in df.columns:
                df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq="D").strftime("%Y-%m-%d")

            # Clean
            df["review_text"] = df["review_text"].astype(str)
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).clip(1, 5).astype(int)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["review_text"])
            df = df.drop_duplicates(subset=["review_text"])

        with st.spinner("Running sentiment analysis…"):
            sentiment_results = df["review_text"].apply(lambda t: get_sentiment(t, sia))
            df["sentiment_score"] = [r[0] for r in sentiment_results]
            df["sentiment_label"] = [r[1] for r in sentiment_results]
            df["pos_score"] = [r[2] for r in sentiment_results]
            df["neg_score"] = [r[3] for r in sentiment_results]
            df["neu_score"] = [r[4] for r in sentiment_results]

        with st.spinner("Cleaning text & extracting features…"):
            df["clean_text"] = df["review_text"].apply(lambda t: clean_text(t, lemmatizer, stop_words))
            clean_texts = df["clean_text"].tolist()
            raw_texts = df["review_text"].tolist()

        with st.spinner("Running topic modeling (LDA)…"):
            topics, doc_topics = run_lda(clean_texts, n_topics=n_topics)
            if len(doc_topics) > 0:
                df["dominant_topic"] = np.argmax(doc_topics, axis=1) + 1

        with st.spinner("Extracting keywords & n-grams…"):
            keywords = extract_keywords(clean_texts, top_k=top_k_keywords)
            bigrams = extract_ngrams(clean_texts, n=2, top_k=20)
            trigrams = extract_ngrams(clean_texts, n=3, top_k=15)

        with st.spinner("Scoring purchase drivers & barriers…"):
            drivers = score_drivers_barriers(raw_texts, PURCHASE_DRIVERS)
            barriers = score_drivers_barriers(raw_texts, PURCHASE_BARRIERS)
            n_total = len(df)
            loyalty_count = sum(1 for t in raw_texts if isinstance(t, str) and any(kw in t.lower() for kw in LOYALTY_INDICATORS))
            value_count = sum(1 for t in raw_texts if isinstance(t, str) and any(kw in t.lower() for kw in VALUE_TERMS))

        st.session_state.df_results = df
        st.session_state.analysis = {
            "data_source": data_source,
            "topics": topics,
            "doc_topics": doc_topics,
            "keywords": keywords,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "drivers": drivers,
            "barriers": barriers,
            "loyalty_count": loyalty_count,
            "value_count": value_count,
        }

    df = st.session_state.df_results
    analysis = st.session_state.analysis

    # Apply rating filter
    if min_rating_filter == "1-2 (Negative)":
        df_view = df[df["rating"] <= 2]
    elif min_rating_filter == "3 (Neutral)":
        df_view = df[df["rating"] == 3]
    elif min_rating_filter == "4-5 (Positive)":
        df_view = df[df["rating"] >= 4]
    else:
        df_view = df

    # Metrics
    n_total = len(df_view)
    avg_rating = df_view["rating"].mean()
    avg_sent = df_view["sentiment_score"].mean()
    pct_positive = (df_view["sentiment_label"] == "Positive").mean() * 100
    pct_negative = (df_view["sentiment_label"] == "Negative").mean() * 100
    loyalty_pct = analysis["loyalty_count"] / max(len(df), 1) * 100
    value_pct = analysis["value_count"] / max(len(df), 1) * 100

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Total Reviews</div>
        <div class="metric-value">{n_total:,}</div>
        <div class="metric-delta delta-neu">{analysis['data_source'][:28]}…</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Rating</div>
        <div class="metric-value">{'⭐' * round(avg_rating)} {avg_rating:.2f}</div>
        <div class="metric-delta {'delta-pos' if avg_rating >= 3.5 else 'delta-neg'}">out of 5.0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Positive Sentiment</div>
        <div class="metric-value" style="color:#3fb950">{pct_positive:.1f}%</div>
        <div class="metric-delta delta-neg">Negative: {pct_negative:.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Brand Loyalty Signals</div>
        <div class="metric-value" style="color:#d29922">{loyalty_pct:.1f}%</div>
        <div class="metric-delta delta-neu">Value mentions: {value_pct:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Wow Insight Card ──────────────────────────────────────────────────────
    insight, recommendation, urgency, urgency_color, top_drv, top_bar = generate_wow_insight(
        df_view, analysis["drivers"], analysis["barriers"],
        analysis["keywords"], pct_positive, pct_negative, avg_rating
    )
    st.markdown(f"""
    <div class="wow-card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem">
        <span style="color:#00b7ff;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">&#9889; AI Insight Engine</span>
        <span class="urgency-badge" style="color:{urgency_color}">{urgency}</span>
      </div>
      <div class="wow-grid">
        <div class="wow-block">
          <div class="wow-block-label" style="color:#00b7ff">&#128269; Top Insight</div>
          <div class="wow-block-text">{insight}</div>
        </div>
        <div class="wow-block">
          <div class="wow-block-label" style="color:#3fb950">&#127919; #1 Recommendation</div>
          <div class="wow-block-text">{recommendation}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["📊 Sentiment & Ratings", "🔑 Keywords & N-grams", "🧩 Topic Modeling", "🛒 Drivers & Barriers", "📋 Executive Summary", "📄 Raw Data"])

    # ── Tab 1: Sentiment & Ratings ────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            sent_counts = df_view["sentiment_label"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                sent_counts, values="Count", names="Sentiment",
                color="Sentiment",
                color_discrete_map={"Positive": COLORS["positive"], "Negative": COLORS["negative"], "Neutral": COLORS["neutral"]},
                hole=0.55,
                title="Sentiment Distribution",
            )
            fig.update_layout(**PLOTLY_TEMPLATE)
            fig.update_traces(textfont_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            rating_counts = df_view["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["Rating", "Count"]
            colors_bar = [COLORS["negative"], COLORS["negative"], COLORS["neutral"], COLORS["positive"], COLORS["positive"]]
            fig2 = go.Figure(go.Bar(
                x=rating_counts["Rating"].astype(str).apply(lambda x: f"⭐{x}"),
                y=rating_counts["Count"],
                marker_color=[colors_bar[r - 1] for r in rating_counts["Rating"]],
                text=rating_counts["Count"],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig2.update_layout(**PLOTLY_TEMPLATE, title="Rating Distribution", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Rating trend over time
        df_time = df_view.copy()
        df_time = df_time.dropna(subset=["date"])
        if len(df_time) > 10:
            df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
            trend = df_time.groupby("month").agg(
                avg_rating=("rating", "mean"),
                avg_sentiment=("sentiment_score", "mean"),
                count=("rating", "count")
            ).reset_index()
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(
                x=trend["month"], y=trend["avg_rating"],
                name="Avg Rating", line=dict(color=COLORS["primary"], width=2.5),
                mode="lines+markers", marker=dict(size=5),
            ), secondary_y=False)
            fig3.add_trace(go.Bar(
                x=trend["month"], y=trend["count"],
                name="Review Count", marker_color="#1e2d3d", opacity=0.6,
            ), secondary_y=True)
            fig3.update_layout(**PLOTLY_TEMPLATE, title="Rating Trend Over Time", showlegend=True)
            fig3.update_yaxes(title_text="Avg Rating", range=[1, 5], secondary_y=False, title_font=dict(color="#7d8fa3"))
            fig3.update_yaxes(title_text="# Reviews", secondary_y=True, title_font=dict(color="#7d8fa3"))
            st.plotly_chart(fig3, use_container_width=True)

        # Sentiment score distribution
        fig4 = px.histogram(
            df_view, x="sentiment_score", nbins=40,
            color_discrete_sequence=[COLORS["primary"]],
            title="Sentiment Score Distribution (VADER Compound)",
        )
        fig4.add_vline(x=0.05, line_dash="dash", line_color=COLORS["positive"], annotation_text="Positive threshold")
        fig4.add_vline(x=-0.05, line_dash="dash", line_color=COLORS["negative"], annotation_text="Negative threshold")
        fig4.update_layout(**PLOTLY_TEMPLATE)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2: Keywords & N-grams ─────────────────────────────────────────────
    with tabs[1]:
        keywords = analysis["keywords"]
        bigrams = analysis["bigrams"]
        trigrams = analysis["trigrams"]

        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
            fig_kw = px.bar(
                kw_df.head(25), x="TF-IDF Score", y="Keyword",
                orientation="h",
                color="TF-IDF Score",
                color_continuous_scale=[[0, "#0d2137"], [0.5, "#0061c8"], [1, "#00b7ff"]],
                title=f"Top {min(25, len(kw_df))} Keywords by TF-IDF Score",
            )
            fig_kw.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if bigrams:
                bg_df = pd.DataFrame(bigrams, columns=["Bigram", "Count"])
                fig_bg = px.bar(
                    bg_df.head(15), x="Count", y="Bigram",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, "#0d2137"], [1, "#3fb950"]],
                    title="Top Bigrams",
                )
                fig_bg.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
                fig_bg.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_bg, use_container_width=True)

        with c2:
            if trigrams:
                tg_df = pd.DataFrame(trigrams, columns=["Trigram", "Count"])
                fig_tg = px.bar(
                    tg_df.head(15), x="Count", y="Trigram",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, "#0d2137"], [1, "#d29922"]],
                    title="Top Trigrams",
                )
                fig_tg.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
                fig_tg.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_tg, use_container_width=True)

        # Keyword pills display
        st.markdown("<div class='section-header'><span class='section-tag'>KEYWORDS</span>Visual Overview</div>", unsafe_allow_html=True)
        pills_html = "".join(f"<span class='topic-pill'>{kw} ({score:.2f})</span>" for kw, score in keywords[:40])
        st.markdown(f"<div style='line-height:2.2'>{pills_html}</div>", unsafe_allow_html=True)

    # ── Tab 3: Topic Modeling ─────────────────────────────────────────────────
    with tabs[2]:
        topics = analysis["topics"]
        if not topics:
            st.warning("Topic modeling failed — try with more data.")
        else:
            st.markdown("<div class='section-header'><span class='section-tag'>LDA</span>Identified Topics</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(topics), 3))
            for i, topic in enumerate(topics):
                with cols[i % len(cols)]:
                    words_html = "".join(f"<span class='topic-pill'>{w}</span>" for w in topic["words"])
                    st.markdown(f"""
                    <div class='exec-block'>
                      <div class='exec-block-title'>Topic {topic['id']}: {topic['words'][0].upper()}</div>
                      <div>{words_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Topic distribution chart
            if "dominant_topic" in df_view.columns:
                topic_dist = df_view["dominant_topic"].value_counts().sort_index()
                topic_labels = [f"T{i}: {topics[i-1]['words'][0].title()}" for i in topic_dist.index if i-1 < len(topics)]
                fig_topics = go.Figure(go.Bar(
                    x=topic_labels,
                    y=topic_dist.values,
                    marker_color=[COLORS["primary"], COLORS["positive"], COLORS["warning"], COLORS["negative"], COLORS["purple"]][:len(topic_labels)],
                    text=topic_dist.values,
                    textposition="outside",
                    textfont=dict(color="#e6edf3"),
                ))
                fig_topics.update_layout(**PLOTLY_TEMPLATE, title="Review Volume by Topic", showlegend=False)
                st.plotly_chart(fig_topics, use_container_width=True)

            # Sentiment by topic heatmap
            if "dominant_topic" in df_view.columns:
                topic_sentiment = df_view.groupby("dominant_topic")["sentiment_score"].mean()
                topic_ratings = df_view.groupby("dominant_topic")["rating"].mean()
                topic_counts = df_view.groupby("dominant_topic")["sentiment_score"].count()
                heatmap_data = pd.DataFrame({
                    "Avg Sentiment": topic_sentiment,
                    "Avg Rating": topic_ratings,
                    "Review Count": topic_counts,
                }).reset_index()
                heatmap_data.columns = ["Topic", "Avg Sentiment", "Avg Rating", "Review Count"]
                heatmap_data["Topic Label"] = heatmap_data["Topic"].apply(
                    lambda x: f"T{x}: {topics[x-1]['words'][0].title()}" if x-1 < len(topics) else f"T{x}"
                )
                fig_heat = px.scatter(
                    heatmap_data,
                    x="Avg Sentiment", y="Avg Rating",
                    size="Review Count", color="Avg Sentiment",
                    color_continuous_scale=[[0, "#f85149"], [0.5, "#d29922"], [1, "#3fb950"]],
                    text="Topic Label",
                    title="Topic Sentiment vs Rating Bubble Map",
                    size_max=50,
                )
                fig_heat.update_traces(textposition="top center", textfont=dict(color="#e6edf3", size=10))
                fig_heat.update_layout(**PLOTLY_TEMPLATE)
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 4: Drivers & Barriers ─────────────────────────────────────────────
    with tabs[3]:
        drivers = analysis["drivers"]
        barriers = analysis["barriers"]
        n_total_reviews = len(df)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'><span class='section-tag'>DRIVERS</span>Why Customers Buy</div>", unsafe_allow_html=True)
            max_d = max((v["mentions"] for v in drivers.values()), default=1)
            for cat, vals in list(drivers.items())[:7]:
                pct_bar = int(vals["mentions"] / max_d * 100)
                st.markdown(f"""
                <div class='driver-card'>
                  <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div class='card-title'>{cat}</div>
                    <div style='color:#3fb950;font-size:0.8rem;font-weight:600'>{vals['pct']}% reviews</div>
                  </div>
                  <div class='card-count'>{vals['mentions']} mentions · {vals['reviews']} reviews</div>
                  <div class='card-bar'><div class='card-fill-driver' style='width:{pct_bar}%'></div></div>
                </div>
                """, unsafe_allow_html=True)

            # Driver chart
            d_df = pd.DataFrame([{"Driver": k, "Mentions": v["mentions"], "Reviews%": v["pct"]} for k, v in drivers.items()])
            fig_d = px.bar(d_df, x="Mentions", y="Driver", orientation="h",
                           color="Mentions",
                           color_continuous_scale=[[0, "#0d2137"], [1, "#3fb950"]],
                           title="Purchase Drivers Heatmap")
            fig_d.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
            fig_d.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'><span class='section-tag'>BARRIERS</span>Why Customers Don't Buy</div>", unsafe_allow_html=True)
            max_b = max((v["mentions"] for v in barriers.values()), default=1)
            for cat, vals in list(barriers.items())[:7]:
                pct_bar = int(vals["mentions"] / max_b * 100)
                st.markdown(f"""
                <div class='barrier-card'>
                  <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div class='card-title'>{cat}</div>
                    <div style='color:#f85149;font-size:0.8rem;font-weight:600'>{vals['pct']}% reviews</div>
                  </div>
                  <div class='card-count'>{vals['mentions']} mentions · {vals['reviews']} reviews</div>
                  <div class='card-bar'><div class='card-fill-barrier' style='width:{pct_bar}%'></div></div>
                </div>
                """, unsafe_allow_html=True)

            b_df = pd.DataFrame([{"Barrier": k, "Mentions": v["mentions"], "Reviews%": v["pct"]} for k, v in barriers.items()])
            fig_b = px.bar(b_df, x="Mentions", y="Barrier", orientation="h",
                           color="Mentions",
                           color_continuous_scale=[[0, "#0d2137"], [1, "#f85149"]],
                           title="Purchase Barriers Heatmap")
            fig_b.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
            fig_b.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_b, use_container_width=True)

        # Brand Loyalty & Value
        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            loyalty_pct = analysis["loyalty_count"] / max(n_total_reviews, 1) * 100
            st.markdown(f"""
            <div class='loyalty-card'>
              <div class='card-title'>🏆 Brand Loyalty Indicators</div>
              <div class='card-count'>{analysis['loyalty_count']} mentions · {loyalty_pct:.1f}% of reviews</div>
              <div class='card-bar'><div class='card-fill-loyalty' style='width:{min(loyalty_pct*3,100):.0f}%'></div></div>
            </div>
            <div style='color:#7d8fa3;font-size:0.8rem;margin-top:0.5rem'>
              Signals: repurchase intent, "will buy again", recommendations, "loyal customer", brand mentions
            </div>
            """, unsafe_allow_html=True)

        with c4:
            value_pct = analysis["value_count"] / max(n_total_reviews, 1) * 100
            st.markdown(f"""
            <div class='driver-card'>
              <div class='card-title'>💰 Value-for-Money Mentions</div>
              <div class='card-count'>{analysis['value_count']} mentions · {value_pct:.1f}% of reviews</div>
              <div class='card-bar'><div class='card-fill-driver' style='width:{min(value_pct*3,100):.0f}%'></div></div>
            </div>
            <div style='color:#7d8fa3;font-size:0.8rem;margin-top:0.5rem'>
              Signals: "value for money", "worth the price", "great deal", "bang for buck", "cost effective"
            </div>
            """, unsafe_allow_html=True)

        # Combined comparison radar
        all_cats = list(drivers.keys()) + list(barriers.keys())
        all_vals = [v["pct"] for v in drivers.values()] + [v["pct"] for v in barriers.values()]
        all_colors = ["#3fb950"] * len(drivers) + ["#f85149"] * len(barriers)

        fig_radar = go.Figure()
        driver_names = list(drivers.keys())
        driver_vals = [v["pct"] for v in drivers.values()]
        barrier_names = list(barriers.keys())
        barrier_vals = [v["pct"] for v in barriers.values()]

        fig_radar.add_trace(go.Scatterpolar(
            r=driver_vals + [driver_vals[0]],
            theta=driver_names + [driver_names[0]],
            fill="toself",
            name="Purchase Drivers",
            line_color=COLORS["positive"],
            fillcolor="rgba(63,185,80,0.12)",
        ))
        fig_radar.update_layout(
            **PLOTLY_TEMPLATE,
            title="Purchase Driver Coverage (% of Reviews)",
            polar=dict(
                bgcolor="#0d1220",
                radialaxis=dict(visible=True, gridcolor="#1e2d3d", tickfont=dict(color="#7d8fa3")),
                angularaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#c9d1d9")),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 5: Executive Summary ───────────────────────────────────────────────
    with tabs[4]:
        top_driver = max(drivers, key=lambda k: drivers[k]["mentions"]) if drivers else "N/A"
        top_barrier = max(barriers, key=lambda k: barriers[k]["mentions"]) if barriers else "N/A"
        top_keywords_list = [kw for kw, _ in analysis["keywords"][:8]]
        top_topic_words = analysis["topics"][0]["words"][:5] if analysis["topics"] else []

        pct_pos = (df["sentiment_label"] == "Positive").mean() * 100
        pct_neg = (df["sentiment_label"] == "Negative").mean() * 100
        avg_r = df["rating"].mean()
        loyalty_pct2 = analysis["loyalty_count"] / max(len(df), 1) * 100

        driver_summary = ", ".join(f"{k} ({v['pct']}%)" for k, v in list(drivers.items())[:4])
        barrier_summary = ", ".join(f"{k} ({v['pct']}%)" for k, v in list(barriers.items())[:4])

        marketing_implication = []
        if drivers.get("Value", {}).get("pct", 0) > 10:
            marketing_implication.append("Emphasize value-for-money in ad copy and product pages.")
        if drivers.get("Quality", {}).get("pct", 0) > 10:
            marketing_implication.append("Lead with quality/durability proof points (certifications, materials).")
        if drivers.get("Design", {}).get("pct", 0) > 8:
            marketing_implication.append("Invest in visual storytelling — aesthetics drive purchase decisions.")
        if barriers.get("Price", {}).get("pct", 0) > 8:
            marketing_implication.append("Counter price objections with EMI options, bundle deals, or ROI messaging.")
        if barriers.get("Service Issues", {}).get("pct", 0) > 5:
            marketing_implication.append("Improve post-purchase support; feature warranty and return policy prominently.")
        if loyalty_pct2 > 15:
            marketing_implication.append("Launch a loyalty/referral program — strong repurchase intent detected.")
        if not marketing_implication:
            marketing_implication = ["Maintain product quality standards.", "Focus on customer education and onboarding.", "Invest in review generation campaigns."]

        st.markdown(f"""
        <div class="exec-summary">
          <div style="margin-bottom:1.2rem">
            <span class="section-tag">EXECUTIVE SUMMARY</span>
            <span style="color:#7d8fa3;font-size:0.8rem">Based on {len(df):,} reviews · Avg Rating {avg_r:.2f}★ · {pct_pos:.0f}% Positive Sentiment</span>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">🟢 Why Customers Buy</div>
            <div class="exec-block-text">
              The primary purchase driver is <strong style="color:#3fb950">{top_driver}</strong>, appearing in reviews across the dataset.
              The top 4 drivers are: <strong style="color:#e6edf3">{driver_summary}</strong>.
              Customers are motivated by tangible product attributes — particularly how the product performs in real-world use and its perceived quality vs. price ratio.
            </div>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">🔴 Why Customers Don't Buy / Churn</div>
            <div class="exec-block-text">
              The most significant purchase barrier is <strong style="color:#f85149">{top_barrier}</strong>.
              Key friction points: <strong style="color:#e6edf3">{barrier_summary}</strong>.
              Negative sentiment ({pct_neg:.1f}% of reviews) clusters around post-purchase disappointment — suggesting a gap between marketing promise and product delivery.
            </div>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">⭐ Most Appreciated Features</div>
            <div class="exec-block-text">
              Review mining surfaces these most-praised attributes: <strong style="color:#e6edf3">{', '.join(top_keywords_list[:6])}</strong>.
              {"Topic modeling reveals themes around: " + ', '.join(top_topic_words) + "." if top_topic_words else ""}
              Positive reviewers (rating 4-5) consistently emphasize ease of use, build quality, and price-value alignment.
            </div>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">⚠️ Most Common Complaints</div>
            <div class="exec-block-text">
              Critical reviews (rating 1-2) highlight: {barrier_summary}.
              Recurring complaint patterns suggest systemic issues in product quality control and customer service responsiveness.
              Unresolved complaints have the highest churn risk — these customers rarely return.
            </div>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">📈 Brand Loyalty Signal</div>
            <div class="exec-block-text">
              <strong style="color:#d29922">{loyalty_pct2:.1f}%</strong> of reviewers show repurchase or recommendation intent — 
              {"strong loyalty signals detected. This cohort is ideal for referral program activation." if loyalty_pct2 > 15 else "moderate loyalty. Room to improve through post-purchase engagement."}
              Value-for-money is mentioned in {analysis['value_count']} reviews, indicating price sensitivity among this customer base.
            </div>
          </div>

          <div class="exec-block">
            <div class="exec-block-title">🎯 Marketing Implications</div>
            <div class="exec-block-text">
              {'<br>'.join(f"• {imp}" for imp in marketing_implication)}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Download summary
        summary_text = f"""CONSUMER PSYCHOLOGY ENGINE — EXECUTIVE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data: {analysis['data_source']} | Reviews: {len(df):,} | Avg Rating: {avg_r:.2f}

WHY CUSTOMERS BUY
{driver_summary}

WHY CUSTOMERS DON'T BUY
{barrier_summary}

MOST APPRECIATED FEATURES
{', '.join(top_keywords_list)}

BRAND LOYALTY: {loyalty_pct2:.1f}% of reviews
VALUE MENTIONS: {analysis['value_count']} reviews

MARKETING IMPLICATIONS
{chr(10).join('• ' + i for i in marketing_implication)}
"""
        st.download_button(
            "⬇️ Download Executive Summary (.txt)",
            data=summary_text,
            file_name=f"consumer_psychology_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )

    # ── Tab 6: Raw Data ───────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(f"<div class='section-header'><span class='section-tag'>DATA</span>{len(df_view):,} Reviews</div>", unsafe_allow_html=True)
        show_cols = ["review_text", "rating", "sentiment_label", "sentiment_score", "date"]
        show_cols = [c for c in show_cols if c in df_view.columns]
        st.dataframe(
            df_view[show_cols].style.background_gradient(
                subset=["sentiment_score"] if "sentiment_score" in show_cols else [],
                cmap="RdYlGn", vmin=-1, vmax=1
            ),
            use_container_width=True,
            height=500,
        )
        csv = df_view.to_csv(index=False).encode()
        st.download_button("⬇️ Download Processed Data (CSV)", data=csv, file_name="processed_reviews.csv", mime="text/csv")

if __name__ == "__main__":
    main()
