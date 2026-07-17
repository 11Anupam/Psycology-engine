# 🧠 Consumer Psychology Engine

> Decode why customers buy — and why they don't.

A Streamlit NLP application that analyzes customer reviews to surface purchase drivers, barriers, sentiment trends, and brand perception — all visualized in a dark-themed dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Upload CSV** or use the built-in 300-review sample dataset
- **Sentiment Analysis** via VADER (Positive / Negative / Neutral)
- **Topic Modeling** via LDA (3–8 configurable topics)
- **Keyword Extraction** via TF-IDF
- **N-gram Analysis** (Bigrams + Trigrams)
- **Purchase Driver Scoring** — Quality, Performance, Design, Convenience, Value, Features, Brand Trust
- **Purchase Barrier Scoring** — Price, Quality Issues, Service, Delivery, Usability, Performance
- **Brand Loyalty & Value-for-Money** signal detection
- **6-tab Plotly dashboard** with dark theme
- **Auto-generated Executive Summary** (downloadable as `.txt`)
- **Processed data export** as CSV

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/consumer-psychology-engine.git
cd consumer-psychology-engine
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📂 Project Structure

```
consumer-psychology-engine/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── sample_data/
│   └── sample_reviews.csv  # Example CSV for testing
├── .gitignore
└── README.md
```

---

## 📋 CSV Format

If uploading your own data, your CSV must have these columns:

| Column        | Type   | Required | Description                  |
|---------------|--------|----------|------------------------------|
| `review_text` | string | ✅ Yes   | The review content           |
| `rating`      | int    | ✅ Yes   | Rating (1–5)                 |
| `date`        | string | ❌ No    | Date (YYYY-MM-DD)            |
| `category`    | string | ❌ No    | Product category (optional)  |

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — live in ~2 minutes

---

## 🧩 Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web app framework |
| VADER | Sentiment analysis |
| NLTK | Tokenization, lemmatization, stopwords |
| scikit-learn | TF-IDF, LDA topic modeling |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📊 Sentiment & Ratings | Sentiment pie, rating histogram, trend over time, score distribution |
| 🔑 Keywords & N-grams | TF-IDF keywords, bigrams, trigrams, keyword pills |
| 🧩 Topic Modeling | LDA topic cards, volume chart, sentiment bubble map |
| 🛒 Drivers & Barriers | Ranked cards, heatmaps, radar chart |
| 📋 Executive Summary | Auto-generated report, downloadable |
| 📄 Raw Data | Filtered table, CSV export |

---

## 🎓 Built by

Ajax (Anupam Gajbhiye) · MBA International Business · SIIB Pune  
Marketing Analytics Portfolio Project

---

## 📄 License

MIT — free to use, modify, and distribute.
