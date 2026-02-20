import streamlit as st
import os
import matplotlib.pyplot as plt
from nlp_utils import (
    load_sentiment_model,
    get_word_frequency,
    analyze_sentiment,
    extract_date_from_filename,
    load_sentiment_cache,
    save_sentiment_cache
)

ARTICLE_FOLDER = "ans_articles"
CACHE_FILE = "sentiment_cache.csv"

@st.cache_resource
def get_model():
    return load_sentiment_model()

model = get_model()

st.title("News Article NLP Dashboard")

files = [f for f in os.listdir(ARTICLE_FOLDER) if f.endswith(".txt")]

if not files:
    st.warning("No articles found.")
    st.stop()

# -----------------------------------
# Load or Build Sentiment Cache
# -----------------------------------
sentiment_cache = load_sentiment_cache(CACHE_FILE)
updated = False

for file in files:
    if file not in sentiment_cache:
        with open(os.path.join(ARTICLE_FOLDER, file), "r", encoding="utf-8") as f:
            text = f.read()

        score = analyze_sentiment(text, model)
        sentiment_cache[file] = score
        updated = True

if updated:
    save_sentiment_cache(CACHE_FILE, sentiment_cache)

# -----------------------------------
# Single Article View
# -----------------------------------
selected_file = st.selectbox("Choose an article", files)

if selected_file:
    with open(os.path.join(ARTICLE_FOLDER, selected_file), "r", encoding="utf-8") as f:
        text = f.read()

    st.subheader("Word Frequency (Stopwords Removed)")
    freq_data = get_word_frequency(text)

    words = [w[0] for w in freq_data]
    counts = [w[1] for w in freq_data]

    fig, ax = plt.subplots()
    ax.bar(words, counts)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    score = sentiment_cache[selected_file]

    if score > 0.05:
        label = "Positive"
    elif score < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    st.subheader("Sentiment (BERT)")
    st.write(f"Sentiment: **{label}**")
    st.write(f"Score: {round(score, 3)}")

# -----------------------------------
# Sentiment Trend
# -----------------------------------
st.subheader("Sentiment Trend Over Time")

dates = []
scores = []

for file, score in sentiment_cache.items():
    date = extract_date_from_filename(file)
    if not date:
        continue
    dates.append(date)
    scores.append(score)

if dates:
    sorted_data = sorted(zip(dates, scores))
    dates_sorted, scores_sorted = zip(*sorted_data)

    fig2, ax2 = plt.subplots()
    ax2.plot(dates_sorted, scores_sorted)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Sentiment Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.write("No valid dates found in filenames.")
