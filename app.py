import streamlit as st
import pandas as pd
import pickle
from gensim import corpora, models
from keybert import KeyBERT
from preprocess import clean_text

# Load models
clf = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
lda = models.LdaModel.load("lda_model.gensim")
dictionary = corpora.Dictionary.load("dictionary.dict")
kw_model = KeyBERT()

st.set_page_config(page_title="Instructor Engagement Review Analyzer")
st.title("📊 Sentiment & Topic Review Analyzer")

def analyze_review(text):
    tokens = clean_text(text)
    joined = " ".join(tokens)
    sentiment = clf.predict(tfidf.transform([joined]))[0]
    bow = dictionary.doc2bow(tokens)
    topics = lda.get_document_topics(bow)
    dominant_topic = max(topics, key=lambda x: x[1]) if topics else (-1, 0.0)
    keywords = [kw[0] for kw in kw_model.extract_keywords(text, top_n=3)]
    return {
        "sentiment_label": sentiment,
        "dominant_topic": dominant_topic[0],
        "confidence_level": round(dominant_topic[1], 3),
        "keyword_tags": keywords
    }

# UI
st.subheader("📝 Analyze a Single Review")
review_input = st.text_area("Enter review text here:")
if st.button("Analyze Review"):
    result = analyze_review(review_input)
    st.json(result)

st.subheader("📁 Upload Batch of Reviews (CSV)")
uploaded_file = st.file_uploader("Upload a CSV with a 'review_text' column")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "review_text" in df.columns:
        with st.spinner("Analyzing..."):
            results = [analyze_review(text) for text in df["review_text"]]
            output_df = df.copy()
            output_df["sentiment_label"] = [r["sentiment_label"] for r in results]
            output_df["dominant_topic"] = [r["dominant_topic"] for r in results]
            output_df["keyword_tags"] = [", ".join(r["keyword_tags"]) for r in results]
            output_df["confidence_level"] = [r["confidence_level"] for r in results]
            st.dataframe(output_df.head())
            st.download_button("📥 Download Results", output_df.to_csv(index=False), "structured_output.csv", "text/csv")
    else:
        st.error("CSV must contain a 'review_text' column.")
