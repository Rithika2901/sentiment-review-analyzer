import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download VADER lexicon (only the first time)
nltk.download('vader_lexicon')

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Title
st.title("📝 Product Review Sentiment Analyzer (VADER Version)")

# Input
user_input = st.text_area("Enter your product review:")

# Analyze Button
if st.button("Analyze Sentiment"):
    scores = analyzer.polarity_scores(user_input)
    polarity = scores['compound']

    # Determine sentiment label
    if polarity >= 0.05:
        sentiment = "Positive 😊"
    elif polarity <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Compound Score:** {round(polarity, 3)}")

    # Word Cloud
    if user_input:
        wordcloud = WordCloud(width=600, height=400, background_color='white').generate(user_input)
        st.subheader("Word Cloud")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
