
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Title
st.title("📝 Product Review Sentiment Analyzer")

# User input
user_input = st.text_area("Enter your product review:")

# Analyze button
if st.button("Analyze Sentiment"):
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Polarity Score:** {round(polarity, 3)}")

    # Word cloud
    if user_input:
        wordcloud = WordCloud(width=600, height=400, background_color='white').generate(user_input)
        st.subheader("Word Cloud")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
