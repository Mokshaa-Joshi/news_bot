import streamlit as st
import os
import re
from pinecone import Pinecone
from deep_translator import GoogleTranslator
import openai
from dotenv import load_dotenv

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("newsbot")

# Common words to ignore in keyword extraction
STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

# Function to extract important keywords from user query
def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(keywords)

# Function to translate input to Gujarati if needed
def translate_to_gujarati(text):
    try:
        if re.search(r'[a-zA-Z]', text):  # If input contains English letters
            return GoogleTranslator(source='en', target='gu').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
    return text

# Function to generate query embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to highlight multiple keywords in text
def highlight_keywords(text, keywords):
    if not text or not keywords:
        return text
    
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    highlighted_text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    
    return highlighted_text

# Function to search news using keyword filtering and vector search
def search_news(query):
    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)

    possible_queries = [cleaned_query, translated_query]
    all_results = []
    
    for q in possible_queries:
        try:
            keyword_results = index.query(id="", top_k=50, include_metadata=True)
            news_matches = keyword_results.get("matches", [])
            filtered_news = [news for news in news_matches if q in news["metadata"]["content"]]

            if filtered_news:
                all_results.extend(filtered_news[:5])
        except Exception as e:
            print(f"Metadata filtering error: {e}")

    if all_results:
        return all_results, cleaned_query, translated_query

    query_embedding = get_embedding(cleaned_query)
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"], cleaned_query, translated_query

# Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4CAF50;
            padding: 10px;
            border-radius: 8px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: none;
            width: 100%;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .news-card {
            background-color: ##e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        .highlight {
            background-color: green;
            padding: 2px 5px;
            border-radius: 3px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    "<h1 style='text-align: center;'>📰 Gujarati News Search Bot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 16px;'>"
    "Find the latest Gujarati news in seconds! Enter your query in English or Gujarati, "
    "and this bot will fetch the most relevant articles for you."
    "</p>",
    unsafe_allow_html=True
)

# Input Section
user_query = st.text_input("🔎 Enter your query (English or Gujarati):")

if st.button("Search News"):
    if user_query:
        results, cleaned_query, translated_query = search_news(user_query)

        # Display translated query
        st.markdown(f"**🔑 Search Keywords:** `{cleaned_query}`")
        if translated_query and translated_query != cleaned_query:
            st.markdown(f"**🌐 Gujarati Translation:** `{translated_query}` 🇮🇳")

        # Display Results
        if results:
            for news in results:
                metadata = news["metadata"]

                highlighted_title = highlight_keywords(metadata["title"], translated_query)
                highlighted_content = highlight_keywords(metadata["content"], translated_query)

                st.markdown(
                    f"""
                    <div class="news-card">
                        <h3>{highlighted_title}</h3>
                        <p><strong>📅 Date:</strong> {metadata['date']}</p>
                        <p>{highlighted_content}</p>
                        <p><a href="{metadata['link']}" target="_blank">🔗 Read More</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ No news found matching your query.")
