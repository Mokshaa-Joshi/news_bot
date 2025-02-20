import streamlit as st
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from pinecone import Pinecone
from deep_translator import GoogleTranslator
import openai
from dotenv import load_dotenv

# 🔹 Fix: Manually Download NLTK Resources for Streamlit Cloud
nltk_resources = ["punkt", "stopwords", "averaged_perceptron_tagger"]
for resource in nltk_resources:
    nltk.download(resource, download_dir="/usr/local/nltk_data")  # Force specific directory

# Set NLTK data path (Fix for Streamlit Cloud)
nltk.data.path.append("/usr/local/nltk_data")

# 🔹 Load API keys securely
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("newsbot")

# 🔹 Extract important keywords using NLP
def extract_keywords(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words("english")]  # Remove stopwords
    
    # Extract only proper nouns and nouns (important keywords)
    keywords = [word for word, tag in pos_tag(words) if tag in ["NN", "NNS", "NNP", "NNPS"]]
    return keywords  # List of extracted keywords

# 🔹 Translate input to Gujarati if needed
def translate_to_gujarati(text):
    if re.search(r'[a-zA-Z]', text):  # If input contains English letters
        return GoogleTranslator(source='en', target='gu').translate(text)
    return text  # Already in Gujarati

# 🔹 Generate query embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 🔹 Highlight multiple keywords in text
def highlight_keywords(text, keywords):
    if not text or not keywords:
        return text
    
    # Create regex pattern for multiple words
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    
    # Highlight words
    highlighted_text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    
    return highlighted_text

# 🔹 Search news using keyword filtering and vector search
def search_news(query):
    # Extract important keywords from query
    keywords = extract_keywords(query)
    
    if not keywords:
        return [], ""  # No valid keywords found

    translated_keywords = [translate_to_gujarati(word) for word in keywords]
    
    # Use both English and Gujarati queries
    possible_queries = keywords + translated_keywords
    all_results = []
    
    for q in possible_queries:
        try:
            # Fetch results from Pinecone
            keyword_results = index.query(id="", top_k=50, include_metadata=True)
            news_matches = keyword_results.get("matches", [])

            # Filter manually for keyword presence
            filtered_news = [news for news in news_matches if q in news["metadata"]["content"]]

            if filtered_news:
                all_results.extend(filtered_news[:5])  # Limit to top 5
        except Exception as e:
            print(f"Metadata filtering error: {e}")

    # If keyword search has results, return them
    if all_results:
        return all_results, keywords

    # If no keyword match, fall back to vector search
    query_embedding = get_embedding(" ".join(keywords))  # Use extracted keywords
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"], keywords

# 🔹 Streamlit UI
st.title("Gujarati News Search 📰")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Search news using Keyword + Vector Search
        results, extracted_keywords = search_news(user_query)
        
        # Display results
        if results:
            for news in results:
                metadata = news["metadata"]
                
                # Highlight extracted keywords in title & content
                highlighted_title = highlight_keywords(metadata["title"], extracted_keywords)
                highlighted_content = highlight_keywords(metadata["content"], extracted_keywords)

                st.markdown(f"### {highlighted_title}", unsafe_allow_html=True)
                st.write(f"**Date:** {metadata['date']}")
                st.write(f"**[Read More]({metadata['link']})**")
                st.markdown(highlighted_content, unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.write("No news found matching your query.")
