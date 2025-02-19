import streamlit as st
import pinecone
import openai
import os
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index = pinecone.Index("news-index")

# Function to translate input to Gujarati if needed
def translate_to_gujarati(text):
    return GoogleTranslator(source='auto', target='gu').translate(text)

# Function to generate query embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    return response["data"][0]["embedding"]

# Search Pinecone for news articles
def search_news(query):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return results["matches"]

# Streamlit UI
st.title("Gujarati News Search")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Translate input if in English
        translated_query = translate_to_gujarati(user_query)
        
        # Search news in Pinecone
        results = search_news(translated_query)
        
        # Display results
        if results:
            for news in results:
                metadata = news["metadata"]
                st.subheader(metadata["title"])
                st.write(f"**Date:** {metadata['date']}")
                st.write(f"**[Read More]({metadata['link']})**")
                st.write(metadata["content"])
                st.markdown("---")
        else:
            st.write("No news found matching your query.")
