import streamlit as st
import os
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

# Function to translate input to Gujarati if needed
def translate_to_gujarati(text):
    return GoogleTranslator(source='auto', target='gu').translate(text)

# Function to generate query embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to search news in Pinecone (Keyword in Content + Vector Search)
def search_news(query):
    # Translate query to Gujarati (if needed)
    translated_query = translate_to_gujarati(query)

    # **1. Try metadata filtering (search by keyword in content)**
    try:
        keyword_results = index.query(
            id="",  # Fix: Use a valid ID placeholder (empty string) instead of None
            top_k=5,
            include_metadata=True,
            filter={
                "content": {"$regex": translated_query}  # Search for exact keyword match
            }
        )
    except Exception as e:
        print(f"Metadata filtering error: {e}")
        keyword_results = {"matches": []}

    # **If keyword search finds results, return them first**
    if keyword_results["matches"]:
        return keyword_results["matches"]

    # **2. If no keyword match, fall back to vector search**
    query_embedding = get_embedding(translated_query)
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"]

# Streamlit UI
st.title("Gujarati News Search")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Search news using Keyword + Vector Search
        results = search_news(user_query)
        
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
