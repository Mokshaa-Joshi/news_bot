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

# Function to check for an exact match using metadata filtering
def exact_match_search(query):
    try:
        results = index.query(
            top_k=5,
            include_metadata=True,
            filter={"title": {"$eq": query}}  # Exact title match
        )
        return results["matches"]
    except Exception as e:
        st.error(f"Metadata search error: {e}")
        return []

# Function to search news using vector search (fallback)
def semantic_search(query):
    query_embedding = get_embedding(query)
    try:
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        return results["matches"]
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return []

# Function to perform hybrid search
def search_news(query):
    # Try exact match first
    exact_results = exact_match_search(query)
    if exact_results:
        return exact_results  # If found, return exact matches
    
    # If no exact match, fall back to vector search
    return semantic_search(query)

# Streamlit UI
st.title("Gujarati News Search")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Translate input to Gujarati
        translated_query = translate_to_gujarati(user_query)
        
        # Search news in Pinecone (Hybrid Approach)
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
