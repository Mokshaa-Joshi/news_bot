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

# Function to highlight keywords in text
def highlight_keywords(text, keyword):
    if not text or not keyword:
        return text  # Return original text if empty
    
    # Use word boundaries (\b) for exact word matching (case-insensitive)
    pattern = re.compile(rf"({re.escape(keyword)})", re.IGNORECASE)
    
    # Replace matched keywords with highlighted version
    highlighted_text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    
    return highlighted_text

# Function to search news using keyword filtering and vector search
def search_news(query):
    # Translate query to Gujarati (if needed)
    translated_query = translate_to_gujarati(query)

    # **1. Try metadata filtering (search by keyword in content)**
    try:
        keyword_results = index.query(
            id="",
            top_k=50,  # Retrieve more results to improve keyword matching
            include_metadata=True
        )
        all_news = keyword_results.get("matches", [])
        
        # **Perform keyword search manually in retrieved content**
        filtered_news = [
            news for news in all_news if translated_query in news["metadata"]["content"]
        ]
    except Exception as e:
        print(f"Metadata filtering error: {e}")
        filtered_news = []

    # **If keyword search finds results, return them first**
    if filtered_news:
        return filtered_news[:5], translated_query  # Return top 5 matches + translated query

    # **2. If no keyword match, fall back to vector search**
    query_embedding = get_embedding(translated_query)
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"], translated_query

# Streamlit UI
st.title("Gujarati News Search")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Search news using Keyword + Vector Search
        results, translated_query = search_news(user_query)
        
        # Display results
        if results:
            for news in results:
                metadata = news["metadata"]
                
                # Highlight translated query in title & content
                highlighted_title = highlight_keywords(metadata["title"], translated_query)
                highlighted_content = highlight_keywords(metadata["content"], translated_query)

                st.markdown(f"### {highlighted_title}", unsafe_allow_html=True)
                st.write(f"**Date:** {metadata['date']}")
                st.write(f"**[Read More]({metadata['link']})**")
                st.markdown(highlighted_content, unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.write("No news found matching your query.")
