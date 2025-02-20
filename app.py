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
    if re.search(r'[a-zA-Z]', text):  # If input contains English letters
        return GoogleTranslator(source='en', target='gu').translate(text)
    return text  # Already in Gujarati

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
    
    # Split query into words for better highlighting
    words = keywords.split()
    
    # Create regex pattern for multiple words
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    
    # Highlight words
    highlighted_text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    
    return highlighted_text

# Function to search news using keyword filtering and vector search
def search_news(query):
    # Translate query to Gujarati
    translated_query = translate_to_gujarati(query)

    # Use both English and Gujarati queries
    possible_queries = [query, translated_query]
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
        return all_results, translated_query

    # If no keyword match, fall back to vector search
    query_embedding = get_embedding(query)  # Use English for embeddings
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"], translated_query

# Streamlit UI
st.title("Gujarati News Search 📰")

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
