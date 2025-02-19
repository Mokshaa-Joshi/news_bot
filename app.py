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

# Function to extract date from query
def extract_date(query):
    date_pattern = r"\b\d{2}-\d{2}-\d{4}\b"
    match = re.search(date_pattern, query)
    return match.group() if match else None

# Function to generate query embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to search news in Pinecone with hybrid search
def search_news(query):
    # Extract date if mentioned in query
    extracted_date = extract_date(query)
    
    # Remove date from query before translation
    cleaned_query = re.sub(r"\b\d{2}-\d{2}-\d{4}\b", "", query).strip()
    
    # Generate multiple query embeddings (English & Gujarati)
    translated_query = translate_to_gujarati(cleaned_query)
    query_variants = [cleaned_query, translated_query]

    # Get embeddings for both queries
    embeddings = [get_embedding(q) for q in query_variants]

    # Define filter (if date is present)
    filters = {}
    if extracted_date:
        filters["date"] = {"$contains": extracted_date}  # Allows partial matching

    # Perform Pinecone search for both embeddings
    all_results = []
    for emb in embeddings:
        results = index.query(
            vector=emb, 
            top_k=5, 
            include_metadata=True,
            filter=filters if filters else None  # Apply filter only if date is present
        )
        all_results.extend(results["matches"])

    # Remove duplicates & sort by highest similarity
    unique_results = {res["id"]: res for res in all_results}.values()
    sorted_results = sorted(unique_results, key=lambda x: x["score"], reverse=True)

    return sorted_results

# Streamlit UI
st.title("Gujarati News Search")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Search news in Pinecone
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
