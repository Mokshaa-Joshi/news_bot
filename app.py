import streamlit as st
import os
import re
import spacy
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

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract important keywords (NER + noun phrases)
def extract_keywords(text):
    doc = nlp(text)
    keywords = set()

    # Extract Named Entities (e.g., ISRO, Prime Minister)
    for ent in doc.ents:
        keywords.add(ent.text)

    # Extract Noun Phrases (e.g., "space agency", "government policy")
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text)

    return list(keywords)

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
    
    # Create regex pattern for multiple words
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    
    # Highlight words
    highlighted_text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    
    return highlighted_text

# Function to search news using keyword filtering and vector search
def search_news(query):
    # Extract important keywords
    important_keywords = extract_keywords(query)
    
    # Translate keywords to Gujarati
    translated_keywords = [translate_to_gujarati(word) for word in important_keywords]

    # Use both English and Gujarati keywords
    possible_queries = important_keywords + translated_keywords
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
        return all_results, important_keywords

    # If no keyword match, fall back to vector search
    query_embedding = get_embedding(" ".join(important_keywords))  # Use extracted keywords
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return vector_results["matches"], important_keywords

# Streamlit UI
st.title("Gujarati News Search 📰")

user_query = st.text_input("Enter your query (English or Gujarati):")

if st.button("Search"):
    if user_query:
        # Search news using extracted keywords
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
