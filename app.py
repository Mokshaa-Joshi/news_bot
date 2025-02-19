import os
import streamlit as st
import pinecone
import openai
import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def extract_keywords(prompt):
    """Extract keywords using OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract keywords from the given prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.split(', ')

def standardize_date(user_date):
    """Convert various date formats into a standard YYYY-MM-DD format."""
    try:
        parsed_date = datetime.datetime.strptime(user_date, "%Y-%m-%d")
    except ValueError:
        try:
            parsed_date = datetime.datetime.strptime(user_date, "%d-%m-%Y")
        except ValueError:
            try:
                parsed_date = datetime.datetime.strptime(user_date, "%d/%m/%Y")
            except ValueError:
                parsed_date = None
    
    return parsed_date.strftime("%Y-%m-%d") if parsed_date else None

def search_news(prompt, user_date=None):
    """Search news articles in Pinecone using keyword matching."""
    translated_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
    keywords = extract_keywords(translated_prompt)
    query_embedding = client.embeddings.create(input=[translated_prompt], model="text-embedding-ada-002").data[0].embedding
    
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    
    standardized_date = standardize_date(user_date) if user_date else None
    
    for match in results["matches"]:
        metadata = match["metadata"]
        if all(keyword.lower() in metadata["title"].lower() or keyword.lower() in metadata["content"].lower() for keyword in keywords):
            if user_date is None or metadata["date"] == standardized_date:
                return {
                    "title": metadata["title"],
                    "date": metadata["date"],
                    "link": metadata["link"],
                    "content": metadata["content"]
                }
    return None

# Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", layout="wide")
st.title("📰 Gujarati News Bot")

# Input fields
query = st.text_input("Enter your query in Gujarati:")
date_input = st.text_input("Enter date (optional, any format):")

if st.button("Search News"):
    if query:
        result = search_news(query, date_input)
        if result:
            st.subheader(result["title"])
            st.write(f"📅 **Date:** {result["date"]}")
            st.write(f"🔗 [Read more]({result['link']})")
            st.write("---")
            st.write(result["content"])
        else:
            st.warning("No matching news found.")
    else:
        st.error("Please enter a query.")
