import streamlit as st
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from deep_translator import GoogleTranslator
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("gujarati-news")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_keywords(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo",
        prompt=f"Extract important keywords from this query: {prompt}",
        max_tokens=10
    )
    return response.choices[0].text.strip()

def parse_date(user_date):
    formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(user_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def search_news(query):
    translated_query = GoogleTranslator(source='auto', target='gu').translate(query)
    keywords = extract_keywords(translated_query)
    
    date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', query)
    date_formatted = parse_date(date_match.group()) if date_match else None
    
    results = index.query(keywords, top_k=5, include_metadata=True)
    for result in results["matches"]:
        record = result["metadata"]
        if keywords in record["title"] and (date_formatted is None or date_formatted == record["date"]):
            return record
    return None

def main():
    st.title("Gujarati News Bot 📰")
    user_query = st.text_input("Enter your news query:")
    
    if st.button("Search") and user_query:
        result = search_news(user_query)
        
        if result:
            st.subheader(result["title"])
            st.write(f"**Date:** {result['date']}")
            st.write(f"[Read more]({result['link']})")
            st.write(result["content"])
        else:
            st.error("No exact match found. Try refining your query.")

if __name__ == "__main__":
    main()
