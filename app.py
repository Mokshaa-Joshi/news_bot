import os
import openai
import streamlit as st
from pinecone import Pinecone
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# Check if index exists
if index_name not in pc.list_indexes().names():
    st.error(f"Index '{index_name}' not found in Pinecone. Please check your configuration.")
    st.stop()

index = pc.Index(index_name)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

st.title("📰 Gujarati News Bot")
st.write("Enter your news query in **English or Gujarati** and get relevant news articles.")

# User Input
user_query = st.text_input("🔎 Search for news:")

def extract_date(query):
    """Extracts date (if any) from user input."""
    date_pattern = r"\d{2}-\d{2}-\d{4}"  # Pattern for DD-MM-YYYY format
    match = re.search(date_pattern, query)
    return match.group(0) if match else None

if user_query:
    # Extract date from query (if present)
    news_date = extract_date(user_query)

    # Detect language
    translated_text = GoogleTranslator(source="auto", target="en").translate(user_query)
    detected_lang = "gu" if translated_text != user_query else "en"

    # Translate to Gujarati if input is English
    if detected_lang == "en":
        user_query = GoogleTranslator(source="en", target="gu").translate(user_query)

    # Remove the date from the query (if found) so embeddings are cleaner
    if news_date:
        clean_query = user_query.replace(news_date, "").strip()
    else:
        clean_query = user_query

    st.write(f"🔄 Searching news for: **{clean_query}**")

    # Convert query to vector using OpenAI embeddings
    embed_response = openai.embeddings.create(input=[clean_query], model="text-embedding-ada-002")
    query_vector = embed_response.data[0].embedding  # Correct vector extraction

    # Debugging: Check vector dimension
    st.write(f"✅ Query Vector Dimension: {len(query_vector)}")

    # Build filter for Pinecone search
    filters = {}
    if news_date:
        filters["date"] = {"$eq": news_date}  # Exact match for the date

    # Search Pinecone for top 3 relevant news articles
    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        namespace="news_data",
        filter=filters  # Apply filter if date is found
    )

    # Debugging: Check if results exist
    st.write(f"✅ Pinecone Results: {search_results}")

    # Display Results
    if "matches" in search_results and search_results["matches"]:
        news_articles = []
        for match in search_results["matches"]:
            metadata = match["metadata"]
            news_articles.append(f"**{metadata.get('title', 'Untitled')}**\n📅 {metadata.get('date', 'Unknown Date')}\n🔗 [Read More]({metadata.get('link', '#')})\n\n{metadata.get('content', 'No content available')[:300]}...")

        # Format response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Gujarati news bot. Format news concisely and in Gujarati."},
                {"role": "user", "content": "\n\n".join(news_articles)}
            ]
        )
        st.markdown(response.choices[0].message.content)
    else:
        st.warning("❌ No relevant news found. Try a different query.")
