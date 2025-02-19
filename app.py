import os
import streamlit as st
import pinecone
import openai
from dateutil import parser
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")  # Change environment if needed
index = pinecone.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def extract_keywords(prompt):
    """Extract keywords using OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract the most relevant keywords from the given query."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.lower().split(", ")

def standardize_date(user_date):
    """Convert various date formats into a standard YYYY-MM-DD format."""
    try:
        parsed_date = parser.parse(user_date)
        return parsed_date.strftime("%Y-%m-%d")
    except Exception:
        return None

def search_news(prompt, user_date=None):
    """Search exact matching news articles in Pinecone using metadata filtering."""
    translated_prompt = GoogleTranslator(source="auto", target="en").translate(prompt)
    keywords = extract_keywords(translated_prompt)

    # Standardize the date format
    standardized_date = standardize_date(user_date) if user_date else None

    # Construct metadata filter query
    metadata_filter = {"$and": []}
    
    for keyword in keywords:
        metadata_filter["$and"].append({
            "$or": [
                {"title": {"$contains": keyword}},
                {"content": {"$contains": keyword}}
            ]
        })
    
    if standardized_date:
        metadata_filter["$and"].append({"date": standardized_date})

    # Query Pinecone with metadata filter
    results = index.query(filter=metadata_filter, top_k=10, include_metadata=True)

    for match in results["matches"]:
        metadata = match["metadata"]
        return {
            "title": metadata["title"],
            "date": metadata["date"],
            "link": metadata["link"],
            "content": metadata["content"],
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
            # Display formatted output
            st.subheader(result["title"])
            st.markdown(f"**📅 Date:** `{result['date']}`")
            st.markdown(f"**🔗 Read More:** [Click Here]({result['link']})")
            st.markdown("---")
            st.markdown(f"**📝 Content:**\n{result['content']}")
        else:
            st.warning("❌ No matching news found.")
    else:
        st.error("⚠️ Please enter a query.")
