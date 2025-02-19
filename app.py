import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "news_data"
COLLECTION_NAME = "dd_news_articles"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Streamlit UI
st.title("Gujarati News Search Bot 📰")
st.write("Enter a keyword and date to find relevant news.")

# User input
query = st.text_input("Search News (e.g., 'ISRO on 29-01-2025')")

if st.button("Search"):
    if query:
        # Extract date from the query if present
        date_match = re.search(r"\d{2}-\d{2}-\d{4}", query)
        date_filter = date_match.group(0) if date_match else None

        # Convert query to regex for better matching
        regex_pattern = re.compile(query, re.IGNORECASE)

        # Build MongoDB filter
        mongo_query = {"$or": [{"title": regex_pattern}, {"content": regex_pattern}]}
        if date_filter:
            mongo_query["date"] = {"$regex": date_filter}

        # Fetch results
        results = collection.find(mongo_query)

        # Display results
        if results:
            for news in results:
                st.subheader(news["title"])
                st.write(f"📅 **Date:** {news['date']}")
                st.write(f"🔗 [Read More]({news['link']})")
                st.write(f"📰 **Content:**\n {news['content']}")
                st.markdown("---")
        else:
            st.write("❌ No news found.")
    else:
        st.warning("Please enter a search query.")
