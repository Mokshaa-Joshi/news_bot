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

# Debugging: Check connection
try:
    db.list_collection_names()  # This will fail if the connection is incorrect
except Exception as e:
    st.error(f"MongoDB Connection Error: {e}")
    st.stop()

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

        # Debugging: Show extracted date
        if date_filter:
            st.write(f"🔍 Searching for news on: {date_filter}")

        # Convert query to regex
        regex_pattern = re.compile(re.escape(query), re.IGNORECASE)

        # Build MongoDB filter
        mongo_query = {"$or": [{"title": regex_pattern}, {"content": regex_pattern}]}
        if date_filter:
            mongo_query["date"] = {"$regex": re.escape(date_filter)}

        # Debugging: Show query being used
        st.write(f"🛠 MongoDB Query: {mongo_query}")

        # Fetch results
        results = list(collection.find(mongo_query))  # Convert cursor to list

        # Debugging: Check if results exist
        if not results:
            st.write("❌ No news found.")
        else:
            for news in results:
                st.subheader(news["title"])
                st.write(f"🆔 **ID:** {news['_id']}")
                st.write(f"📅 **Date:** {news['date']}")
                st.write(f"🔗 [Read More]({news['link']})")
                st.write(f"📰 **Content:**\n {news['content']}")
                st.markdown("---")
    else:
        st.warning("Please enter a search query.")
