import os
import pinecone
import openai
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

st.title("📰 Gujarati News Bot")
st.write("Enter your news query in **English or Gujarati** and get relevant news articles.")

# User Input
user_query = st.text_input("🔎 Search for news:")

if user_query:
    # Detect and Translate Input to Gujarati if it's in English
    detected_lang = GoogleTranslator(source="auto", target="gu").detect(user_query)
    if detected_lang != "gu":
        user_query = GoogleTranslator(source="en", target="gu").translate(user_query)

    st.write(f"🔄 Searching news for: **{user_query}**")

    # Convert query to vector using OpenAI embeddings
    embed_response = openai.Embedding.create(input=user_query, model="text-embedding-ada-002")
    query_vector = embed_response["data"][0]["embedding"]

    # Search Pinecone for top 3 relevant news articles
    search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    # Display Results
    if search_results["matches"]:
        news_articles = []
        for match in search_results["matches"]:
            metadata = match["metadata"]
            news_articles.append(f"**{metadata['title']}**\n📅 {metadata['date']}\n🔗 [Read More]({metadata['link']})\n\n{metadata['content'][:300]}...")

        # Format response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Gujarati news bot. Format news concisely and in Gujarati."},
                {"role": "user", "content": "\n\n".join(news_articles)}
            ]
        )
        st.markdown(response["choices"][0]["message"]["content"])
    else:
        st.warning("❌ No relevant news found. Try a different query.")
