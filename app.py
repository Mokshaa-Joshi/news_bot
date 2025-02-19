import os
import openai
import streamlit as st
from pinecone import Pinecone
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

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

if user_query:
    # Detect language and translate if needed
    translated_text = GoogleTranslator(source="auto", target="en").translate(user_query)
    detected_lang = "gu" if translated_text != user_query else "en"

    if detected_lang == "en":
        user_query = GoogleTranslator(source="en", target="gu").translate(user_query)

    st.write(f"🔄 Searching news for: **{user_query}**")

    # Generate embeddings for query
    embed_response = openai.embeddings.create(input=[user_query], model="text-embedding-ada-002")
    query_vector = embed_response.data[0].embedding  # Extract vector

    # Perform vector search (Hybrid: Title & Content relevance)
    search_results = index.query(
        vector=query_vector,  
        top_k=5,
        include_metadata=True,
        filter={  # Only filter metadata fields that exist in Pinecone
            "date": {"$eq": "2025-02-19"}  # Example: Filter by today's date (modify as needed)
        }
    )

    # Display Results
    if search_results.get("matches"):
        news_articles = []
        for match in search_results["matches"]:
            metadata = match["metadata"]
            news_articles.append(f"**{metadata['title']}**\n📅 {metadata['date']}\n🔗 [Read More]({metadata['link']})\n\n{metadata['content'][:300]}...")

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
