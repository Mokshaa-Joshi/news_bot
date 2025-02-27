import streamlit as st
import os
import re
import time
import pinecone
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer  # ✅ Use MiniLM for embeddings

# 🔐 Load API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🌲 Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

# 🚀 Load MiniLM Embedding Model (384 dimensions)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🚫 Stopwords for keyword extraction
STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a", "from"}

# 📌 Define Newspaper Namespaces
NEWSPAPER_NAMESPACES = {
    "gujarat samachar": "gujarat_samachar",
    "divya bhaskar": "divya_bhaskar",
    "sandesh": "sandesh"
}

def extract_keywords(text):
    """ Extracts meaningful keywords from the query """
    words = text.split()
    keywords = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(keywords)

def translate_to_gujarati(text):
    """ Translates text to Gujarati using DeepTranslate """
    try:
        return GoogleTranslator(source='auto', target='gu').translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
    return text  # Fallback to original text

def get_embedding(text):
    """ Generate text embeddings using Sentence Transformers (384 dimensions) """
    return embedding_model.encode(text).tolist()

def highlight_keywords(text, keywords):
    """ Highlights keywords in text using HTML """
    if not text or not keywords:
        return text
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

def search_news(query, newspaper):
    """ Searches news articles using Pinecone vector search with improved keyword matching """
    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)
    query_embedding = get_embedding(cleaned_query)

    # 🔍 Query Pinecone with correct namespace (384-dimension vector)
    results = index.query(vector=query_embedding, top_k=5, namespace=newspaper, include_metadata=True)

    articles = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        title, date, link = metadata["title"], metadata["date"], metadata.get("link", "")
        
        if title not in articles:
            full_chunks = index.query(
                vector=[0] * 384,  # ✅ Dummy vector (384-dimension)
                top_k=100,
                namespace=newspaper,
                include_metadata=True,
                filter={"title": title}
            )

            merged_content = {chunk["metadata"]["chunk_index"]: chunk["metadata"]["content_chunk"] for chunk in full_chunks["matches"]}
            full_text = " ".join([merged_content[i] for i in sorted(merged_content)])

            articles[title] = {
                "date": date,
                "newspaper": newspaper.replace("_", " ").title(),
                "content": full_text,
                "link": link
            }

    return articles, cleaned_query, translated_query

# 🎨 Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align: center;'>📰 Gujarati News Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask about news articles in English or Gujarati.</p>", unsafe_allow_html=True)

user_query = st.text_input("🔎 Ask me about news articles...")

if st.button("Search News"):
    if user_query:
        # Detect newspaper
        newspaper = None
        for paper in NEWSPAPER_NAMESPACES.keys():
            if paper in user_query.lower():
                newspaper = NEWSPAPER_NAMESPACES[paper]
                break

        if not newspaper:
            st.warning("⚠️ Please mention a newspaper (Gujarat Samachar, Divya Bhaskar, Sandesh).")
        else:
            with st.spinner("Searching news..."):
                articles, cleaned_query, translated_query = search_news(user_query, newspaper)

            st.markdown(f"**🔑 Keywords Used:** `{cleaned_query}`")
            if translated_query and translated_query != cleaned_query:
                st.markdown(f"**🌐 Gujarati Translation:** `{translated_query}` 🇮🇳")

            # 📰 Display results
            if articles:
                for title, details in articles.items():
                    highlighted_title = highlight_keywords(title, translated_query)
                    highlighted_content = highlight_keywords(details["content"], translated_query)

                    st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:8px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
                        <h3>{highlighted_title}</h3>
                        <p><strong>📅 Date:</strong> {details['date']}</p>
                        <p><strong>🗞 Newspaper:</strong> {details['newspaper']}</p>
                        <p>{highlighted_content}</p>
                        <p><a href="{details['link']}" target="_blank" style="background-color: #333; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px;">🔗 Read More</a></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No matching news articles found.")
