import streamlit as st
import os
import re
import pinecone
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

# Load MiniLM Embedding Model (384 dimensions)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Stopwords for keyword extraction
STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a", "from"}

# Newspaper Namespaces
NEWSPAPER_NAMESPACES = {
    "Gujarat Samachar": "gujarat_samachar",
    "Divya Bhaskar": "divya_bhaskar",
    "Sandesh": "sandesh"
}

def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(keywords)

def translate_text(text, target_lang='gu'):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
    return text

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def highlight_keywords(text, keywords):
    if not text or not keywords:
        return text
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

def search_news(query, newspaper):
    cleaned_query = extract_keywords(query)
    translated_query = translate_text(cleaned_query)
    query_embedding = get_embedding(cleaned_query)

    results = index.query(vector=query_embedding, top_k=5, namespace=newspaper, include_metadata=True)

    articles = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        title, date, link = metadata["title"], metadata["date"], metadata.get("link", "")
        
        if title not in articles:
            full_chunks = index.query(
                vector=[0] * 384,
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

def summarize_and_rerank(articles, query):
    """ Uses OpenAI GPT-4 to re-rank and summarize search results. """
    if not articles:
        return "No relevant articles found."

    formatted_articles = "\n\n".join([f"Title: {title}\nContent: {details['content']}" for title, details in articles.items()])
    
    system_prompt = "You are an AI assistant that ranks and summarizes news articles based on relevance to a query."
    user_prompt = f"Query: {query}\n\nArticles:\n{formatted_articles}\n\nRank these articles by relevance and summarize them."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="wide")

st.markdown("""
    <style>
        .big-title { text-align: center; font-size: 32px; font-weight: bold; }
        .sub-title { text-align: center; font-size: 18px; color: gray; }
        .article-container { background-color:#f9f9f9; padding:15px; border-radius:8px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .highlight { background-color: yellow; color: black; }
    </style>
    <h1 class="big-title">📰 Gujarati News Bot</h1>
    <p class="sub-title">Ask about news articles in English or Gujarati.</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🏆 Choose Your Newspaper")
    newspaper_choice = st.radio("", ["Gujarat Samachar", "Divya Bhaskar", "Sandesh"])
    st.markdown("---")
    st.markdown("📌 This bot fetches news from leading Gujarati newspapers using AI-powered search.")

# Main UI
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input("🔎 Ask me about news articles...", placeholder="Type your query here")
    if st.button("Search News"):
        if user_query:
            newspaper = NEWSPAPER_NAMESPACES[newspaper_choice]

            with st.spinner("🔍 Searching news..."):
                articles, cleaned_query, translated_query = search_news(user_query, newspaper)
                summarized_results = summarize_and_rerank(articles, user_query)

            st.markdown(f"**🔑 Keywords Used:** `{cleaned_query}`")
            if translated_query and translated_query != cleaned_query:
                st.markdown(f"**🌐 Gujarati Translation:** `{translated_query}` 🇮🇳")

            if articles:
                for title, details in articles.items():
                    highlighted_title = highlight_keywords(title, translated_query)
                    highlighted_content = highlight_keywords(details["content"], translated_query)

                    st.markdown(f"""
                    <div class="article-container">
                        <h3>{highlighted_title}</h3>
                        <p><strong>📅 Date:</strong> {details['date']}</p>
                        <p><strong>🗞 Newspaper:</strong> {details['newspaper']}</p>
                        <p>{highlighted_content}</p>
                        <p><a href="{details['link']}" target="_blank" style="background-color: #333; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px;">🔗 Read More</a></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("## 🔥 AI-Powered Summary & Reranking")
                st.markdown(f"<p>{summarized_results}</p>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No matching news articles found.")
