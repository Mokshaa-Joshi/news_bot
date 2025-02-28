import streamlit as st
import os
import re
import time
import pinecone
import openai
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (for embeddings)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("newsbot")

STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

# Newspaper options
NEWSPAPER_OPTIONS = {
    "Gujarat Samachar": "gujarat_samachar",
    "Divya Bhaskar": "divya_bhaskar",
    "Sandesh": "sandesh"
}

def extract_keywords(text):
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
    """ Generate text embeddings using OpenAI """
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def highlight_keywords(text, keywords):
    """ Highlights keywords in text using HTML markup """
    if not text or not keywords:
        return text
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

def search_news(query, namespace):
    """ Searches news articles using Pinecone vector search """
    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)
    query_embedding = get_embedding(cleaned_query)
    vector_results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
    return vector_results["matches"], cleaned_query, translated_query

# Streamlit UI Configuration
st.set_page_config(page_title="Gujarati News Chatbot", page_icon="📰", layout="wide")

st.markdown("""
    <style>
        .chat-container {
            background-color: #f0f2f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-bubble {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>📰 Gujarati News Chatbot</h1>
    <p style='text-align: center;'>Chat with our bot to get the latest news updates in Gujarati.</p>
    """, unsafe_allow_html=True)

# Chatbot UI
selected_newspaper = st.selectbox("🗞️ Select Newspaper:", list(NEWSPAPER_OPTIONS.keys()))
st.write("💬 Type your query below:")
chat_input = st.text_input("You:", placeholder="Enter your query here...")

if st.button("Search News"):
    if chat_input:
        with st.spinner("Fetching news... Please wait."):
            time.sleep(1)  # Simulating processing delay
            results, cleaned_query, translated_query = search_news(chat_input, NEWSPAPER_OPTIONS[selected_newspaper])
        
        st.markdown(f"<div class='chat-bubble'><strong>🧠 Bot:</strong> Searching news for '{chat_input}'...</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><strong>🔑 Keywords:</strong> {cleaned_query}</div>", unsafe_allow_html=True)
        if translated_query and translated_query != cleaned_query:
            st.markdown(f"<div class='chat-bubble'><strong>🌐 Gujarati Translation:</strong> {translated_query} 🇮🇳</div>", unsafe_allow_html=True)

        if results:
            for news in results:
                metadata = news["metadata"]
                highlighted_title = highlight_keywords(metadata["title"], translated_query)
                highlighted_content = highlight_keywords(metadata["content"], translated_query)
                
                st.markdown(f"""
                <div class="chat-container">
                    <h3>{highlighted_title}</h3>
                    <p><strong>📅 Date:</strong> {metadata['date']}</p>
                    <p>{highlighted_content}</p>
                """, unsafe_allow_html=True)
                
                if "link" in metadata and metadata["link"]:
                    st.markdown(f"""
                    <p><a href="{metadata['link']}" target="_blank" style="background-color: #333333; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px; font-size: 14px;">🔗 Read More</a></p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='chat-bubble' style='background-color: red;'>⚠️ No news found matching your query.</div>", unsafe_allow_html=True)
