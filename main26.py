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

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news3")

STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

# Newspaper options
NEWSPAPER_OPTIONS = {
    "Gujarat Samachar": "gujarat_samachar",
    "Divya Bhaskar": "divya_bhaskar",
    "Sandesh": "sandesh"
}

def extract_keywords(text):
    words = text.split()
    keywords = [word.lower() for word in words if word.lower() not in STOPWORDS]
    return keywords

def translate_to_gujarati(text):
    """ Translates text to Gujarati using DeepTranslate """
    try:
        return GoogleTranslator(source='auto', target='gu').translate(f'"{text}"')
    except Exception as e:
        st.error(f"Translation error: {e}")
    return text  # Fallback to original text

def filter_news_by_title(query, namespace):
    """ Fetches all news articles and filters them based on keyword matches in the title """
    keywords = extract_keywords(query)
    translated_keywords = extract_keywords(translate_to_gujarati(query))

    # Fetch all records from Pinecone
    news_records = index.query(vector=[0]*1536, top_k=100, include_metadata=True, namespace=namespace)["matches"]

    # Filter results based on title keywords
    filtered_news = [
        news for news in news_records
        if any(keyword in news["metadata"]["title"].lower() for keyword in keywords + translated_keywords)
    ]
    
    return filtered_news, keywords, translated_keywords

def highlight_keywords(text, keywords):
    """ Highlights keywords in text using HTML markup """
    if not text or not keywords:
        return text
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

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
            background-color: #808080;
            color: white;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>📰 Gujarati News Search Assistant</h1>
    <p style='text-align: center;'>Chat with it to get the latest news updates in Gujarati.</p>
    """, unsafe_allow_html=True)

# Chatbot UI
selected_newspaper = st.selectbox("🗞️ Select Newspaper:", list(NEWSPAPER_OPTIONS.keys()))
st.write("💬 Type your query (in English or Gujarati) below:")
chat_input = st.text_input("You:", placeholder="Enter your query here...")

if st.button("Search News"):
    if chat_input:
        with st.spinner("Fetching news... Please wait."):
            time.sleep(1)  # Simulating processing delay
            results, cleaned_query, translated_query = filter_news_by_title(chat_input, NEWSPAPER_OPTIONS[selected_newspaper])
        
        st.markdown(f"<div class='chat-bubble'><strong>Bot:</strong> Searching news for '{chat_input}'...</div>", unsafe_allow_html=True)
        if translated_query and translated_query != cleaned_query:
            st.markdown(f"<div class='chat-bubble'><strong>Gujarati Translation:</strong> {' '.join(translated_query)} 🇮🇳</div>", unsafe_allow_html=True)

        if results:
            for news in results:
                metadata = news["metadata"]
                highlighted_title = highlight_keywords(metadata["title"], cleaned_query + translated_query)
                highlighted_content = highlight_keywords(metadata["content"], cleaned_query + translated_query)
                
                st.markdown(f"""
                <div class="chat-container">
                    <h3>{highlighted_title}</h3>
                    <p><strong>📅 Date:</strong> {metadata['date']}</p>
                    <p><strong>Source:</strong> {selected_newspaper}</p>
                    <p>{highlighted_content}</p>
                """, unsafe_allow_html=True)
                
                if "link" in metadata and metadata["link"]:
                    st.markdown(f"""
                    <p><a href="{metadata['link']}" target="_blank" style="background-color: #333333; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px; font-size: 14px;">🔗 Read More</a></p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='chat-bubble' style='background-color: red;'>⚠️ No news found matching your query.</div>", unsafe_allow_html=True)
