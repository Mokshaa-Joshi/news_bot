import streamlit as st
import os
import re
import time
import pinecone
import openai
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# 🔐 Load API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🚀 Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🌲 Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

# 🚫 Stopwords for keyword extraction
STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

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
    """ Generate text embeddings using OpenAI """
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def highlight_keywords(text, keywords):
    """ Highlights keywords in text using HTML """
    if not text or not keywords:
        return text
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

def search_news(query, newspaper, date_filter):
    """ Searches news articles using Pinecone vector search """
    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)
    query_embedding = get_embedding(cleaned_query)

    # 🔍 Query Pinecone
    results = index.query(vector=query_embedding, top_k=10, namespace=newspaper, include_metadata=True)

    # 📌 Fetch Full Articles
    articles = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        title, date, link = metadata["title"], metadata["date"], metadata.get("link", "")

        # Skip non-matching dates
        if date_filter and date != date_filter:
            continue  

        if title not in articles:
            full_chunks = index.query(
                vector=[0] * 1536,  # Dummy vector
                top_k=100,
                namespace=newspaper,
                include_metadata=True,
                filter={"title": title}
            )

            merged_content = {chunk["metadata"]["chunk_index"]: chunk["metadata"]["content_chunk"] for chunk in full_chunks["matches"]}

            articles[title] = {
                "date": date,
                "newspaper": newspaper.replace("_", " ").title(),
                "content": " ".join([merged_content[i] for i in sorted(merged_content)]),
                "link": link
            }

    return articles, cleaned_query, translated_query

# 🎨 Streamlit UI
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align: center;'>📰 Gujarati News Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask about news articles in English or Gujarati.</p>", unsafe_allow_html=True)

# 📜 Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about news articles by keyword, date, and newspaper."}]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🎤 User Input
user_input = st.chat_input("Ask me about news articles...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Extract date & newspaper
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
    date_filter = date_match.group(0) if date_match else None

    newspaper = None
    for paper in NEWSPAPER_NAMESPACES.keys():
        if paper in user_input.lower():
            newspaper = NEWSPAPER_NAMESPACES[paper]
            break

    if not newspaper:
        response = "⚠️ Please mention a newspaper (Gujarat Samachar, Divya Bhaskar, Sandesh)."
    else:
        with st.spinner("Searching news..."):
            articles, cleaned_query, translated_query = search_news(user_input, newspaper, date_filter)

        # Display search details
        st.markdown(f"**🔑 Keywords Used:** `{cleaned_query}`")
        if translated_query and translated_query != cleaned_query:
            st.markdown(f"**🌐 Gujarati Translation:** `{translated_query}` 🇮🇳")

        # 📰 Display results
        if articles:
            response = "✅ **Search Results**"
            for title, details in articles.items():
                highlighted_title = highlight_keywords(title, translated_query)
                highlighted_content = highlight_keywords(details["content"], translated_query)

                with st.expander(f"📌 {highlighted_title}"):
                    st.markdown(f"📅 **Date:** {details['date']}")
                    st.markdown(f"🗞 **Newspaper:** {details['newspaper']}")
                    if details["link"]:
                        st.markdown(f"🔗 [Read More]({details['link']})")
                    st.write(f"📜 **Full Article:**\n{highlighted_content}")
        else:
            response = "❌ No matching news articles found."

    # 💬 Store Response in Chat History
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 💡 Display Response
    with st.chat_message("assistant"):
        st.markdown(response)
