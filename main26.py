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
index = pc.Index("news")

STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

# Map user selection to Pinecone namespace
NEWSPAPER_NAMESPACE = {
    "Gujarat Samachar": "gujarat_samachar",
    "Divya Bhaskar": "divya_baskar",
    "Sandesh": "sandesh",
}

def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(keywords)

def translate_to_gujarati(text):
    """ Translates text to Gujarati sentence-wise for better accuracy """
    try:
        sentences = text.split(". ")
        translated_sentences = [GoogleTranslator(source='auto', target='gu').translate(sentence) for sentence in sentences]
        return ". ".join(translated_sentences)
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

def merge_article_chunks(chunks):
    """ Merges all content chunks of an article into a single string in the correct order """
    sorted_chunks = sorted(chunks, key=lambda x: x["metadata"]["chunk_index"])
    return " ".join([chunk["metadata"]["content_chunk"] for chunk in sorted_chunks])

def search_news(query, newspaper):
    """ Searches news articles using Pinecone vector search """
    namespace = NEWSPAPER_NAMESPACE[newspaper]

    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)

    # Generate embeddings for both original and translated query for better accuracy
    original_embedding = get_embedding(cleaned_query)
    translated_embedding = get_embedding(translated_query)

    # Perform vector search using both embeddings
    vector_results1 = index.query(vector=original_embedding, top_k=5, include_metadata=True, namespace=namespace)
    vector_results2 = index.query(vector=translated_embedding, top_k=5, include_metadata=True, namespace=namespace)

    # Combine results
    all_results = vector_results1["matches"] + vector_results2["matches"]

    # Group articles by title
    articles = {}
    for result in all_results:
        metadata = result["metadata"]
        title = metadata["title"]

        if title not in articles:
            articles[title] = []

        articles[title].append(result)

    # Merge chunks for each article
    merged_articles = []
    for title, chunks in articles.items():
        full_content = merge_article_chunks(chunks)
        metadata = chunks[0]["metadata"]  # Use first chunk's metadata for link and date
        merged_articles.append({
            "title": title,
            "content": full_content,
            "date": metadata["date"],
            "link": metadata.get("link", "No Link Available"),  # Handle missing 'link'
            "source": newspaper  # Store the newspaper name
        })

    return merged_articles, cleaned_query, translated_query

def summarize_and_rerank(articles, query):
    """ Uses OpenAI GPT-4 to re-rank and summarize search results. """
    if not articles:
        return "No relevant articles found."

    formatted_articles = "\n\n".join([
        f"Title: {article['title']}\n"
        f"Date: {article['date']}\n"
        f"Content: {article['content']}\n"
        f"Source: {article['source']}\n"
        f"Link: {article['link']}"
        for article in articles
    ])

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Summarize and rank the following news articles based on relevance to the query."},
            {"role": "user", "content": f"Query: {query}\n\n{formatted_articles}"}
        ]
    )

    return response.choices[0].message.content.strip()

# Streamlit UI Configuration
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align: center;'>📰 Gujarati News Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your query in English or Gujarati and get the latest news instantly.</p>", unsafe_allow_html=True)

# Newspaper Selection
selected_newspaper = st.selectbox("🗞 Select Newspaper:", list(NEWSPAPER_NAMESPACE.keys()))

user_query = st.text_input("🔎 Enter your query (English or Gujarati):")

if st.button("Search News"):
    if user_query:
        with st.spinner("Fetching news... Please wait."):
            time.sleep(1)  # Simulating processing delay
            results, cleaned_query, translated_query = search_news(user_query, selected_newspaper)

        st.markdown(f"**🔑 Search Keywords:** `{cleaned_query}`")
        if translated_query and translated_query != cleaned_query:
            st.markdown(f"**🌐 Gujarati Translation:** `{translated_query}` 🇮🇳")

        if results:
            summary = summarize_and_rerank(results, user_query)
            st.markdown(f"### 📌 Summary of Results:\n{summary}")

            for news in results:
                highlighted_title = highlight_keywords(news["title"], translated_query)
                highlighted_content = highlight_keywords(news["content"], translated_query)

                st.markdown(f"""
                <div style="background-color: #d9e2ec; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 15px; color: black;">
                    <h3>{highlighted_title}</h3>
                    <p><strong>📅 Date:</strong> {news['date']}</p>
                    <p><strong>📰 Source:</strong> {news['source']}</p>
                    <p>{highlighted_content}</p>
                    <p><a href="{news['link']}" target="_blank" style="display: inline-block; padding: 5px 10px; background-color: #333333; color: white !important; text-decoration: none; border-radius: 5px; font-size: 14px;">🔗 Read More</a></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No news found matching your query.")
