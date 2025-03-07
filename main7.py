import streamlit as st
import re
import requests
from langchain.llms import HuggingFacePipeline

def download_articles_from_github(repo_url, filename):
    """Fetches articles from a GitHub raw file URL."""
    url = f"{repo_url}/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None

def load_articles(content, newspaper):
    """Splits raw content into individual articles based on newspaper format."""
    articles = []
    if content:
        content = content.replace("\r\n", "\n")  # Normalize newlines
        if newspaper in ["Gujarat Samachar", "Divya Bhaskar"]:
            articles = content.split("================================================================================")
        elif newspaper == "Sandesh":
            articles = re.split(r"(?=\w{3} \d{1,2}, \d{4} \d{2}:\d{2} (am|pm))", content)
            articles = [article.strip() for article in articles if article.strip()]
    return [article.strip() for article in articles if article.strip()]

def parse_article(article, newspaper):
    """Extracts structured data (title, date, content) from raw article text."""
    article = article.strip().replace("\r\n", "\n")  # Normalize newlines
    if newspaper in ["Gujarat Samachar", "Divya Bhaskar"]:
        match = re.search(r"Title:\s*(.*?)\nDate:\s*(.*?)\nLink:\s*(.*?)\nContent:\s*(.*)", article, re.DOTALL)
        if match:
            return {
                "title": match.group(1).strip(),
                "date": match.group(2).strip(),
                "link": match.group(3).strip(),
                "content": match.group(4).strip()
            }
    elif newspaper == "Sandesh":
        lines = article.split("\n")
        if len(lines) >= 3:
            return {
                "date": lines[0].strip(),
                "title": lines[1].strip(),
                "content": "\n".join(lines[2:]).strip()
            }
    return None

def highlight_keywords(text, keywords):
    """Highlights the keywords in the given text using HTML."""
    for keyword in keywords:
        text = re.sub(f"({re.escape(keyword)})", r'<span style="background-color: yellow; font-weight: bold;">\1</span>', text, flags=re.IGNORECASE)
    return text

def search_articles(articles, query, search_type, newspaper):
    """Searches for keywords in article titles and content based on search type."""
    results = []
    
    if " અને " in query:  # AND search
        query_keywords = query.strip().split(" અને ")
        keyword_pattern = r".*".join(re.escape(k) for k in query_keywords)
    elif " અથવા " in query:  # OR search
        query_keywords = query.strip().split(" અથવા ")
        keyword_pattern = r"|".join(re.escape(k) for k in query_keywords)
    else:
        query_keywords = [query.strip()]
        keyword_pattern = re.escape(query.strip())

    for article in articles:
        parsed_article = parse_article(article, newspaper)
        if parsed_article:
            content_to_search = f"{parsed_article['title']} {parsed_article['content']}".lower()
            
            if search_type == "contains":
                match = re.search(keyword_pattern, content_to_search, re.IGNORECASE)
            else:  # "matches with" (exact word match)
                match = re.search(r"\b" + keyword_pattern + r"\b", content_to_search, re.IGNORECASE)

            if match:
                parsed_article['title'] = highlight_keywords(parsed_article['title'], query_keywords)
                parsed_article['content'] = highlight_keywords(parsed_article['content'], query_keywords)
                results.append(parsed_article)

    return results

# Load Hugging Face API key safely
hf_api_key = st.secrets.get("HUGGINGFACE_API_KEY", None)

def query_mixtral(prompt):
    """Uses Hugging Face API to generate AI-based responses."""
    if not hf_api_key:
        return "Error: Hugging Face API Key is missing!"
    model = HuggingFacePipeline.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key)
    return model(prompt)

# Streamlit UI
st.title("Gujarati Newspaper Search 📰")

selected_newspaper = st.selectbox("📌 Select Newspaper", ["Gujarat Samachar", "Divya Bhaskar", "Sandesh"])
search_type = st.selectbox("🔍 Search Type", ["matches with", "contains"])
query = st.text_input("🔎 Enter Gujarati Keywords")
search_button = st.button("Search 🔍")

repo_url = "https://raw.githubusercontent.com/Mokshaa-Joshi/news_bot/main"
file_paths = {
    "Gujarat Samachar": "gs.txt",
    "Divya Bhaskar": "db.txt",
    "Sandesh": "s.txt"
}

if search_button and query:
    content = download_articles_from_github(repo_url, file_paths[selected_newspaper])
    
    if content:
        articles = load_articles(content, selected_newspaper)
        results = search_articles(articles, query, search_type, selected_newspaper)
        
        if results:
            for res in results:
                with st.container():
                    st.markdown(f"<h3>{res['title']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<strong>📅 Date:</strong> {res['date']}", unsafe_allow_html=True)
                    if 'link' in res:
                        st.markdown(f'<a href="{res["link"]}" target="_blank">🔗 Read more
