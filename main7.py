import streamlit as st
import re
import os
import requests
from langchain.llms import HuggingFacePipeline

def download_articles_from_github(repo_url, filename):
    url = f"{repo_url}/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def load_articles(content, newspaper):
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
    """Highlights the keywords in the given text."""
    for keyword in keywords:
        text = re.sub(f"({re.escape(keyword)})", r"**\1**", text, flags=re.IGNORECASE)
    return text

def search_articles(articles, query, newspaper):
    """Searches for the keyword in the title and content of articles. Only returns matching articles."""
    results = []
    query_keywords = query.strip().split()  # Split query into words
    for article in articles:
        parsed_article = parse_article(article, newspaper)
        if parsed_article:
            title = parsed_article['title'].lower()
            content = parsed_article['content'].lower()

            # Show the article only if keyword is found in title or content
            if any(keyword.lower() in title or keyword.lower() in content for keyword in query_keywords):
                parsed_article['title'] = highlight_keywords(parsed_article['title'], query_keywords)
                parsed_article['content'] = highlight_keywords(parsed_article['content'], query_keywords)
                results.append(parsed_article)
    return results

# Load Hugging Face API key safely
hf_api_key = st.secrets.get("HUGGINGFACE_API_KEY", None)

def query_mixtral(prompt):
    if not hf_api_key:
        return "Error: Hugging Face API Key is missing!"
    model = HuggingFacePipeline.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key)
    return model(prompt)

# Streamlit UI
st.title("Gujarati Newspaper Search")

selected_newspaper = st.selectbox("Select Newspaper", ["Gujarat Samachar", "Divya Bhaskar", "Sandesh"])
query = st.text_input("Enter Gujarati Keywords")
search_button = st.button("Search")

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
        results = search_articles(articles, query, selected_newspaper)
        
        if results:
            for res in results:
                with st.container():
                    st.markdown(f"### {res['title']}")
                    st.markdown(f"**Date:** {res['date']}")
                    if 'link' in res:
                        st.markdown(f"[Read more]({res['link']})")
                    st.markdown(f"{res['content']}")
                    st.markdown("---")
        else:
            st.write("No matching articles found. Try different keywords.")
    else:
        st.write("Error fetching file from GitHub. Make sure the file path is correct.")
