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
        if newspaper in ["Gujarat Samachar", "Divya Bhaskar"]:
            articles = content.split("================================================================================")
        else:  # Sandesh
            articles = content.strip().split("\n\n")  # Assuming each article is separated by a double newline
    return [article.strip() for article in articles if article.strip()]

def parse_article(article, newspaper):
    if newspaper in ["Gujarat Samachar", "Divya Bhaskar"]:
        match = re.search(r"Title: (.*?)\nDate: (.*?)\nLink: (.*?)\nContent:\n(.*)", article, re.DOTALL)
        if match:
            return {
                "title": match.group(1),
                "date": match.group(2),
                "link": match.group(3),
                "content": match.group(4)
            }
    elif newspaper == "Sandesh":
        lines = article.strip().split("\n")
        if len(lines) >= 3:
            return {
                "date": lines[0],
                "title": lines[1],
                "content": "\n".join(lines[2:])
            }
    return None

def search_articles(articles, keyword_pattern, search_type, newspaper):
    results = []
    for article in articles:
        parsed_article = parse_article(article, newspaper)
        if parsed_article:
            content_to_search = f"{parsed_article['title']} {parsed_article['content']}"
            if search_type == "matches with":
                match = re.fullmatch(keyword_pattern, content_to_search)
            else:  # 'contains'
                match = re.search(keyword_pattern, content_to_search)
            if match:
                results.append(parsed_article)
    return results

def build_regex(query):
    query = query.strip()
    if " અને " in query:
        keywords = query.split(" અને ")
        return r".*".join(re.escape(k) for k in keywords)
    elif " અથવા " in query:
        keywords = query.split(" અથવા ")
        return r"|".join(re.escape(k) for k in keywords)
    else:
        return re.escape(query)

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
search_type = st.selectbox("Search Type", ["matches with", "contains"])
query = st.text_input("Enter Gujarati Keywords")
search_button = st.button("Search")

repo_url = "https://raw.githubusercontent.com/Mokshaa-Joshi/news_bot/main"
file_paths = {
    "Gujarat Samachar": "gs.txt",
    "Divya Bhaskar": "db.txt",
    "Sandesh": "s.txt"
}

if search_button and query:
    keyword_pattern = build_regex(query)
    content = download_articles_from_github(repo_url, file_paths[selected_newspaper])
    if content:
        articles = load_articles(content, selected_newspaper)
        results = search_articles(articles, keyword_pattern, search_type, selected_newspaper)
        
        if results:
            for res in results:
                st.subheader(res['title'])
                st.write(f"**Date:** {res['date']}")
                if 'link' in res:
                    st.write(f"[Read more]({res['link']})")
                st.write(res['content'][:300] + "...")
        else:
            st.write("No articles found.")
    else:
        st.write("Error fetching file from GitHub. Make sure the file path is correct.")
