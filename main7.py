import streamlit as st
import re
import os
from langchain.llms import HuggingFacePipeline

def load_articles(file_path):
    articles = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        articles = content.split("================================================================================")
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

file_paths = {
    "Gujarat Samachar": "D:/Projects/4-Dbs/news_website/news/gs.txt",
    "Divya Bhaskar": "D:/Projects/4-Dbs/news_website/news/db.txt",
    "Sandesh": "D:/Projects/4-Dbs/news_website/news/s.txt"
}

if search_button and query:
    keyword_pattern = build_regex(query)
    articles = load_articles(file_paths[selected_newspaper])
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
