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
        st.write(f"Error fetching file: {url} - Status Code: {response.status_code}")
        return None

def load_articles(content, newspaper):
    articles = []
    if content:
        if newspaper in ["Gujarat Samachar", "Divya Bhaskar"]:
            articles = content.split("================================================================================")
        elif newspaper == "Sandesh":
            # Splitting articles based on the date pattern at the start of each article
            articles = re.split(r"(?=\w{3} \d{1,2}, \d{4} \d{2}:\d{2} (am|pm))", content)
            articles = [article.strip() for article in articles if article.strip()]
    return articles

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
                "date": lines[0],  # The first line is the date
                "title": lines[1],  # The second line is the title
                "content": "\n".join(lines[2:])  # The remaining lines are content
            }
    return None

def search_articles(articles, keyword_pattern, search_type, newspaper):
    results = []
    for article in articles:
        parsed_article = parse_article(article, newspaper)
        if parsed_article:
            content_to_search = f"{parsed_article['title']} {parsed_article['content']}"
            match = re.search(keyword_pattern, content_to_search, re.IGNORECASE)
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
    st.write(f"Using Regex Pattern: `{keyword_pattern}`")  # Debugging
    
    content = download_articles_from_github(repo_url, file_paths[selected_newspaper])
    if content:
        articles = load_articles(content, selected_newspaper)
        st.write(f"Total Articles Loaded: {len(articles)}")  # Debugging
        
        results = search_articles(articles, keyword_pattern, search_type, selected_newspaper)
        
        if results:
            for res in results:
                st.subheader(res['title'])
                st.write(f"**Date:** {res['date']}")
                if 'link' in res:
                    st.write(f"[Read more]({res['link']})")
                st.write(res['content'][:300] + "...")
        else:
            st.write("No matching articles found. Try different keywords.")
    else:
        st.write("Error fetching file from GitHub. Make sure the file path is correct.")
