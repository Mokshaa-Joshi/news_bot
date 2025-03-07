import streamlit as st
import re
import os
from langchain.llms import HuggingFacePipeline

def load_articles(directory, newspaper):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and newspaper in filename:
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                articles.append(file.read())
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
                "date": lines[0],
                "title": lines[1],
                "content": "\n".join(lines[2:])
            }
    return None

def search_articles(articles, keyword_pattern, search_type):
    results = []
    for article in articles:
        parsed_article = parse_article(article, selected_newspaper)
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

def query_mixtral(prompt):
    model = HuggingFacePipeline.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    return model(prompt)

# Streamlit UI
st.title("Gujarati Newspaper Search")

selected_newspaper = st.selectbox("Select Newspaper", ["Gujarat Samachar", "Divya Bhaskar", "Sandesh"])
search_type = st.selectbox("Search Type", ["matches with", "contains"])
query = st.text_input("Enter Gujarati Keywords")
search_button = st.button("Search")

if search_button and query:
    keyword_pattern = build_regex(query)
    articles = load_articles("news_articles", selected_newspaper)
    results = search_articles(articles, keyword_pattern, search_type)
    
    if results:
        for res in results:
            st.subheader(res['title'])
            st.write(f"**Date:** {res['date']}")
            if 'link' in res:
                st.write(f"[Read more]({res['link']})")
            st.write(res['content'][:300] + "...")
    else:
        st.write("No articles found.")
