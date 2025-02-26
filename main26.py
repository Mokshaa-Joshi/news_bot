import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import re
from deep_translator import GoogleTranslator

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

st.set_page_config(page_title="NewsBot", page_icon="📰", layout="wide")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

NEWSPAPER_NAMESPACES = {
    "gujarat samachar": "gujarat_samachar",
    "divya bhaskar": "divya_bhaskar",
    "sandesh": "sandesh"
}

st.markdown("<h1 style='text-align: center;'>🤖 NewsBot - Your Personal News Assistant</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about news articles by keyword, date, and newspaper."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about news articles...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Translate input to English if it's in Gujarati
    translated_input = GoogleTranslator(source="auto", target="en").translate(user_input)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", translated_input)
    date_filter = date_match.group(0) if date_match else None

    newspaper = None
    for paper in NEWSPAPER_NAMESPACES.keys():
        if paper in translated_input.lower():
            newspaper = NEWSPAPER_NAMESPACES[paper]
            break

    words = translated_input.lower().split()
    keywords = [word for word in words if word not in ["give", "me", "news", "on", "from", "of", "date"]]
    search_query = " ".join(keywords)

    if not newspaper:
        response = "❌ Please mention a newspaper (Gujarat Samachar, Divya Bhaskar, Sandesh)."
    elif not search_query:
        response = "❌ Please enter a valid search query."
    else:
        query_vector = model.encode(search_query).tolist()

        results = index.query(
            vector=query_vector,
            top_k=10,
            namespace=newspaper,
            include_metadata=True
        )

        articles = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            title = metadata["title"]
            date = metadata["date"]
            link = metadata.get("link", "")

            if date_filter and date != date_filter:
                continue  

            # Retrieve full article chunks using keyword search in both title and content
            full_chunks = index.query(
                vector=[0] * 384,
                top_k=100,
                namespace=newspaper,
                include_metadata=True,
                filter={
                    "$or": [
                        {"title": {"$regex": search_query}},
                        {"content_chunk": {"$regex": search_query}}
                    ]
                }
            )

            merged_content = {chunk["metadata"]["chunk_index"]: chunk["metadata"]["content_chunk"] for chunk in full_chunks["matches"]}

            if merged_content:
                articles[title] = {
                    "date": date,
                    "newspaper": newspaper.replace("_", " ").title(),
                    "content": " ".join([merged_content[i] for i in sorted(merged_content)]),
                    "link": link
                }

        if articles:
            response = "### 🔍 Search Results"
            for title, details in articles.items():
                with st.expander(f"📌 {title}"):
                    st.markdown(f"📅 **Date:** {details['date']}")
                    st.markdown(f"🗞 **Newspaper:** {details['newspaper']}")
                    if details["link"]:
                        st.markdown(f"🔗 [Read More]({details['link']})")
                    st.write(f"📜 **Full Article:**\n{details['content']}")
        else:
            response = "❌ No matching news articles found."

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
