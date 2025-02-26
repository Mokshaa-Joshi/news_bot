import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import re
import asyncio

# 🔐 Securely Load API Keys
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# 🎨 Streamlit UI Settings
st.set_page_config(page_title="NewsBot", page_icon="📰", layout="wide")

# 🚀 Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

# 🧠 Load SentenceTransformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 📌 Define Newspaper Namespaces
NEWSPAPER_NAMESPACES = {
    "gujarat samachar": "gujarat_samachar",
    "divya bhaskar": "divya_bhaskar",
    "sandesh": "sandesh"
}

# 💬 Chat Interface Header
st.markdown("<h1 style='text-align: center;'>🤖 NewsBot - Your Personal News Assistant</h1>", unsafe_allow_html=True)

# 📝 Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about news articles in English or Gujarati by keyword, date, and newspaper."}]

# 📜 Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🎤 User Input Box
user_input = st.chat_input("Ask me about news articles...")

# 🎯 Process Query if User Inputs Text
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 🌍 Detect & Translate Query (If Needed)
    try:
        translated_query = GoogleTranslator(source="auto", target="en").translate(user_input)
    except Exception:
        translated_query = user_input  # Fallback if translation fails

    # Extract date from query
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", translated_query)
    date_filter = date_match.group(0) if date_match else None

    # Extract newspaper name
    newspaper = None
    for paper in NEWSPAPER_NAMESPACES.keys():
        if paper in translated_query.lower():
            newspaper = NEWSPAPER_NAMESPACES[paper]
            break

    # Extract keywords
    words = translated_query.lower().split()
    keywords = [word for word in words if word not in ["give", "me", "news", "on", "from", "of", "date"]]
    search_query = " ".join(keywords)

    # 📰 Validate Query
    if not newspaper:
        response = "❌ Please mention a newspaper (Gujarat Samachar, Divya Bhaskar, Sandesh)."
    elif not search_query:
        response = "❌ Please enter a valid search query."
    else:
        # Generate Query Vector
        query_vector = model.encode(search_query).tolist()

        # 🔍 Query Pinecone (Retrieve Top 50 Results)
        results = index.query(
            vector=query_vector,
            top_k=50,  # Get enough results for manual filtering
            namespace=newspaper,
            include_metadata=True
        )

        # 📰 Fetch Full Articles (Merge All Chunks)
        articles = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            title = metadata["title"]
            date = metadata["date"]
            content_chunk = metadata["content_chunk"]
            link = metadata.get("link", "")

            # Skip non-matching dates
            if date_filter and date != date_filter:
                continue  

            # **Manual Keyword Matching in Title & Content**
            if any(kw in title.lower() for kw in keywords) or any(kw in content_chunk.lower() for kw in keywords):
                # Fetch all chunks of the same article
                if title not in articles:
                    full_chunks = index.query(
                        vector=query_vector,  # Use the same vector to get more relevant chunks
                        top_k=100,  # Ensure all chunks are retrieved
                        namespace=newspaper,
                        include_metadata=True
                    )

                    # Merge all chunks in correct order
                    merged_content = {}
                    for chunk in full_chunks["matches"]:
                        chunk_index = chunk["metadata"]["chunk_index"]
                        merged_content[chunk_index] = chunk["metadata"]["content_chunk"]

                    # Store full article
                    articles[title] = {
                        "date": date,
                        "newspaper": newspaper.replace("_", " ").title(),
                        "content": " ".join([merged_content[i] for i in sorted(merged_content)]),
                        "link": link
                    }

        # 📌 Display Merged Articles
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

    # 🧠 Store Bot Response in Chat History
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 💬 Display Bot Response
    with st.chat_message("assistant"):
        st.markdown(response)

# ✅ Fix RuntimeError: "no running event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Ensures event loop is properly managed
