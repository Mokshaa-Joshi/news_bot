import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import re

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
    "📰 Gujarat Samachar": "gujarat_samachar",
    "🗞️ Divya Bhaskar": "divya_bhaskar",
    "🗃️ Sandesh": "sandesh"
}

# 🌟 Sidebar - Newspaper Selection
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Newspaper.svg/600px-Newspaper.svg.png", width=150)
    st.markdown("### 📰 Choose a Newspaper")
    newspaper = st.radio("Select:", list(NEWSPAPER_NAMESPACES.keys()))
    st.markdown("💡 Example Queries:")
    st.markdown("- _'Give me news on Budget from Gujarat Samachar of date 2025-02-23'_")
    st.markdown("- _'Find articles about elections from Sandesh'_")

# 💬 Chat Interface Header
st.markdown("<h1 style='text-align: center;'>🤖 NewsBot - Your Personal News Assistant</h1>", unsafe_allow_html=True)

# 📝 Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about news articles by keyword, date, and newspaper."}]

# 📜 Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🎤 User Input Box
user_input = st.chat_input("Ask me about news articles...")

# 🎯 Process Query if User Inputs Text
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Extract date from query
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
    date_filter = date_match.group(0) if date_match else None

    # Extract keywords
    words = user_input.lower().split()
    keywords = [word for word in words if word not in ["give", "me", "news", "on", "from", "of", "date"]]
    search_query = " ".join(keywords)

    # 🔍 Perform Search Only if Query is Valid
    if search_query:
        # Generate Query Vector
        query_vector = model.encode(search_query).tolist()

        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=5,
            namespace=NEWSPAPER_NAMESPACES[newspaper],
            include_metadata=True
        )

        response = ""
        if results["matches"]:
            for match in results["matches"]:
                metadata = match["metadata"]
                if date_filter and metadata.get("date") != date_filter:
                    continue  # Skip articles that do not match date

                response += f"📅 **{metadata['date']}**\n🔹 **{metadata['title']}**\n📝 {metadata['content_chunk'][:200]}...\n\n"

        if not response:
            response = "❌ No matching news articles found."

    else:
        response = "❌ Please enter a valid search query."

    # 🧠 Store Bot Response in Chat History
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 💬 Display Bot Response
    with st.chat_message("assistant"):
        st.markdown(response)
