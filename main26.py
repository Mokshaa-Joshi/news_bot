import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import re
from deep_translator import GoogleTranslator

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

    # 🈯 Translate User Input to Gujarati (If Needed)
    lang_detected = GoogleTranslator(source="auto", target="en").detect(user_input)
    if lang_detected != "gu":
        translated_input = GoogleTranslator(source="auto", target="gu").translate(user_input)
    else:
        translated_input = user_input  # Already in Gujarati

    # Extract date from query
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", translated_input)
    date_filter = date_match.group(0) if date_match else None

    # Extract newspaper name
    newspaper = None
    for paper in NEWSPAPER_NAMESPACES.keys():
        if paper in translated_input.lower():
            newspaper = NEWSPAPER_NAMESPACES[paper]
            break

    # Extract keywords (English & Gujarati)
    words = translated_input.lower().split()
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

        # Query Pinecone (Vector Search)
        results = index.query(
            vector=query_vector,
            top_k=10,
            namespace=newspaper,
            include_metadata=True
        )

        # 📰 Fetch Full Articles (Merge All Chunks & Filter by Keywords)
        articles = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            title = metadata["title"]
            date = metadata["date"]
            link = metadata.get("link", "")
            chunk_content = metadata["content_chunk"]

            # Skip non-matching dates
            if date_filter and date != date_filter:
                continue  

            # If the article title or content chunk contains search keywords, fetch all its chunks
            if search_query in title.lower() or any(word in chunk_content.lower() for word in keywords):
                if title not in articles:
                    full_chunks = index.query(
                        vector=[0] * 384,  # Dummy vector (not searching by vector, just getting full article)
                        top_k=100,  # Get all chunks
                        namespace=newspaper,
                        include_metadata=True,
                        filter={"title": title}  # Fetch all chunks of the same article
                    )

                    merged_content = {chunk["metadata"]["chunk_index"]: chunk["metadata"]["content_chunk"] for chunk in full_chunks["matches"]}

                    # Store the full article
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
