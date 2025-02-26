import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import re
import string
from deep_translator import GoogleTranslator

# Load Pinecone API key
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

st.set_page_config(page_title="NewsBot", page_icon="📰", layout="wide")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("news2")

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define newspaper namespaces
NEWSPAPER_NAMESPACES = {
    "gujarat samachar": "gujarat_samachar",
    "divya bhaskar": "divya_bhaskar",
    "sandesh": "sandesh"
}

st.markdown("<h1 style='text-align: center;'>🤖 NewsBot - Your Personal News Assistant</h1>", unsafe_allow_html=True)

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about news articles by keyword, date, and newspaper."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input("Ask me about news articles...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Translate input to English if it's in Gujarati
    translated_input = GoogleTranslator(source="auto", target="en").translate(user_input)

    # Extract date from input
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", translated_input)
    date_filter = date_match.group(0) if date_match else None

    # Identify the newspaper
    newspaper = None
    for paper in NEWSPAPER_NAMESPACES.keys():
        if paper in translated_input.lower():
            newspaper = NEWSPAPER_NAMESPACES[paper]
            break

    # Extract keywords from input (remove common words)
    words = translated_input.lower().translate(str.maketrans("", "", string.punctuation)).split()
    stopwords = {"give", "me", "news", "on", "from", "of", "date", "about", "the", "is", "a"}
    keywords = [word for word in words if word not in stopwords]

    search_query = " ".join(keywords)

    if not newspaper:
        response = "❌ Please mention a newspaper (Gujarat Samachar, Divya Bhaskar, Sandesh)."
    elif not search_query:
        response = "❌ Please enter a valid search query."
    else:
        query_vector = model.encode(search_query).tolist()

        # Perform vector search in Pinecone
        results = index.query(
            vector=query_vector,
            top_k=50,  # Retrieve more results
            namespace=newspaper,
            include_metadata=True
        )

        # Debugging: Check retrieved results
        st.write("🔍 Pinecone Raw Results:", results)

        articles = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            title = metadata.get("title", "").lower()
            date = metadata.get("date", "")
            link = metadata.get("link", "")
            content_chunk = metadata.get("content_chunk", "").lower()

            if date_filter and date != date_filter:
                continue  # Skip if date does not match

            # **Check if any keyword exists in title OR content chunk**
            keyword_found = any(keyword in title or keyword in content_chunk for keyword in keywords)

            if keyword_found:
                if title not in articles:
                    articles[title] = {
                        "date": date,
                        "newspaper": newspaper.replace("_", " ").title(),
                        "content": content_chunk,
                        "link": link
                    }
                else:
                    articles[title]["content"] += " " + content_chunk  # Merge chunks

        # Display results
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
