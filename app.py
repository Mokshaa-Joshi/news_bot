import os
import pinecone
import openai
import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

openai.api_key = OPENAI_API_KEY

def extract_keywords(prompt):
    """Extract keywords using OpenAI API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract keywords from the given prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"].split(', ')

def standardize_date(user_date):
    """Convert various date formats into a standard YYYY-MM-DD format."""
    try:
        parsed_date = datetime.datetime.strptime(user_date, "%Y-%m-%d")
    except ValueError:
        try:
            parsed_date = datetime.datetime.strptime(user_date, "%d-%m-%Y")
        except ValueError:
            try:
                parsed_date = datetime.datetime.strptime(user_date, "%d/%m/%Y")
            except ValueError:
                parsed_date = None
    
    return parsed_date.strftime("%Y-%m-%d") if parsed_date else None

def search_news(prompt, user_date=None, lang="gu"):
    """Search news articles in Pinecone using keyword matching."""
    translated_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
    keywords = extract_keywords(translated_prompt)
    query_embedding = openai.Embedding.create(input=translated_prompt, model="text-embedding-ada-002")["data"][0]["embedding"]
    
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    
    if user_date:
        standardized_date = standardize_date(user_date)
    
    for match in results["matches"]:
        metadata = match["metadata"]
        if all(keyword.lower() in metadata["title"].lower() or keyword.lower() in metadata["content"].lower() for keyword in keywords):
            if user_date is None or metadata["date"] == standardized_date:
                return {
                    "title": metadata["title"],
                    "date": metadata["date"],
                    "link": metadata["link"],
                    "content": metadata["content"]
                }
    return "No matching news found."

# Example usage
user_prompt = "ગુજરાતમાં ભારે વરસાદ"
user_date = "15-02-2024"
result = search_news(user_prompt, user_date)
print(result)
