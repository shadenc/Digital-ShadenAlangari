import os
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import requests
from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"] = "sk-proj-ZvvRmuGFJ7dYRpxM6KrMT3BlbkFJpIvGSadlG5RAVA8xITl9"

def scrape_data(url):
        # Fetch the content of the URL
        response = requests.get(url)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from elements
        all_text = soup.get_text(separator=' ', strip=True)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=16384, chunk_overlap=2048, length_function=len)
        return text_splitter.split_text(all_text)

def main(urls):
    index_name = "chatbot"
    pinecone = Pinecone(api_key="b6fafa8d-73ee-4692-8a9f-c6d877d214e9")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    index = pinecone.Index(index_name)
     
    for url in urls:    
        docs = scrape_data(url)

        embeddings = embedding_model.embed_documents(docs)

        documents = []
        for i, embedding in enumerate(embeddings):
            print(i, embedding, docs[i])
            documents.append({
                "id": "{}_{}".format(url, i),
                "values": embedding,
                "metadata": {
                    "file_path": url,
                    "text": docs[i]
                    },
        })

        index.upsert(documents)

if __name__ == "__main__":
    urls = ["https://u.ae/en/information-and-services", 
            "https://u.ae/en/information-and-services/visa-and-emirates-id", 
            "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas", 
            "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa" 
]
    main(urls)
