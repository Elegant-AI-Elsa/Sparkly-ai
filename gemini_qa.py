# gemini_qa.py

import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

def get_chain():
    prompt = PromptTemplate.from_template("Answer based on context:\n{context}\n\nQ: {question}\nA:")
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def get_answer(question):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectordb = Qdrant(
        client=client,
        collection_name="ai-web-assistant",  # Match this with vector_store.py
        embeddings=embeddings
    )

    docs = vectordb.similarity_search(question, k=5)
    chain = get_chain()
    return chain.run(input_documents=docs, question=question)
