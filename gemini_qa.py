import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def embed_chunks(chunks):
    return embedding_model.embed_documents(chunks)

def embed_query(query):
    return embedding_model.embed_query(query)

def get_answer(context, query):
    prompt = f"""You are an assistant answering questions from a website.
Use the following context:\n{context}\n
Question: {query}
Answer:"""
    return llm.invoke(prompt).content
