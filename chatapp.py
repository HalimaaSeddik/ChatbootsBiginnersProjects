import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = Ollama(model="phi3")

st.set_page_config(page_title="Chatbot + RAG", page_icon="🤖")
st.title("🤖 Chatbot with File Upload (RAG Ready)")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


uploaded_file = st.file_uploader("Upload a PDF for RAG", type="pdf")

if uploaded_file:
   
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(pages)

    st.info(f"Loaded {len(docs)} chunks")

    
    #embeddings = HuggingFaceEmbeddings() cuz there is no enugh memory to install sentence transformers ,so i use ollama embedding model which is more efficient and faster than sentence transformers
    embeddings = OllamaEmbeddings(model="phi3")
    
    db = FAISS.from_documents(docs, embeddings)

   
    st.session_state.vector_db = db

    st.success("Document processed and ready for questions!")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input("Ask something...")

if user_input:
    st.chat_message("user").write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

   
    if st.session_state.vector_db:
        retriever = st.session_state.vector_db.as_retriever()

        docs = retriever.invoke(user_input)
            
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a document QA assistant.

Rules:
- Answer ONLY using the context below.
- Do NOT hallucinate or repeat information not present.
- Be concise.

Context:
{context}

Answer:
        Question: {user_input}
        """

        response = llm.invoke(prompt)

    else:
        
        response = llm.invoke(user_input)

   
    st.chat_message("assistant").write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })