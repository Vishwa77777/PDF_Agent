import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama

# Load local LLM
llm = Ollama(model="llama3")

# Streamlit UI
st.set_page_config(page_title="PDF Research Assistant", layout="wide")
st.title("📄 PDF Research Assistant")
st.write("Upload a PDF and ask multiple questions. Powered by Llama 3 + LangChain.")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    # Read PDF text
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    # Create embeddings & FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(docs, embeddings)

    # Conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever())

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    query = st.text_input("Ask a question about the PDF:")

    if query:
        result = qa({"question": query, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((query, result["answer"]))

    # Display chat history for multiple questions
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Question {i+1}:** {q}")
        st.markdown(f"**Answer {i+1}:** {a}")

    # Save chat history to file & download
    if st.session_state.chat_history:
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for i, (q, a) in enumerate(st.session_state.chat_history):
                f.write(f"Question {i+1}: {q}\n")
                f.write(f"Answer {i+1}: {a}\n\n")

        st.download_button(
            label="📥 Download Chat History",
            data=open("chat_history.txt", "r", encoding="utf-8").read(),
            file_name="chat_history.txt",
            mime="text/plain"
        )
