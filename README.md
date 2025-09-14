PDF Research Assistant AI Agent
Overview

The PDF Research Assistant is a local AI agent that allows users to upload PDF documents and ask multiple questions about their content. The agent provides context-grounded answers using a local LLM (Llama 3 via Ollama) and LangChain for document retrieval. Users can also download the entire chat history.

This project is fully offline, free, and requires no sign-in, making it ideal for demonstrations and adding to your resume or portfolio.

Features

Upload PDF documents (e.g., research papers, company policies).

Ask multiple questions sequentially.

Answers are based on the uploaded PDF content.

Save all Q&A in a downloadable text file.

Built using Streamlit, LangChain, FAISS, HuggingFace embeddings, and Llama 3 (Ollama).

Installation & Setup

Install Python 3.10+ and Ollama
 with Llama 3:

ollama pull llama3


Clone the repository:

git clone https://github.com/Vishwa77777/PDF_Agent.git
cd PDF_Agent


(Optional) Create a virtual environment:

python -m venv env
env\Scripts\activate


Install dependencies:

pip install streamlit langchain langchain-community PyPDF2 faiss-cpu sentence-transformers


Run the app:

streamlit run app.py

Usage

Upload a PDF file in the app.

Ask multiple questions about the document.

Download chat history for future reference.
