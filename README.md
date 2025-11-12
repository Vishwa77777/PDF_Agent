📄 PDF Research Assistant AI Agent  

 🚀 Overview  
The **PDF Research Assistant** is a **local AI-powered tool** that lets you upload and analyze large PDF files (up to **500 MB** each). It uses **Llama 3 (via Ollama)** and **LangChain** for intelligent, context-aware question answering — all processed **locally and offline**.  

With a clean interface, customizable themes, and multi-PDF management, it’s ideal for research, portfolio projects, or offline document analysis.  

📸 Interface Preview
📂 Upload & Manage PDFs
💬 Chat
🔧 Settings
ℹ️ About

 ✨ Features  

✅ **Upload & Manage Multiple PDFs** – Handle several PDFs at once (supports up to 500 MB each).  
✅ **Ask Smart Questions** – Get context-grounded answers from your PDFs.  
✅ **Theme Customization** – Switch between **Light** 🌞 and **Dark** 🌙 modes.  
✅ **Sidebar Navigation** – Easily move between uploads, chat, and settings.  
✅ **Download Chat History** – Save your Q&A conversations as a `.txt` file.  
✅ **Fully Local & Secure** – Runs 100% offline. No sign-in. No data tracking.  
✅ **Optimized for Large Files** – Efficient memory handling and garbage collection.  

---

 🧠 Tech Stack  

- **Streamlit** – Interactive web UI  
- **LangChain** – Document retrieval and QA chain  
- **FAISS** – Vector database for semantic search  
- **HuggingFace Sentence Transformers** – Embeddings model  
- **Ollama + Llama 3** – Local LLM inference
  
  🛡️ Security & Privacy

All processing is done locally on your system.

No files or chat data leave your machine.

Automatic file-size validation prevents oversized uploads.
  
- Usage

Upload one or more PDFs (up to 500 MB each).

Select a PDF from the sidebar to start chatting.

Ask any question — the model gives contextual answers.

Switch themes or change settings in the sidebar.

Download chat history when done.


 ⚙️ Installation & Setup  

 1️⃣ Prerequisites  
Install **Python 3.10+** and **Ollama** (with Llama 3 model):

```bash
ollama pull llama3

2️⃣ Create a Virtual Environment
python -m venv env
env\Scripts\activate

3️⃣ Install Dependencies
pip install streamlit langchain langchain-community PyPDF2 faiss-cpu sentence-transformers

4️⃣ Run the App
streamlit run app.py
 give all in correct format and in one font highlight only imp one
