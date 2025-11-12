import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
import tempfile
import gc

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="📄 PDF Research Assistant", layout="wide")

# ---------------------- SIDEBAR NAVIGATION ----------------------
st.sidebar.title("⚙️ Settings & Navigation")
menu = st.sidebar.radio("Choose Section", ["📁 Upload & Manage PDFs", "💬 Chat", "🔧 Settings", "ℹ️ About"])

# Theme toggle
theme_choice = st.sidebar.selectbox("🎨 Theme", ["Light Mode", "Dark Mode"])

# ---------------------- CUSTOM THEMES ----------------------
light_theme = """
<style>
.stApp { background: linear-gradient(135deg, #f7faff 0%, #e3f2fd 100%); color: #111; }
.title { font-size: 40px; text-align: center; color: #1a73e8; font-weight: bold; }
.subtitle { text-align: center; font-size: 18px; color: #444; margin-bottom: 20px; }
.question { background: #e8f0fe; padding: 10px 15px; border-radius: 10px; margin-bottom: 8px; }
.answer { background: #fff; padding: 10px 15px; border-left: 4px solid #1a73e8; border-radius: 8px; 
          box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px; }
div.stDownloadButton > button { background-color: #1a73e8; color: white; border-radius: 8px; padding: 10px 20px; border: none; }
div.stDownloadButton > button:hover { background-color: #155ab6; }
</style>
"""

dark_theme = """
<style>
.stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); color: #f1f1f1; }
.title { font-size: 40px; text-align: center; color: #00bcd4; font-weight: bold; }
.subtitle { text-align: center; font-size: 18px; color: #ddd; margin-bottom: 20px; }
.question { background: rgba(255,255,255,0.1); padding: 10px 15px; border-radius: 10px; margin-bottom: 8px; color: #e3f2fd; }
.answer { background: rgba(255,255,255,0.15); padding: 10px 15px; border-left: 4px solid #00bcd4; 
          border-radius: 8px; margin-bottom: 15px; color: #fafafa; }
div.stDownloadButton > button { background-color: #00bcd4; color: white; border-radius: 8px; padding: 10px 20px; border: none; }
div.stDownloadButton > button:hover { background-color: #008c9e; }
</style>
"""

# Apply selected theme
st.markdown(light_theme if theme_choice == "Light Mode" else dark_theme, unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown('<h1 class="title">📚 PDF Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload multiple PDFs, customize settings, and ask smart questions powered by <b>Llama 3 + LangChain + FAISS</b>.</p>', unsafe_allow_html=True)

# ---------------------- SIDEBAR SETTINGS ----------------------
st.sidebar.markdown("### 🧩 Model & Chunk Settings")
model_name = st.sidebar.selectbox("Select LLM Model", ["llama3", "mistral", "llama2"])
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 100, step=10)

# ---------------------- SESSION STATE ----------------------
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

# ---------------------- UPLOAD SECTION ----------------------
if menu == "📁 Upload & Manage PDFs":
    st.subheader("📤 Upload Your PDFs")
    uploaded_files = st.file_uploader("Select one or more PDF files (up to 500 MB each)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for pdf_file in uploaded_files:
            pdf_name = pdf_file.name
            if pdf_name not in st.session_state.pdf_data:
                with st.spinner(f"⏳ Processing {pdf_name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(pdf_file.read())
                        temp_pdf_path = temp_pdf.name

                    pdf_reader = PdfReader(temp_pdf_path)
                    text_parts = []
                    for page in pdf_reader.pages:
                        content = page.extract_text()
                        if content:
                            text_parts.append(content)
                    text = " ".join(text_parts)
                    del text_parts
                    gc.collect()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    docs = splitter.split_text(text)
                    del text
                    gc.collect()

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    db = FAISS.from_texts(docs, embeddings)
                    llm = Ollama(model=model_name)
                    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever())

                    st.session_state.pdf_data[pdf_name] = {"qa": qa_chain, "chat": []}
                    st.success(f"✅ {pdf_name} processed successfully!")

    if st.session_state.pdf_data:
        st.markdown("### 📚 Your Uploaded PDFs:")
        for name in st.session_state.pdf_data.keys():
            if st.button(f"📖 Open {name}"):
                st.session_state.selected_pdf = name
                st.success(f"Now chatting with: **{name}**")

# ---------------------- CHAT SECTION ----------------------
elif menu == "💬 Chat":
    if not st.session_state.pdf_data:
        st.warning("⚠️ Please upload and process at least one PDF first.")
    else:
        pdf_names = list(st.session_state.pdf_data.keys())
        selected = st.selectbox("Select a PDF to chat with:", pdf_names)
        st.session_state.selected_pdf = selected

        if selected:
            qa = st.session_state.pdf_data[selected]["qa"]
            chat_history = st.session_state.pdf_data[selected]["chat"]

            query = st.text_input(f"💬 Ask a question about **{selected}**:")
            if query:
                with st.spinner("🤔 Thinking..."):
                    result = qa({"question": query, "chat_history": chat_history})
                    chat_history.append((query, result["answer"]))

            if chat_history:
                st.markdown("---")
                st.subheader("🗂 Conversation History")
                for i, (q, a) in enumerate(chat_history):
                    st.markdown(f"<div class='question'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='answer'><b>A{i+1}:</b> {a}</div>", unsafe_allow_html=True)

                chat_text = "\n".join([f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n" for i, (q, a) in enumerate(chat_history)])
                st.download_button(
                    label="📥 Download Chat History",
                    data=chat_text,
                    file_name=f"{selected}_chat_history.txt",
                    mime="text/plain",
                )

# ---------------------- SETTINGS SECTION ----------------------
elif menu == "🔧 Settings":
    st.subheader("🔧 Current App Settings")
    st.markdown(f"**Model:** {model_name}")
    st.markdown(f"**Chunk Size:** {chunk_size}")
    st.markdown(f"**Chunk Overlap:** {chunk_overlap}")
    st.markdown(f"**Theme:** {theme_choice}")
    st.info("Use the sidebar to adjust settings before uploading PDFs.")

# ---------------------- ABOUT SECTION ----------------------
elif menu == "ℹ️ About":
    st.subheader("📘 About This App")
    st.write("""
    The **PDF Research Assistant** helps you:
    - Upload and manage multiple PDFs (even large 500 MB files)
    - Ask AI-driven questions about any document  
    - Switch between PDFs easily  
    - Customize settings and theme  
      
    Built with **Streamlit, LangChain, FAISS, and Llama 3**.
    """)
