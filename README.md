# 🤖 HR Assistant Chatbot

A conversational AI chatbot that answers questions about company HR policies using RAG (Retrieval-Augmented Generation).

## Features
- 📄 PDF document ingestion
- 🧠 Semantic search using FAISS vector store
- 💬 Conversation memory across multiple questions
- 🔍 Query rewriting for vague follow-up questions
- 📌 Source citations showing which page answers came from

## Tech Stack
- **LLM:** Llama 3.3 70B via Groq API
- **Framework:** LangChain (LCEL)
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **UI:** Streamlit

## How It Works
1. HR policy PDF is loaded and split into chunks
2. Chunks are converted to vectors using HuggingFace embeddings
3. Vectors are stored in FAISS for fast similarity search
4. User asks a question → converted to vector → FAISS finds relevant chunks
5. Chunks + question sent to Llama 3.3 70B → human-readable answer returned

## Live Demo
[Live demo link: https://hr-assistant-chatbot-3s9qlzovwdtigbi93amy2u.streamlit.app/]

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to Streamlit secrets
4. Run: `streamlit run app.py`
