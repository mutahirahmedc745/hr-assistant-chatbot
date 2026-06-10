import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Page configuration
st.set_page_config(
    page_title="TechCorp HR Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 TechCorp HR Assistant")
st.markdown("Ask me anything about TechCorp Pakistan's HR policies.")
st.divider()

# Load API key
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    else:
        st.error("GROQ_API_KEY not found in secrets!")
        st.stop()
except Exception as e:
    st.error(f"Error loading API key: {e}")
    st.stop()

# Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

# Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# Load vectorstore
@st.cache_resource
def load_vectorstore(_embeddings):
    loader = PyPDFLoader("hr_policy.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, _embeddings)
    return vectorstore

# Initialize everything
try:
    llm = load_llm()
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Query rewriting
def rewrite_question(question, history):
    if not history:
        return question
    history_text = "\n".join([
        f"Human: {msg['content']}\nAssistant: {history[i+1]['content']}"
        for i, msg in enumerate(history)
        if msg['role'] == 'user' and i+1 < len(history)
    ])
    rewrite_prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question as a complete standalone question.
Return ONLY the rewritten question, nothing else.

Follow-up question: {question}
Standalone question:"""
    rewritten = llm.invoke(rewrite_prompt).content
    return rewritten.strip()

# Main RAG function
def ask_question(question, chat_history):
    standalone_question = rewrite_question(question, chat_history)
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"Human: {msg['content']}\nAssistant: {chat_history[i+1]['content']}"
            for i, msg in enumerate(chat_history)
            if msg['role'] == 'user' and i+1 < len(chat_history)
        ])

    prompt_text = f"""You are a helpful HR assistant for TechCorp Pakistan.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in my knowledge base."

Previous conversation:
{history_text}

Context: {{context}}

Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(standalone_question)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get('page', 0) + 1
            formatted.append(f"{doc.page_content}\n[Source: Page {page}]")
        return "\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(standalone_question)

    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get('page', 0) + 1
        section = doc.page_content[:60].strip().replace('\n', ' ')
        source_str = f"Page {page} — '{section}...'"
        if source_str not in sources:
            sources.append(source_str)

    return answer, sources

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm the TechCorp HR Assistant. Ask me anything about our HR policies — leaves, working hours, salary, benefits, or code of conduct."
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.caption(source)

# Chat input
if question := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_question(
                    question,
                    st.session_state.messages[:-1]
                )
            except Exception as e:
                answer = f"Error generating answer: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for source in sources:
                    st.caption(source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
