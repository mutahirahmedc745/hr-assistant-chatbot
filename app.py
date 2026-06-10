# Query rewriting function
def rewrite_question(question, history):
    if not history:
        return question
    
    history_text = "\n".join([
        f"Human: {msg['content']}\nAssistant: {msg['content']}"
        for msg in history if msg['role'] != 'system'
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
    # Rewrite vague questions
    standalone_question = rewrite_question(question, chat_history)
    
    # Build conversation history text
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

    # Extract sources
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get('page', 0) + 1
        section = doc.page_content[:60].strip().replace('\n', ' ')
        source_str = f"Page {page} — '{section}...'"
        if source_str not in sources:
            sources.append(source_str)

    return answer, sources

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
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

# Chat input at the bottom
if question := st.chat_input("Ask about HR policies..."):
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_question(
                question,
                st.session_state.messages[:-1]
            )
        st.markdown(answer)
        with st.expander("📄 Sources"):
            for source in sources:
                st.caption(source)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })