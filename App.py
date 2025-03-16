import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Create a text generation pipeline
pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    model_kwargs={
        "temperature": 0,
        "repetition_penalty": 1.5
    }
)

# Wrap the pipeline in a LangChain HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Load the vector store (replace with your actual vector store path)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="../vector-store/chroma_db", embedding_function=embedding_model)

# Create the ConversationalRetrievalChain
chain = ConversationalRetrievalChain(
    retriever=vectordb.as_retriever(),
    question_generator=llm,
    combine_docs_chain=llm,
    return_source_documents=True,
    memory=st.session_state.get("memory", None),
    verbose=True,
    get_chat_history=lambda h: h
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app
st.title("Chatbot Demo")
st.write("Ask me anything about Aymen!")

# Input box for user message
user_input = st.text_input("Type your message here:")

# Chat interface
if user_input:
    # Generate response using the chain
    response = chain({"question": user_input})
    answer = response["answer"]
    source_documents = response["source_documents"]

    # Add user input and bot response to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
        st.write("---")

    # Display source documents (if available)
    if source_documents:
        st.write("**Source Documents:**")
        for doc in source_documents:
            st.write(f"- {doc.page_content}")