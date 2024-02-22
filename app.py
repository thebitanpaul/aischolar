import sys
import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load environment variables
load_dotenv()

# Main Streamlit app
def main():
    st.title("Query your PDF here")
    with st.sidebar:
        st.title('Ai Scholar')
        st.markdown('''
        ## About
        Choose the desired PDF or upload your own, then perform a query.
        ''')

        # File uploader for uploading PDFs
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        text = read_pdf(uploaded_file)
        st.info("The content of the PDF is hidden. Type your query in the chat window.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the PDF text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        # Save vectorstore using pickle
        pickle_data = pickle.dumps(vectorstore)

        # Store pickle data in a SessionState variable
        if "pickle_data" not in st.session_state:
            st.session_state.pickle_data = pickle_data
        else:
            st.session_state.pickle_data = pickle_data

        # Load the Langchain chatbot
        llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask your questions from PDF {uploaded_file.name}?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
