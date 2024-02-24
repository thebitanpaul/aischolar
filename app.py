import sys
import os
import streamlit as st
import codecs
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read HTML content
def read_html(file):
    html_content = file.getvalue().decode("utf-8")
    soup = BeautifulSoup(html_content, "html.parser")
    # Extract text from HTML
    text = soup.get_text(separator=" ")
    return text

# Load environment variables
load_dotenv()

# Main Streamlit app
def main():
    st.title("Query your PDF or HTML")
    with st.sidebar:
        st.title('Ai Scholar')
        st.markdown('''
        ## About
        Choose the desired PDF or HTML file, then perform a query.
        ''')

        # File uploader for uploading PDFs or HTML files
        uploaded_file = st.file_uploader("Upload PDF or HTML", type=["pdf", "html"])

    if uploaded_file is None:
        st.info("Please upload a file of type: " + ", ".join(["pdf", "html"]) + " to start analysing your data.")
        st.image("waitingForScholar.webp", use_column_width=True)
        return

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/html":
            text = read_html(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a PDF or HTML file.")
            return

        st.info("The content of the file is hidden. Type your query in the chat window.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot
        llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask your questions from {uploaded_file.name}?"):
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
