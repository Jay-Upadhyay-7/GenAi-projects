# ChatGroq Demo with HuggingFace Embeddings
import streamlit as st
import os
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Correct import path for Document
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('API_KEY')

# Streamlit UI setup
st.title("ChatGroq Demo with HuggingFace Embeddings")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to generate embeddings and store vectors
def vector_embeddings():
    if "vectors" not in st.session_state:
        # Use HuggingFace embeddings (free alternative)
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load documents from the PDF file
        text_file_path = "/Users/jay/Downloads/Undertaking.pdf"  # Update path as necessary
        
        try:
            doc = fitz.open(text_file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            st.write(f"Error reading PDF file: {e}")
            return

        if len(text) == 0:
            st.write("No text extracted from the file.")
            return

        # Create a list of Document objects from the text
        st.session_state.docs = [Document(page_content=text)]
        
        # Split the documents into smaller chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting

        # Debug: Check if documents were split
        st.write(f"Split into {len(st.session_state.final_documents)} document chunks")
        
        # Check if any chunks were created
        if len(st.session_state.final_documents) == 0:
            st.write("No document chunks created. Check the text splitting logic.")
            return

        # Create a FAISS vector store using the HuggingFace embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input field for user questions
prompt1 = st.text_input("Enter your Question")

# Button to trigger the embedding creation process
if st.button("Create Document Embeddings"):
    vector_embeddings()
    st.write("Vector store (FAISS) is ready.")

# Check if vectors have been created before proceeding
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])

    # Display similar documents using Streamlit expander
    with st.expander("Document Similarity Search"):
        for doc in response["context"]:  # Assuming response["context"] contains Document objects
            st.write(doc.page_content)  # Access the content using the page_content attribute
            st.write("--------------------------------")
else:
    st.write("Please create document embeddings first.")
