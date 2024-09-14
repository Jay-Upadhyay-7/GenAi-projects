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
from langchain.schema import Document
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('API_KEY')

# Streamlit UI setup
st.title("ChatGroq Demo with HuggingFace Embeddings")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Initialize session state for chat history, context, and buttons
if "history" not in st.session_state:
    st.session_state.history = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Function to generate embeddings and store vectors
def vector_embeddings(file):
    if st.session_state.vectors is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        try:
            # Open the uploaded PDF file
            doc = fitz.open(stream=file.read(), filetype='pdf')
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

        st.session_state.docs = [Document(page_content=text)]
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        st.write(f"Split into {len(st.session_state.final_documents)} document chunks")
        
        if len(st.session_state.final_documents) == 0:
            st.write("No document chunks created. Check the text splitting logic.")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader for user to upload the PDF document
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Display the appropriate button
if uploaded_file is not None and st.session_state.vectors is None:
    if st.button("Create Document Embeddings"):
        vector_embeddings(uploaded_file)
        st.write("Vector store (FAISS) is ready.")
        st.session_state.chat_started = True
elif not st.session_state.chat_started and st.session_state.vectors is not None:
    if st.button("Start Chat"):
        st.session_state.chat_started = True

# Show chat history with timestamps
if st.session_state.history:
    st.write("### Chat History")
    for entry in st.session_state.history:
        timestamp = entry.get('timestamp', 'N/A')  # Provide a default value if 'timestamp' is missing
        st.write(f"**[{timestamp}] Question:** {entry['question']}")
        st.write(f"**Response:** {entry['response']}")
        st.write("---")

# Add a restart button to clear history and context
if st.button("Restart Chat"):
    st.session_state.history = []
    st.session_state.context = ""
    st.session_state.vectors = None
    st.session_state.chat_started = False
    st.write("Chat reset. Please upload a new document to start again.")

# If chat is started, show input field and move it below the response
if st.session_state.chat_started:
    # Input field for user questions
    prompt1 = st.text_input("Enter your Question")

    # Button to process the user question
    if st.button("Submit"):
        response_text = "Unable to process the request."
        response = {}  # Initialize response to avoid NameError

        if prompt1:
            # Check for greetings
            greetings = ["hi", "hello", "hey", "greetings"]
            if prompt1.lower() in greetings:
                response_text = "Hello! How can I assist you with questions related to the document?"
            else:
                # Add the previous context to the prompt
                context = st.session_state.context
                prompt = prompt_template.format(input=prompt1, context=context)
                
                # Create document chain and retrieval chain
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                try:
                    start = time.process_time()
                    response = retrieval_chain.invoke({'input': prompt1})
                    response_time = time.process_time() - start

                    response_text = response.get('answer', "No relevant information found in the document.")
                    
                    # Update context and history with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.context += f"\nQuestion: {prompt1}\nAnswer: {response_text}"
                    st.session_state.history.append({
                        'timestamp': timestamp,
                        'question': prompt1,
                        'response': response_text
                    })

                    st.write(f"Response time: {response_time:.2f} seconds")
                except Exception as e:
                    response_text = "An error occurred while processing your question."
                    st.write(response_text)

        st.write(response_text)

        # Display similar documents using Streamlit expander
        if isinstance(response, dict) and "context" in response:
            with st.expander("Document Similarity Search"):
                for doc in response.get("context", []):  # Adjust based on actual response structure
                    st.write(doc.page_content)
                    st.write("--------------------------------")
