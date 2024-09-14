Advanced Document-Based Chatbot with LangChain and FAISS Integration
Description:

Developed a sophisticated chatbot application that leverages the power of document embeddings and advanced natural language processing techniques. The system is built using Streamlit, LangChain, and FAISS, enabling dynamic interaction with user-uploaded documents and providing accurate, context-aware responses.
Features:

    Document Embedding Generation: Converts user-uploaded PDFs into vector embeddings using HuggingFace's sentence-transformer models, allowing for efficient document-based querying.
    Contextual Question Answering: Utilizes LangChain’s ChatGroq model to generate precise answers based on the provided document context.
    Interactive Chat Interface: Streamlit-based user interface that allows seamless document upload, question input, and response display, with real-time chat history and timestamps.
    Document Similarity Search: Displays similar document chunks using FAISS and Streamlit expanders to help users understand relevant sections related to their queries.
    Session Management: Implements session state management to handle chat history, context updates, and chat session control, including restart functionality.

Advanced Functionalities:

    Contextual Response Enhancement: Integrates LangChain's prompt templates to enhance response accuracy by dynamically incorporating previous context and user queries.
    Performance Monitoring: Measures and displays response times for document retrieval and processing, optimizing performance insights.
    Error Handling and Robustness: Includes comprehensive error handling to manage PDF extraction issues and query processing errors, ensuring smooth user experience.
    Custom Document Splitting: Employs RecursiveCharacterTextSplitter for efficient text chunking, facilitating scalable and precise document segmentation.

Libraries and Technologies:

    Streamlit: For creating a user-friendly web interface.
    LangChain: For advanced document-based question answering and retrieval.
    FAISS: For efficient vector storage and similarity search.
    HuggingFace Transformers: For generating high-quality document embeddings.
    PyMuPDF: For extracting text from PDF files.

This project demonstrates expertise in integrating cutting-edge NLP technologies and building robust, scalable applications for real-world document-based interactions.
