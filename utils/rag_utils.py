from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

def create_rag_index(docs, pdf_name: str):
    """
    Create or load Pinecone index from document chunks using HuggingFace embeddings.
    """
    try:
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "quizer-index")
        
        if not api_key:
            print("Error: PINECONE_API_KEY environment variable not set.")
            return None
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Initialize embeddings (lightweight model)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create PineconeVectorStore and upsert documents
        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name,
            namespace=pdf_name.replace(" ", "_").lower()
        )
        
        print(f"Upserted documents to Pinecone index '{index_name}' in namespace '{pdf_name}'")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating Pinecone vector store: {str(e)}")
        return None

def retrieve_context(vectorstore, query: str, k: int = 4):
    """
    Retrieve relevant chunks for RAG.
    """
    if vectorstore is None:
        print("Warning: Vectorstore is None. Using generic context.")
        return "No document context available. Generate general questions based on the topic."
    
    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            print(f"Warning: No documents found for query: {query}")
            return "No relevant documents found in the vector store. Generate general questions based on the topic."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return "Error retrieving document context. Generate general questions based on the topic."