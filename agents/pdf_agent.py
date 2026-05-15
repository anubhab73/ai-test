from utils.pdf_utils import load_and_split_pdf
from utils.rag_utils import create_rag_index
from dotenv import load_dotenv
import os

load_dotenv()

def process_pdf(pdf_path: str, pdf_name: str):
    """
    Process PDF and build RAG index.
    """
    docs = load_and_split_pdf(pdf_path)
    vectorstore = create_rag_index(docs, pdf_name)
    return vectorstore