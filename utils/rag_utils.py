from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import math
import os
import re
import time

load_dotenv()


DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_EMBEDDING_DIMENSION = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", os.getenv("PINECONE_INDEX_DIMENSION", "384")))
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quizer-index")
DEFAULT_INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))
DEFAULT_INDEX_METRIC = os.getenv("PINECONE_METRIC", "cosine")
DEFAULT_PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def _normalize_vector(values: list[float]) -> list[float]:
    """
    Normalize embeddings so cosine similarity behaves consistently across providers.
    """
    vector = [float(value) for value in values]
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _read_attr(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_namespace(pdf_name: str) -> str:
    """
    Create a Pinecone-safe namespace for a PDF.
    """
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (pdf_name or "default").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "default"


class GeminiPineconeEmbeddings(Embeddings):
    """
    Hosted Gemini embeddings for Pinecone-backed retrieval.
    """

    def __init__(self, api_key: str, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self.client = genai.Client(api_key=api_key)

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        prepared_texts = [(text or " ").strip() or " " for text in texts]
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension,
        )

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=prepared_texts,
            config=config,
        )

        embeddings = _read_attr(response, "embeddings", [])
        if len(embeddings) != len(prepared_texts):
            raise ValueError(
                f"Expected {len(prepared_texts)} embeddings from {self.model_name}, got {len(embeddings)}."
            )

        return [_normalize_vector(_read_attr(item, "values", [])) for item in embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]


def _wait_for_index_ready(pc: Pinecone, index_name: str, timeout_seconds: int = 120) -> None:
    """
    Poll Pinecone until the index is ready for data operations.
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        index_model = pc.describe_index(name=index_name)
        status = _read_attr(index_model, "status", {})
        if _read_attr(status, "ready", False):
            return
        time.sleep(2)

    raise TimeoutError(f"Pinecone index '{index_name}' was not ready within {timeout_seconds} seconds.")


def _ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str) -> int:
    """
    Create the Pinecone index if needed and ensure its dimension matches the embedding size.
    """
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=DEFAULT_PINECONE_CLOUD,
                region=DEFAULT_PINECONE_REGION,
            ),
            deletion_protection="disabled",
        )

    index_model = pc.describe_index(name=index_name)
    existing_dimension = _read_attr(index_model, "dimension")
    existing_metric = _read_attr(index_model, "metric")

    if existing_dimension != dimension:
        print(
            f"Info: Pinecone index '{index_name}' already uses dimension {existing_dimension}. "
            f"Using that dimension instead of requested {dimension}."
        )

    if existing_metric and str(existing_metric).lower() != metric.lower():
        raise ValueError(
            f"Pinecone index '{index_name}' metric mismatch: index={existing_metric}, expected={metric}."
        )

    _wait_for_index_ready(pc, index_name)
    return int(existing_dimension)


def create_rag_index(docs, pdf_name: str):
    """
    Create or load a Pinecone index using hosted Gemini embeddings.
    """
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY environment variable not set.")
            return None

        if not gemini_api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return None

        namespace = _build_namespace(pdf_name)
        pc = Pinecone(api_key=pinecone_api_key)
        resolved_dimension = _ensure_index(pc, DEFAULT_INDEX_NAME, DEFAULT_INDEX_DIMENSION, DEFAULT_INDEX_METRIC)

        embeddings = GeminiPineconeEmbeddings(
            api_key=gemini_api_key,
            model_name=DEFAULT_EMBEDDING_MODEL,
            dimension=resolved_dimension,
        )

        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=DEFAULT_INDEX_NAME,
            namespace=namespace,
        )

        print(
            f"Upserted documents to Pinecone index '{DEFAULT_INDEX_NAME}' in namespace '{namespace}' "
            f"using {DEFAULT_EMBEDDING_MODEL} ({resolved_dimension}d)."
        )

        return PineconeVectorStore(
            index_name=DEFAULT_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
            pinecone_api_key=pinecone_api_key,
        )

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
