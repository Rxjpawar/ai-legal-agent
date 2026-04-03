from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_vectors"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

def retrieve_documents(query: str, k: int = 5):

    vector_db = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    results = vector_db.similarity_search(query, k=k)

    context_parts = []

    for result in results:
        context_parts.extend([
            f"Content: {result.page_content}",
            f"Page: {result.metadata.get('page_label', '')}",
            f"Source: {result.metadata.get('source', '')}"
        ])

    return "\n\n".join(context_parts)