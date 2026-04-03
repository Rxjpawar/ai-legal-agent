from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import TextIndexParams

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_vectors"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

def build_vector_store():

    data_path = Path("data/legal_corpus")

    documents = []

    for pdf_file in data_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)

    QdrantVectorStore.from_documents(
        documents=split_docs,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    client = QdrantClient(QDRANT_URL)

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=TextIndexParams(type="text")
    )

    print("legal documents indexed successfully")