
import os
import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# =====================================================
# CONFIGURATION
# =====================================================

PDF_PATH = "C:\\Users\\bindh\\Downloads\\Moreddy Bindhu sree-OfferLetter (2).pdf"  # path to enterprise PDF
INDEX_NAME = "documind-week1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_ENV = os.getenv("PINECONE_ENV")  # ex: "us-east-1"
PINECONE_REGION = "us-east-1"   # or the region shown in Pinecone console


if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not set")

# =====================================================
# 1. LOAD PDF DOCUMENTS
# =====================================================

def load_documents(pdf_path: str):
    """Load PDF and return LangChain Documents with metadata."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# =====================================================
# 2. CHUNK DOCUMENTS
# =====================================================

def chunk_documents(documents):
    """Split documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)
    return chunks

# =====================================================
# 3. CREATE EMBEDDINGS
# =====================================================

def create_embedding_model():
    """Initialize HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# =====================================================
# 4. INITIALIZE / GET PINECONE INDEX
# =====================================================

def init_pinecone_index(index_name: str, dimension: int):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )

        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"ℹ️ Using existing Pinecone index: {index_name}")

    return pc.Index(index_name)

# =====================================================
# 5. UPSERT VECTORS
# =====================================================

def upsert_vectors(index, chunks, embedding_model):
    vectors = []

    for chunk in chunks:
        vector_id = str(uuid.uuid4())
        embedding = embedding_model.embed_query(chunk.page_content)

        metadata = {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", -1)+1,
            "text": chunk.page_content
        }

        vectors.append((vector_id, embedding, metadata))

    index.upsert(vectors)
    print(f"✅ Upserted {len(vectors)} vectors to Pinecone")

# =====================================================
# 6. TEST RETRIEVAL (WEEK-1 VERIFICATION)
# =====================================================

def test_retrieval(index, embedding_model, query: str, top_k: int = 3):
    query_embedding = embedding_model.embed_query(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    print("\n🔍 Retrieval Results:\n")
    for match in result.matches:
        print(f"Score: {match.score}")
        print(f"Source: {match.metadata['source']}")
        print(f"Page: {match.metadata['page']}")
        print(f"Text: {match.metadata['text'][:300]}...")
        print("-" * 60)

# =====================================================
# MAIN PIPELINE (WEEK-1 EXECUTION)
# =====================================================

if __name__ == "__main__":
    print("🚀 Starting Week-1 Ingestion Pipeline...")

    docs = load_documents(PDF_PATH)
    chunks = chunk_documents(docs)

    embedding_model = create_embedding_model()
    index = init_pinecone_index(
        INDEX_NAME,
        dimension=len(embedding_model.embed_query("test"))
    )

    upsert_vectors(index, chunks, embedding_model)

    # Mandatory Week-1 verification
    test_retrieval(index, embedding_model, "do they provide job assistance?")

    print("\n✅ Week-1 ingestion completed successfully")
