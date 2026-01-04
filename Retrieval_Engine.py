
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# =====================================================
# CONFIGURATION (MUST MATCH WEEK-1)
# =====================================================

INDEX_NAME = "documind-week1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not set")

# =====================================================
# LOAD VECTORSTORE (FROM WEEK-1 INDEX)
# =====================================================

def load_vectorstore():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding,
        text_key="text"
    )

    return vectorstore

# =====================================================
# LOAD HUGGINGFACE LLM (NO API KEY)
# =====================================================

def load_llm():     #from transformers
    hf_pipe = pipeline(                        #readymade wrapper - instead of writing tokenizer code,model loading,decoding logic ,just we say here is txt- give output
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0
    )

    return HuggingFacePipeline(pipeline=hf_pipe)

# =====================================================
# STRICT ENTERPRISE PROMPT (ANTI-HALLUCINATION)
# =====================================================

PROMPT_TEMPLATE = """
Answer:
You are an enterprise document assistant.

- Use ONLY the context provided.
- DO NOT guess, DO NOT infer, DO NOT use outside knowledge.
- If the answer is not explicitly present in the context, respond exactly: "I don’t know. This information is not present in the documents."

Context:
{context}

Question:
{question}

Answer:

"""

# =====================================================
# WEEK-2 QUESTION ANSWERING LOGIC
# =====================================================
'''
def ask_question(vectorstore, llm, question: str):
    # 1️⃣ Retrieve relevant chunks
    #docs = vectorstore.similarity_search(question, k=3)
    results = vectorstore.similarity_search(question, k=3)
    #results = [doc for doc, score in results if score > 0.5]
    # Filter by similarity (e.g., > 0.5)
    #results = [doc for doc in results if doc.score > 0.5]
    results = [item[0] if isinstance(item, (tuple, list)) else item for item in results]
    if not results:
        print("\nI don’t know. This information is not present in the provided documents.")
        return

    # 2️⃣ Build context
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    # 3️⃣ Generate answer
    response = llm.invoke(prompt)

    # 4️⃣ Print output
    print("\n💬 Question:")
    print(question)

    print("\n✅ Answer:")
    print(response)

    print("\n📄 Sources:")
    for doc in results:
        print(f"- {doc.metadata.get('source')} | Page {doc.metadata.get('page')}")'''
def ask_question(vectorstore, llm, question: str, return_sources=False):
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        answer = "I don’t know. This information is not present in the documents."
        return (answer, []) if return_sources else answer

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in results
    ]

    return (response, sources) if return_sources else response


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("🚀 Starting Week-2 RAG Pipeline (FINAL VERSION)...")

    vectorstore = load_vectorstore()
    llm = load_llm()

    ask_question(vectorstore, llm, "what is the stipend?")
    ask_question(vectorstore, llm, "do you like carrot?")

    print("\n✅ Week-2 completed successfully")
