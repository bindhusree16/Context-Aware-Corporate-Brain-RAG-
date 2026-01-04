from fastapi import FastAPI, Request
from pydantic import BaseModel

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from fastapi.responses import JSONResponse

from Retrieval_Engine import load_vectorstore, load_llm, ask_question

# ================================
# RATE LIMITER
# ================================
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please try again later."}
    )

# ================================
# LOAD RAG COMPONENTS
# ================================
vectorstore = load_vectorstore()
llm = load_llm()

# ================================
# REQUEST MODEL
# ================================
class Question(BaseModel):
    question: str

# ================================
# RATE LIMITED ENDPOINT
# ================================
@app.post("/ask")
@limiter.limit("1/minute")
def ask_doc(data: Question, request: Request):
    answer, sources = ask_question(
        vectorstore,
        llm,
        data.question,
        return_sources=True
    )

    return {
        "question": data.question,
        "answer": answer,
        "sources": sources
    }
