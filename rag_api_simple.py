"""
FastAPI RAG API for ChromaDB - Free Hosting Compatible
Works with Render, Fly.io, Railway, etc.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
import os

app = FastAPI(
    title="Company RAG API",
    description="Query company knowledge base",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB setup
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
print(f"Loading ChromaDB from: {CHROMA_PATH}")

try:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="federal_contracting",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ ChromaDB loaded: {collection.count()} documents")
except Exception as e:
    print(f"✗ ChromaDB error: {e}")
    collection = None


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    filter: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    query: str


class AddDocumentRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None


class AddDocumentResponse(BaseModel):
    message: str
    added: int
    total_documents: int


@app.get("/")
def home():
    return {
        "service": "Company RAG API",
        "status": "running",
        "documents": collection.count() if collection else 0
    }


@app.get("/health")
def health():
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    return {
        "status": "healthy",
        "chromadb": "connected",
        "documents": collection.count(),
        "collection": "federal_contracting"
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Query company knowledge base."""
    
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        # Query ChromaDB
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results,
            where=request.filter
        )
        
        # Format results
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "id": results['ids'][0][i] if results['ids'] else None
                })
        
        return QueryResponse(
            results=formatted_results,
            count=len(formatted_results),
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/add", response_model=AddDocumentResponse)
def add_documents(request: AddDocumentRequest):
    """Add documents to the knowledge base."""
    
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        # Generate IDs if not provided
        if not request.ids:
            import uuid
            request.ids = [str(uuid.uuid4()) for _ in request.documents]
        
        # Add documents to ChromaDB
        collection.add(
            documents=request.documents,
            metadatas=request.metadatas,
            ids=request.ids
        )
        
        return AddDocumentResponse(
            message="Documents added successfully",
            added=len(request.documents),
            total_documents=collection.count()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add failed: {str(e)}")


@app.get("/stats")
def get_stats():
    """Get database statistics."""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    return {
        "total_documents": collection.count(),
        "collection_name": collection.name,
        "metadata": collection.metadata
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
