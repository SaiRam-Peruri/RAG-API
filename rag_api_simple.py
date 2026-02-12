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
    
    # Load both collections (authoritative and drafting)
    collection_auth = client.get_or_create_collection(
        name="authoritative",
        metadata={"hnsw:space": "cosine"}
    )
    collection_draft = client.get_or_create_collection(
        name="drafting",
        metadata={"hnsw:space": "cosine"}
    )
    
    total_docs = collection_auth.count() + collection_draft.count()
    print(f"✓ ChromaDB loaded: {total_docs} documents")
    print(f"  - authoritative: {collection_auth.count()}")
    print(f"  - drafting: {collection_draft.count()}")
    collection = collection_auth  # Default to authoritative for queries
    
except Exception as e:
    print(f"✗ ChromaDB error: {e}")
    collection = None
    collection_auth = None
    collection_draft = None


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
    collection: str = "authoritative"  # or "drafting"


class AddDocumentResponse(BaseModel):
    message: str
    added: int
    total_documents: int


@app.get("/")
def home():
    auth_count = collection_auth.count() if collection_auth else 0
    draft_count = collection_draft.count() if collection_draft else 0
    return {
        "service": "Company RAG API",
        "status": "running",
        "documents": {
            "authoritative": auth_count,
            "drafting": draft_count,
            "total": auth_count + draft_count
        }
    }


@app.get("/health")
def health():
    if not collection_auth and not collection_draft:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    auth_count = collection_auth.count() if collection_auth else 0
    draft_count = collection_draft.count() if collection_draft else 0
    
    return {
        "status": "healthy",
        "chromadb": "connected",
        "documents": {
            "authoritative": auth_count,
            "drafting": draft_count,
            "total": auth_count + draft_count
        },
        "collections": ["authoritative", "drafting"]
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Query company knowledge base (searches both collections)."""
    
    if not collection_auth and not collection_draft:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        all_results = []
        
        # Query authoritative collection (government docs)
        if collection_auth:
            results_auth = collection_auth.query(
                query_texts=[request.query],
                n_results=request.n_results,
                where=request.filter
            )
            
            if results_auth and results_auth['documents'] and results_auth['documents'][0]:
                for i in range(len(results_auth['documents'][0])):
                    all_results.append({
                        "content": results_auth['documents'][0][i],
                        "metadata": results_auth['metadatas'][0][i] if results_auth['metadatas'] else {},
                        "distance": results_auth['distances'][0][i] if results_auth['distances'] else None,
                        "id": results_auth['ids'][0][i] if results_auth['ids'] else None,
                        "collection": "authoritative"
                    })
        
        # Query drafting collection (vendor docs)
        if collection_draft:
            results_draft = collection_draft.query(
                query_texts=[request.query],
                n_results=request.n_results,
                where=request.filter
            )
            
            if results_draft and results_draft['documents'] and results_draft['documents'][0]:
                for i in range(len(results_draft['documents'][0])):
                    all_results.append({
                        "content": results_draft['documents'][0][i],
                        "metadata": results_draft['metadatas'][0][i] if results_draft['metadatas'] else {},
                        "distance": results_draft['distances'][0][i] if results_draft['distances'] else None,
                        "id": results_draft['ids'][0][i] if results_draft['ids'] else None,
                        "collection": "drafting"
                    })
        
        # Sort by distance (lower is better) and take top N
        all_results.sort(key=lambda x: x['distance'] if x['distance'] is not None else float('inf'))
        all_results = all_results[:request.n_results]
        
        return QueryResponse(
            results=all_results,
            count=len(all_results),
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/add", response_model=AddDocumentResponse)
def add_documents(request: AddDocumentRequest):
    """Add documents to the knowledge base."""
    
    if not collection_auth and not collection_draft:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    # Select target collection
    if request.collection == "authoritative":
        target_coll = collection_auth
    elif request.collection == "drafting":
        target_coll = collection_draft
    else:
        raise HTTPException(status_code=400, detail="Invalid collection. Use 'authoritative' or 'drafting'")
    
    if not target_coll:
        raise HTTPException(status_code=503, detail=f"Collection {request.collection} not available")
    
    try:
        # Generate IDs if not provided
        if not request.ids:
            import uuid
            request.ids = [str(uuid.uuid4()) for _ in request.documents]
        
        # Add documents to ChromaDB
        target_coll.add(
            documents=request.documents,
            metadatas=request.metadatas,
            ids=request.ids
        )
        
        auth_count = collection_auth.count() if collection_auth else 0
        draft_count = collection_draft.count() if collection_draft else 0
        
        return AddDocumentResponse(
            message=f"Documents added to {request.collection} successfully",
            added=len(request.documents),
            total_documents=auth_count + draft_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add failed: {str(e)}")


@app.get("/stats")
def get_stats():
    """Get database statistics."""
    if not collection_auth and not collection_draft:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    auth_count = collection_auth.count() if collection_auth else 0
    draft_count = collection_draft.count() if collection_draft else 0
    
    return {
        "total_documents": auth_count + draft_count,
        "collections": {
            "authoritative": {
                "count": auth_count,
                "metadata": collection_auth.metadata if collection_auth else None
            },
            "drafting": {
                "count": draft_count,
                "metadata": collection_draft.metadata if collection_draft else None
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
