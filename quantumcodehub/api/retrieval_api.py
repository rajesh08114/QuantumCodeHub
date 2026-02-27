from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
import uvicorn

from retrieval.hybrid_retriever import HybridRetriever
from models.document import RetrievalQuery, RetrievalResult
from config.constants import Framework, Intent

app = FastAPI(title="QuantumCodeHub Retrieval API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever
retriever = HybridRetriever()

class QueryRequest(BaseModel):
    query: str
    framework: Optional[str] = None
    version: Optional[str] = None
    intent: Optional[str] = None
    top_k: int = 10
    include_deprecated: bool = False

class QueryResponse(BaseModel):
    framework: Optional[str]
    version: Optional[str]
    intent: str
    documents: List[dict]
    deprecation_detected: bool
    deprecation_info: Optional[dict]
    query_time_ms: float
    total_results: int

@app.get("/")
async def root():
    return {
        "service": "QuantumCodeHub Retrieval API",
        "version": "1.0.0",
        "frameworks": [f.value for f in Framework],
        "intents": [i.value for i in Intent]
    }

@app.post("/retrieve", response_model=QueryResponse)
async def retrieve(request: QueryRequest):
    """Retrieve documents based on query"""
    try:
        # Convert string framework to enum
        framework = None
        if request.framework:
            try:
                framework = Framework(request.framework.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid framework. Must be one of: {[f.value for f in Framework]}"
                )
        
        # Convert string intent to enum
        intent = None
        if request.intent:
            try:
                intent = Intent(request.intent.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid intent. Must be one of: {[i.value for i in Intent]}"
                )
        
        # Create query object
        query = RetrievalQuery(
            query=request.query,
            framework=framework,
            version=request.version,
            intent=intent,
            top_k=request.top_k,
            include_deprecated=request.include_deprecated
        )
        
        # Perform retrieval
        result = retriever.retrieve(query)
        
        # Convert to response
        return QueryResponse(
            framework=result.framework.value if result.framework else None,
            version=result.version,
            intent=result.intent.value,
            documents=result.documents,
            deprecation_detected=result.deprecation_detected,
            deprecation_info=result.deprecation_info,
            query_time_ms=result.query_time_ms,
            total_results=result.total_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve")
async def retrieve_get(
    query: str = Query(..., description="Search query"),
    framework: Optional[str] = Query(None, description="Framework name"),
    version: Optional[str] = Query(None, description="Version number"),
    intent: Optional[str] = Query(None, description="Query intent"),
    top_k: int = Query(10, description="Number of results to return"),
    include_deprecated: bool = Query(False, description="Include deprecated APIs")
):
    """Retrieve documents (GET version)"""
    request = QueryRequest(
        query=query,
        framework=framework,
        version=version,
        intent=intent,
        top_k=top_k,
        include_deprecated=include_deprecated
    )
    return await retrieve(request)

@app.get("/stats")
async def get_stats():
    """Get retrieval system statistics"""
    return {
        "collections": retriever.get_collection_stats(),
        "ingestion": retriever.chroma_manager.get_collection_stats("all")
    }

@app.post("/reset")
async def reset_system():
    """Reset all collections (for testing)"""
    retriever.chroma_manager.reset()
    return {"message": "System reset successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)