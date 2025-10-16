from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import asyncio
from datetime import datetime
import numpy as np

# Import your database and crawler
from src.database.vector_database import AdvancedVectorDB, Document, SearchResult
from src.database.graph_database import GraphDatabase
from src.crawler.advanced_crawler import AdvancedWebCrawler, CrawlConfig

app = FastAPI(
    title="Advanced Search Engine API",
    description="Backend for the advanced search engine with web crawling and vector database",
    version="2.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize databases
vector_db = AdvancedVectorDB("./data")
graph_db = GraphDatabase()

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    max_results: int = 100
    min_quality: float = 0.0
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    query: str

class CrawlRequest(BaseModel):
    urls: List[str]
    max_pages: int = 1000
    max_depth: int = 3

class CrawlResponse(BaseModel):
    message: str
    crawl_id: str
    pages_crawled: int

class DocumentAddRequest(BaseModel):
    url: str
    title: str
    content: str
    meta_description: Optional[str] = ""
    categories: Optional[List[str]] = None

@app.get("/")
async def root():
    return FileResponse('index.html')

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = vector_db.get_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Advanced search endpoint"""
    start_time = datetime.utcnow()
    
    try:
        # Generate embedding for vector search (simplified - use real embedding model)
        query_vector = await generate_embedding(request.query)
        
        results = []
        
        if request.search_type == "hybrid":
            results = vector_db.hybrid_search(
                query=request.query,
                query_vector=query_vector,
                k=request.max_results
            )
        elif request.search_type == "vector":
            results = vector_db.search_similar(
                query_vector=query_vector,
                k=request.max_results,
                filters=request.filters
            )
        elif request.search_type == "text":
            results = vector_db.text_search(
                query=request.query,
                k=request.max_results
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        # Filter by quality
        filtered_results = [
            result for result in results 
            if result.document.quality_score >= request.min_quality
        ]
        
        # Convert to JSON-serializable format
        serializable_results = []
        for result in filtered_results:
            doc_dict = {
                "id": result.document.id,
                "url": result.document.url,
                "title": result.document.title,
                "content": result.document.content,
                "meta_description": result.document.meta_description,
                "snippet": result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                "timestamp": result.document.timestamp,
                "quality_score": round(result.document.quality_score, 3),
                "pagerank": round(result.document.pagerank, 3),
                "word_count": result.document.word_count,
                "categories": result.document.categories,
                "language": result.document.language,
                "score": round(result.score, 4),
                "source": "crawled"  # You can categorize based on domain
            }
            serializable_results.append(doc_dict)
        
        search_time = (datetime.utcnow() - start_time).total_seconds()
        
        return SearchResponse(
            results=serializable_results,
            total_count=len(serializable_results),
            search_time=search_time,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/crawl")
async def start_crawl(request: CrawlRequest):
    """Start a web crawl"""
    try:
        config = CrawlConfig(
            seed_urls=request.urls,
            max_pages=request.max_pages,
            max_depth=request.max_depth
        )
        
        crawler = AdvancedWebCrawler(config)
        
        # Run crawl in background
        asyncio.create_task(run_crawl(crawler))
        
        crawl_id = f"crawl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return CrawlResponse(
            message="Crawl started successfully",
            crawl_id=crawl_id,
            pages_crawled=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl error: {str(e)}")

async def run_crawl(crawler: AdvancedWebCrawler):
    """Run crawler in background"""
    try:
        await crawler.crawl()
        
        # After crawl completes, you might want to update indices
        await update_search_indices()
        
    except Exception as e:
        print(f"Crawl failed: {e}")

async def update_search_indices():
    """Update search indices after crawl"""
    # This could include:
    # - Recalculating PageRank
    # - Updating vector indices
    # - Refreshing inverted indexes
    print("Updating search indices...")

@app.post("/documents")
async def add_document(request: DocumentAddRequest):
    """Add a single document to the database"""
    try:
        # Generate document ID from URL
        doc_id = generate_document_id(request.url)
        
        # Generate embedding
        vector = await generate_embedding(request.content)
        
        document = Document(
            id=doc_id,
            url=request.url,
            title=request.title,
            content=request.content,
            meta_description=request.meta_description,
            timestamp=datetime.utcnow().isoformat(),
            vector=vector,
            quality_score=calculate_quality_score(request.content),
            word_count=len(request.content.split()),
            categories=request.categories or []
        )
        
        success = vector_db.add_document(document)
        
        if success:
            return {
                "status": "success",
                "message": "Document added successfully",
                "document_id": doc_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID"""
    try:
        document = vector_db.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "data": {
                "id": document.id,
                "url": document.url,
                "title": document.title,
                "content": document.content,
                "meta_description": document.meta_description,
                "timestamp": document.timestamp,
                "quality_score": document.quality_score,
                "pagerank": document.pagerank,
                "word_count": document.word_count,
                "categories": document.categories,
                "language": document.language
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@app.get("/similar/{document_id}")
async def get_similar_documents(document_id: str, limit: int = 10):
    """Get documents similar to a specific document"""
    try:
        document = vector_db.get_document(document_id)
        
        if not document or document.vector is None:
            raise HTTPException(status_code=404, detail="Document not found or no vector available")
        
        # Use graph-based similarity
        similar_docs = graph_db.find_similar_documents(document_id, limit)
        
        # Get full document details
        results = []
        for doc_id, similarity in similar_docs:
            doc = vector_db.get_document(doc_id)
            if doc:
                results.append({
                    "id": doc.id,
                    "url": doc.url,
                    "title": doc.title,
                    "snippet": doc.content[:150] + "..." if len(doc.content) > 150 else doc.content,
                    "similarity": round(similarity, 4),
                    "quality_score": doc.quality_score
                })
        
        return {
            "status": "success",
            "data": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "advanced-search-engine"
    }

# Utility functions
async def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for text - replace with real model"""
    # This is a simplified version - use SentenceBERT or similar in production
    words = text.lower().split()[:384]
    embedding = np.zeros(384)
    
    for i, word in enumerate(words):
        if i >= 384:
            break
        embedding[i] = hash(word) % 100 / 100.0
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def calculate_quality_score(text: str) -> float:
    """Calculate quality score for content"""
    words = text.split()
    if len(words) < 50:
        return 0.3
    
    # Simple heuristic - replace with more sophisticated analysis
    avg_word_length = sum(len(word) for word in words) / len(words)
    unique_ratio = len(set(words)) / len(words)
    
    score = min(avg_word_length / 10 * 0.5 + unique_ratio * 0.5, 1.0)
    return round(score, 3)

def generate_document_id(url: str) -> str:
    """Generate document ID from URL"""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]

# Mount static files (for serving HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info"
    )
