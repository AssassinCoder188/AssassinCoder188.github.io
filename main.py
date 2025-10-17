import sys
import os

# Add src to Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.vector_database import AdvancedVectorDB
from src.database.graph_database import GraphDatabase
from src.database.document_store import DocumentStore
from src.crawler.advanced_crawler import AdvancedWebCrawler, CrawlConfig
import yaml
import asyncio

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_data_path():
    """Get the correct data path for PythonAnywhere"""
    # On PythonAnywhere, use absolute path
    # On local machine, use relative path
    if 'pythonanywhere' in os.environ.get('HOME', ''):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'data')
    else:
        return "./data"

def main():
    print("🚀 Starting Advanced Search Engine...")
    
    # Load configuration
    config = load_config()
    
    # Get the correct data path
    data_path = get_data_path()
    print(f"📁 Using data path: {data_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'vectors'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'documents'), exist_ok=True)
    
    # Initialize databases with correct path
    vector_db = AdvancedVectorDB(data_path)
    graph_db = GraphDatabase()
    doc_store = DocumentStore()
    
    print("✅ Databases initialized!")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Start web crawler")
    print("2. Start web server (FastAPI)")
    print("3. Just initialize databases")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Start crawling
        print("🚀 Starting web crawler...")
        crawler_config = CrawlConfig(
            seed_urls=config.get('seed_urls', [
                'https://news.ycombinator.com',
                'https://github.com',
                'https://stackoverflow.com'
            ]),
            max_pages=config.get('crawler', {}).get('max_pages', 1000),
            max_depth=config.get('crawler', {}).get('max_depth', 3),
            output_file=os.path.join(data_path, 'crawled_data.jsonl')
        )
        
        crawler = AdvancedWebCrawler(crawler_config)
        asyncio.run(crawler.crawl())
        
    elif choice == "2":
        # Start FastAPI server
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title="Advanced Search Engine")
        
        @app.get("/")
        async def root():
            return {"message": "Search Engine API is running!"}
        
        @app.get("/search")
        async def search(q: str = "python"):
            # Simple search endpoint
            results = vector_db.text_search(q, 10)
            return {
                "query": q,
                "results": [
                    {
                        "title": result.document.title,
                        "url": result.document.url,
                        "snippet": result.document.content[:200] + "...",
                        "score": result.score
                    }
                    for result in results
                ]
            }
        
        print("🌐 Starting FastAPI server...")
        print("📱 Access at: http://localhost:8000")
        print("🔗 API docs: http://localhost:8000/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    else:
        # Just initialize and exit
        print("✅ Databases ready!")
        print("🌐 Open index.html in your browser to use the search engine")
        
        # Keep running so user can open the web interface
        try:
            input("Press Enter to exit...")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
