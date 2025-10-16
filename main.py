from src.database.vector_database import AdvancedVectorDB, Document
from src.database.graph_database import GraphDatabase
import numpy as np

def main():
    # Initialize databases
    vector_db = AdvancedVectorDB("./data")
    graph_db = GraphDatabase()
    
    # Create sample document
    doc = Document(
        id="doc_1",
        url="https://example.com",
        title="Example Document",
        content="This is an example document for testing the database system.",
        meta_description="Test document",
        timestamp="2024-01-01T00:00:00Z",
        vector=np.random.rand(384).astype('float32'),
        quality_score=0.8,
        word_count=10,
        categories=["test", "example"]
    )
    
    # Add to database
    vector_db.add_document(doc)
    
    # Search
    results = vector_db.hybrid_search(
        query="example document",
        query_vector=doc.vector,
        k=10
    )
    
    print(f"Found {len(results)} results")
    
    # Get stats
    stats = vector_db.get_stats()
    print("Database stats:", stats)
    
    # Cleanup
    vector_db.close()

if __name__ == "__main__":
    main()
