import networkx as nx
import sqlite3
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class GraphNode:
    id: str
    type: str  # 'document', 'concept', 'entity'
    properties: Dict[str, Any]
    pagerank: float = 0.0
    centrality: float = 0.0

class GraphDatabase:
    def __init__(self, db_path: str = "./data/graph.db"):
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self.setup_sqlite()
        self.load_graph()
    
    def setup_sqlite(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                properties TEXT,
                pagerank REAL DEFAULT 0.0,
                centrality REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id TEXT,
                target_id TEXT,
                relationship TEXT,
                weight REAL DEFAULT 1.0,
                properties TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id, relationship),
                FOREIGN KEY (source_id) REFERENCES graph_nodes (id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes (id)
            )
        ''')
        
        self.conn.commit()
    
    def add_document_relationships(self, source_doc_id: str, target_doc_ids: List[str]):
        """Add document linking relationships"""
        for target_id in target_doc_ids:
            self.graph.add_edge(source_doc_id, target_id, relationship='links_to')
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO graph_edges 
                (source_id, target_id, relationship, weight)
                VALUES (?, ?, ?, ?)
            ''', (source_doc_id, target_id, 'links_to', 1.0))
        
        self.conn.commit()
    
    def calculate_pagerank(self):
        """Calculate PageRank for all nodes"""
        pagerank_scores = nx.pagerank(self.graph)
        
        cursor = self.conn.cursor()
        for node_id, score in pagerank_scores.items():
            cursor.execute(
                'UPDATE graph_nodes SET pagerank = ? WHERE id = ?',
                (score, node_id)
            )
        
        self.conn.commit()
        return pagerank_scores
    
    def find_similar_documents(self, doc_id: str, max_results: int = 10):
        """Find similar documents using graph algorithms"""
        if doc_id not in self.graph:
            return []
        
        # Use Jaccard similarity on neighbors
        similarities = []
        doc_neighbors = set(self.graph.neighbors(doc_id))
        
        for other_id in self.graph.nodes():
            if other_id == doc_id:
                continue
            
            other_neighbors = set(self.graph.neighbors(other_id))
            intersection = len(doc_neighbors.intersection(other_neighbors))
            union = len(doc_neighbors.union(other_neighbors))
            
            if union > 0:
                similarity = intersection / union
                similarities.append((other_id, similarity))
        
        # Return top similar documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def save_graph(self):
        """Save graph to disk"""
        with open('./data/graphs/main_graph.pkl', 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self):
        """Load graph from disk"""
        try:
            with open('./data/graphs/main_graph.pkl', 'rb') as f:
                self.graph = pickle.load(f)
        except FileNotFoundError:
            self.graph = nx.DiGraph()
