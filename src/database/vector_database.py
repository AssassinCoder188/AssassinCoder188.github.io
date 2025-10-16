import numpy as np
import faiss
import sqlite3
import pickle
import json
import zlib
import leveldb
import lmdb
import rocksdb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from datetime import datetime
import logging
from pathlib import Path

@dataclass
class Document:
    id: str
    url: str
    title: str
    content: str
    meta_description: str
    timestamp: str
    vector: Optional[np.ndarray] = None
    pagerank: float = 1.0
    quality_score: float = 0.0
    word_count: int = 0
    language: str = "en"
    categories: List[str] = None
    entities: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.entities is None:
            self.entities = {}

@dataclass
class SearchResult:
    document: Document
    score: float
    explanation: Dict[str, float]

class AdvancedVectorDB:
    def __init__(self, base_path: str = "./data", dimension: int = 384):
        self.base_path = Path(base_path)
        self.dimension = dimension
        self.setup_directories()
        
        # Multiple storage engines
        self.setup_sqlite()
        self.setup_faiss()
        self.setup_leveldb()
        self.setup_lmdb()
        
        # Cache layer
        self.cache = {}
        self.cache_size = 10000
        
        # Threading
        self.lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Statistics
        self.stats = {
            'queries': 0,
            'inserts': 0,
            'hits': 0,
            'misses': 0
        }
        
        self.setup_logging()
    
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.base_path / "vectors",
            self.base_path / "graphs", 
            self.base_path / "documents",
            self.base_path / "indexes",
            self.base_path / "leveldb",
            self.base_path / "lmdb"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_sqlite(self):
        """Initialize SQLite with advanced schema"""
        self.sqlite_path = self.base_path / "search_engine.db"
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-10000")  # 10MB cache
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        cursor = self.conn.cursor()
        
        # Main documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                meta_description TEXT,
                timestamp DATETIME,
                pagerank REAL DEFAULT 1.0,
                quality_score REAL DEFAULT 0.0,
                word_count INTEGER,
                language TEXT DEFAULT 'en',
                categories TEXT,
                entities TEXT,
                compression_type TEXT DEFAULT 'zlib',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Links graph for PageRank
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT,
                target_id TEXT,
                link_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES documents (id),
                FOREIGN KEY (target_id) REFERENCES documents (id)
            )
        ''')
        
        # Inverted index for text search
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inverted_index (
                term TEXT,
                document_id TEXT,
                tf REAL,
                idf REAL,
                positions TEXT,
                field TEXT DEFAULT 'content',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (term, document_id, field)
            )
        ''')
        
        # Document metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_metadata (
                document_id TEXT PRIMARY KEY,
                image_count INTEGER DEFAULT 0,
                video_count INTEGER DEFAULT 0,
                outbound_links INTEGER DEFAULT 0,
                inbound_links INTEGER DEFAULT 0,
                load_time REAL,
                domain_authority REAL DEFAULT 0.0,
                spam_score REAL DEFAULT 0.0,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # User interactions (for learning)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                query TEXT,
                click_position INTEGER,
                dwell_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_inverted_index_term ON inverted_index(term)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id)')
        
        self.conn.commit()
    
    def setup_faiss(self):
        """Initialize FAISS vector index with multiple indices"""
        self.faiss_path = self.base_path / "vectors" / "faiss.index"
        self.mapping_path = self.base_path / "vectors" / "mapping.pkl"
        
        try:
            self.vector_index = faiss.read_index(str(self.faiss_path))
            with open(self.mapping_path, 'rb') as f:
                self.id_to_index = pickle.load(f)
        except:
            # Create multiple indices for different purposes
            self.vector_index = faiss.IndexFlatIP(self.dimension)
            self.id_to_index = {}
        
        # Additional indices for different content types
        self.title_index = faiss.IndexFlatIP(self.dimension)
        self.title_mapping = {}
    
    def setup_leveldb(self):
        """Initialize LevelDB for key-value storage"""
        self.leveldb_path = self.base_path / "leveldb"
        try:
            self.leveldb = leveldb.LevelDB(str(self.leveldb_path))
        except:
            # LevelDB might not be available, use fallback
            self.leveldb = None
    
    def setup_lmdb(self):
        """Initialize LMDB for high-performance storage"""
        self.lmdb_path = self.base_path / "lmdb"
        self.lmdb_env = lmdb.open(str(self.lmdb_path), map_size=100*1024*1024*1024)  # 100GB
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def add_document(self, document: Document):
        """Add document to all storage systems"""
        with self.lock:
            try:
                # Store in SQLite
                self._store_in_sqlite(document)
                
                # Store in vector index if vector exists
                if document.vector is not None:
                    self._store_in_faiss(document)
                
                # Store in LevelDB
                if self.leveldb:
                    self._store_in_leveldb(document)
                
                # Store in LMDB
                self._store_in_lmdb(document)
                
                # Update cache
                self._update_cache(document)
                
                self.stats['inserts'] += 1
                
                # Periodic maintenance
                if self.stats['inserts'] % 1000 == 0:
                    self._optimize_indices()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to add document {document.id}: {e}")
                return False
    
    def _store_in_sqlite(self, document: Document):
        """Store document in SQLite with compression"""
        compressed_content = zlib.compress(document.content.encode('utf-8'))
        categories_json = json.dumps(document.categories)
        entities_json = json.dumps(document.entities)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (id, url, title, content, meta_description, timestamp, 
             pagerank, quality_score, word_count, language, categories, entities)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document.id, document.url, document.title, 
            compressed_content, document.meta_description,
            document.timestamp, document.pagerank, 
            document.quality_score, document.word_count,
            document.language, categories_json, entities_json
        ))
        self.conn.commit()
    
    def _store_in_faiss(self, document: Document):
        """Store document vectors in FAISS"""
        # Main content vector
        vector = document.vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vector)
        self.vector_index.add(vector)
        self.id_to_index[len(self.id_to_index)] = document.id
        
        # Save periodically
        if len(self.id_to_index) % 1000 == 0:
            self._save_faiss_indices()
    
    def _store_in_leveldb(self, document: Document):
        """Store document in LevelDB"""
        if self.leveldb:
            key = f"doc:{document.id}".encode()
            value = pickle.dumps(document)
            self.leveldb.Put(key, value)
    
    def _store_in_lmdb(self, document: Document):
        """Store document in LMDB"""
        with self.lmdb_env.begin(write=True) as txn:
            key = f"doc:{document.id}".encode()
            value = pickle.dumps(document)
            txn.put(key, value)
    
    def _update_cache(self, document: Document):
        """Update LRU cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[document.id] = {
            'document': document,
            'timestamp': time.time()
        }
    
    def search_similar(self, query_vector: np.ndarray, k: int = 10, filters: Dict = None):
        """Advanced similarity search with filtering"""
        with self.lock:
            self.stats['queries'] += 1
            
            query_vector = query_vector.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search in vector space
            distances, indices = self.vector_index.search(query_vector, k * 2)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx in self.id_to_index:
                    doc_id = self.id_to_index[idx]
                    
                    # Apply filters
                    if filters and not self._passes_filters(doc_id, filters):
                        continue
                    
                    document = self.get_document(doc_id)
                    if document:
                        results.append(SearchResult(
                            document=document,
                            score=float(distances[0][i]),
                            explanation={'vector_similarity': float(distances[0][i])}
                        ))
                    
                    if len(results) >= k:
                        break
            
            return results
    
    def text_search(self, query: str, k: int = 10, fields: List[str] = None):
        """Advanced text search using inverted index"""
        if fields is None:
            fields = ['title', 'content']
        
        terms = self._tokenize_query(query)
        results = {}
        
        cursor = self.conn.cursor()
        
        for term in terms:
            for field in fields:
                cursor.execute('''
                    SELECT document_id, tf, idf 
                    FROM inverted_index 
                    WHERE term = ? AND field = ?
                ''', (term, field))
                
                for doc_id, tf, idf in cursor.fetchall():
                    score = tf * idf
                    if doc_id not in results:
                        results[doc_id] = 0
                    results[doc_id] += score
        
        # Get top results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        search_results = []
        for doc_id, score in sorted_results:
            document = self.get_document(doc_id)
            if document:
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    explanation={'bm25_score': score}
                ))
        
        return search_results
    
    def hybrid_search(self, query: str, query_vector: np.ndarray, k: int = 10, 
                     weights: Dict[str, float] = None):
        """Combine text and vector search"""
        if weights is None:
            weights = {'text': 0.4, 'vector': 0.6}
        
        # Get results from both methods
        text_results = self.text_search(query, k * 2)
        vector_results = self.search_similar(query_vector, k * 2)
        
        # Combine scores
        combined_scores = {}
        
        for result in text_results:
            doc_id = result.document.id
            combined_scores[doc_id] = result.score * weights['text']
        
        for result in vector_results:
            doc_id = result.document.id
            if doc_id in combined_scores:
                combined_scores[doc_id] += result.score * weights['vector']
            else:
                combined_scores[doc_id] = result.score * weights['vector']
        
        # Get top results
        sorted_doc_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        final_results = []
        for doc_id, total_score in sorted_doc_ids:
            document = self.get_document(doc_id)
            if document:
                final_results.append(SearchResult(
                    document=document,
                    score=total_score,
                    explanation={'hybrid_score': total_score}
                ))
        
        return final_results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document from cache or storage with fallback"""
        # Try cache first
        if doc_id in self.cache:
            self.stats['hits'] += 1
            return self.cache[doc_id]['document']
        
        self.stats['misses'] += 1
        
        # Try LMDB (fastest)
        try:
            with self.lmdb_env.begin() as txn:
                key = f"doc:{doc_id}".encode()
                value = txn.get(key)
                if value:
                    document = pickle.loads(value)
                    self._update_cache(document)
                    return document
        except:
            pass
        
        # Try LevelDB
        if self.leveldb:
            try:
                key = f"doc:{doc_id}".encode()
                value = self.leveldb.Get(key)
                if value:
                    document = pickle.loads(value)
                    self._update_cache(document)
                    return document
            except:
                pass
        
        # Fallback to SQLite
        return self._get_from_sqlite(doc_id)
    
    def _get_from_sqlite(self, doc_id: str) -> Optional[Document]:
        """Get document from SQLite"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Decompress content
        content = zlib.decompress(row[3]).decode('utf-8') if row[3] else ""
        categories = json.loads(row[10] or '[]')
        entities = json.loads(row[11] or '{}')
        
        document = Document(
            id=row[0],
            url=row[1],
            title=row[2] or "",
            content=content,
            meta_description=row[4] or "",
            timestamp=row[5],
            pagerank=row[6] or 1.0,
            quality_score=row[7] or 0.0,
            word_count=row[8] or 0,
            language=row[9] or "en",
            categories=categories,
            entities=entities
        )
        
        self._update_cache(document)
        return document
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Simple query tokenization"""
        # Implement more advanced tokenization here
        return [term.lower().strip() for term in query.split() if len(term) > 2]
    
    def _passes_filters(self, doc_id: str, filters: Dict) -> bool:
        """Check if document passes all filters"""
        cursor = self.conn.cursor()
        
        if 'min_quality' in filters:
            cursor.execute('SELECT quality_score FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            if not row or row[0] < filters['min_quality']:
                return False
        
        if 'language' in filters:
            cursor.execute('SELECT language FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            if not row or row[0] != filters['language']:
                return False
        
        # Add more filters as needed
        
        return True
    
    def _optimize_indices(self):
        """Perform maintenance on indices"""
        self.logger.info("Optimizing indices...")
        
        # Vacuum SQLite database
        self.conn.execute("VACUUM")
        
        # Save FAISS indices
        self._save_faiss_indices()
        
        self.logger.info("Indices optimized")
    
    def _save_faiss_indices(self):
        """Save FAISS indices to disk"""
        faiss.write_index(self.vector_index, str(self.faiss_path))
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.id_to_index, f)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM inverted_index")
        total_terms = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM links")
        total_links = cursor.fetchone()[0]
        
        return {
            'total_documents': total_docs,
            'total_terms': total_terms,
            'total_links': total_links,
            'vector_index_size': len(self.id_to_index),
            'cache_size': len(self.cache),
            'cache_hit_ratio': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1),
            **self.stats
        }
    
    def close(self):
        """Clean shutdown"""
        self._save_faiss_indices()
        self.conn.close()
        self.lmdb_env.close()
        self.thread_pool.shutdown()
        
        if self.leveldb:
            del self.leveldb  # LevelDB cleanup
