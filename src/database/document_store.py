import json
import zlib
import msgpack
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import leveldb
import lmdb

@dataclass
class StoredDocument:
    id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    compression: str = "zlib"

class DocumentStore:
    def __init__(self, storage_path: str = "./data/documents"):
        self.storage_path = storage_path
        self.setup_storage()
    
    def setup_storage(self):
        # LMDB for fast document access
        self.lmdb_env = lmdb.open(f"{self.storage_path}/lmdb", map_size=10*1024*1024*1024)
        
        # LevelDB for additional storage
        try:
            self.leveldb = leveldb.LevelDB(f"{self.storage_path}/leveldb")
        except:
            self.leveldb = None
    
    def store_document(self, doc_id: str, data: Dict, metadata: Dict = None):
        """Store document in multiple storage backends"""
        if metadata is None:
            metadata = {}
        
        stored_doc = StoredDocument(
            id=doc_id,
            data=data,
            metadata=metadata
        )
        
        # Store in LMDB
        with self.lmdb_env.begin(write=True) as txn:
            key = f"doc:{doc_id}".encode()
            value = zlib.compress(msgpack.packb(stored_doc.__dict__))
            txn.put(key, value)
        
        # Store in LevelDB
        if self.leveldb:
            key = f"doc:{doc_id}".encode()
            value = zlib.compress(msgpack.packb(stored_doc.__dict__))
            self.leveldb.Put(key, value)
    
    def get_document(self, doc_id: str) -> Optional[StoredDocument]:
        """Retrieve document from storage"""
        # Try LMDB first
        with self.lmdb_env.begin() as txn:
            key = f"doc:{doc_id}".encode()
            value = txn.get(key)
            if value:
                decompressed = zlib.decompress(value)
                data = msgpack.unpackb(decompressed)
                return StoredDocument(**data)
        
        # Try LevelDB
        if self.leveldb:
            try:
                key = f"doc:{doc_id}".encode()
                value = self.leveldb.Get(key)
                if value:
                    decompressed = zlib.decompress(value)
                    data = msgpack.unpackb(decompressed)
                    return StoredDocument(**data)
            except:
                pass
        
        return None
    
    def batch_store(self, documents: List[tuple]):
        """Store multiple documents efficiently"""
        with self.lmdb_env.begin(write=True) as txn:
            for doc_id, data, metadata in documents:
                stored_doc = StoredDocument(
                    id=doc_id,
                    data=data,
                    metadata=metadata or {}
                )
                
                key = f"doc:{doc_id}".encode()
                value = zlib.compress(msgpack.packb(stored_doc.__dict__))
                txn.put(key, value)
