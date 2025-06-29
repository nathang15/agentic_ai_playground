from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .models import Document, SearchResult, SourceType
from .document_processor import create_chunks
from .config import AppSettings
from .logging_config import get_logger

logger = get_logger(__name__)

class VectorStore:    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        db_path = Path(settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.encoder = SentenceTransformer(settings.embedding_model)
        
        try:
            self.collection = self.client.get_collection("documents")
        except:
            self.collection = self.client.create_collection("documents")
        
        logger.info("[OK] Vector store initialized")
    
    async def add_document(self, document: Document) -> None:
        try:
            chunks = create_chunks(
                document.content, 
                self.settings.chunk_size, 
                self.settings.chunk_overlap
            )
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document.id}_chunk_{i}"
                
                try:
                    existing = self.collection.get(ids=[chunk_id])
                    if existing['ids']:
                        continue
                except:
                    pass
                
                embedding = self.encoder.encode([chunk.content])[0]
                
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[chunk.content],
                    metadatas=[{
                        "doc_id": document.id,
                        "document_title": document.title,
                        "filename": document.file_name,
                    }],
                    ids=[chunk_id]
                )
            
            logger.info(f"[DB] Added {document.title} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Vector store error: {e}")
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            query_embedding = self.encoder.encode([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = max(0, 1 - distance)
                    
                    search_results.append(SearchResult(
                        content=doc,
                        source=metadata.get('filename', 'Unknown'),
                        score=similarity_score,
                        metadata=metadata,
                        source_type=SourceType.DOCUMENT,
                        rank=i + 1
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
