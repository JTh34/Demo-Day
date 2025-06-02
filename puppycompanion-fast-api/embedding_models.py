# embedding_models.py
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheManager:
    """Cache manager with limits for Hugging Face Spaces"""
    
    def __init__(self, cache_directory: str = "./cache", max_size_mb: int = 500, max_age_days: int = 7):
        self.cache_directory = Path(cache_directory)
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert to bytes
        self.max_age_seconds = max_age_days * 24 * 60 * 60  # Convert to seconds
        
    def get_cache_size(self) -> int:
        """Compute the total cache size in bytes"""
        total_size = 0
        if self.cache_directory.exists():
            for file_path in self.cache_directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    
    def get_cache_size_mb(self) -> float:
        """Return the cache size in MB"""
        return self.get_cache_size() / (1024 * 1024)
    
    def clean_old_files(self):
        """Delete cache files that are too old"""
        if not self.cache_directory.exists():
            return
            
        current_time = time.time()
        deleted_count = 0
        
        for file_path in self.cache_directory.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > self.max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Unable to delete {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"🧹 Cache cleaned: {deleted_count} old files deleted")
    
    def clear_cache_if_too_large(self):
        """Completely clear the cache if it exceeds the size limit"""
        current_size_mb = self.get_cache_size_mb()
        
        if current_size_mb > (self.max_size_bytes / (1024 * 1024)):
            logger.warning(f"Cache too large ({current_size_mb:.1f}MB > {self.max_size_bytes/(1024*1024)}MB)")
            try:
                if self.cache_directory.exists():
                    shutil.rmtree(self.cache_directory)
                    self.cache_directory.mkdir(parents=True, exist_ok=True)
                    logger.info("Cache fully cleared to save disk space")
            except Exception as e:
                logger.error(f"Error while clearing cache: {e}")
    
    def cleanup_cache(self):
        """Smart cache cleanup"""
        # 1. Clean old files
        self.clean_old_files()
        
        # 2. Check size after cleaning
        current_size_mb = self.get_cache_size_mb()
        
        # 3. If still too large, clear completely
        if current_size_mb > (self.max_size_bytes / (1024 * 1024)):
            self.clear_cache_if_too_large()
        else:
            logger.info(f"Cache size: {current_size_mb:.1f}MB (OK)")


class OpenAIEmbeddingModel:
    """OpenAI embedding model with smart caching for Hugging Face Spaces"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", persist_directory: str = "./vector_stores", 
                 max_cache_size_mb: int = 500, max_cache_age_days: int = 7):
        self.name = "OpenAI Embeddings (Smart Cache)"
        self.description = f"OpenAI embedding model {model_name} with smart caching for HF Spaces"
        self.model_name = model_name
        self.vector_dim = 1536  # Dimension of OpenAI vectors
        
        # Setup directories
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.cache_directory = Path("./cache")
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager with limits for HF Spaces
        self.cache_manager = CacheManager(
            cache_directory=str(self.cache_directory),
            max_size_mb=max_cache_size_mb,
            max_age_days=max_cache_age_days
        )
        
        # Initialize components
        self.client = None
        self.vector_store = None
        self.retriever = None
        self.embeddings = None
        
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup OpenAI embeddings with smart caching"""
        # Clean cache before starting
        logger.info("🔍 Checking cache state...")
        self.cache_manager.cleanup_cache()
        
        # Create base OpenAI embeddings
        base_embeddings = OpenAIEmbeddings(model=self.model_name)
        
        # Create cached version
        namespace_key = f"openai_{self.model_name}"
        safe_namespace = hashlib.md5(namespace_key.encode()).hexdigest()
        
        # Setup local file store for caching
        store = LocalFileStore(str(self.cache_directory))
        
        # Create cached embeddings
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            base_embeddings, 
            store, 
            namespace=safe_namespace,
            batch_size=32
        )
        
        cache_size = self.cache_manager.get_cache_size_mb()
        logger.info(f"[{self.name}] Embeddings configured with smart cache (Size: {cache_size:.1f}MB)")
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection already exists"""
        try:
            collections = self.client.get_collections()
            return any(collection.name == collection_name for collection in collections.collections)
        except Exception as e:
            logger.warning(f"Error while checking collection {collection_name}: {e}")
            return False
    
    def create_vector_store(self, documents: List[Document], collection_name: str, k: int = 5) -> None:
        """Create the vector store for documents"""
        # Path for persistent Qdrant storage - model-specific subdirectory
        qdrant_path = self.persist_directory / "qdrant_db" / "openai_cached"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client with persistent storage
        self.client = qdrant_client.QdrantClient(path=str(qdrant_path))
        
        # Check if the collection already exists
        if self._collection_exists(collection_name):
            logger.info(f"[{self.name}] Collection '{collection_name}' already exists, loading...")
            # Load the existing vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
        else:
            logger.info(f"[{self.name}] Creating new collection '{collection_name}'...")
            # Create a collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            
            # Create the vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
            
            # Add documents (caching will happen automatically)
            logger.info(f"[{self.name}] Adding {len(documents)} documents (with embedding cache)...")
            self.vector_store.add_documents(documents=documents)
            logger.info(f"[{self.name}] Vector store created successfully")
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Check cache size after adding documents
        cache_size = self.cache_manager.get_cache_size_mb()
        if cache_size > 100:  # Alert if > 100MB
            logger.warning(f"Large cache: {cache_size:.1f}MB - consider cleaning soon")
    
    def get_retriever(self):
        """Returns the retriever"""
        if self.retriever is None:
            raise ValueError("The vector store has not been initialized")
        return self.retriever
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Return information about the cache state"""
        return {
            "cache_size_mb": self.cache_manager.get_cache_size_mb(),
            "max_size_mb": self.cache_manager.max_size_bytes / (1024 * 1024),
            "max_age_days": self.cache_manager.max_age_seconds / (24 * 60 * 60),
            "cache_directory": str(self.cache_directory)
        }
    
    def manual_cache_cleanup(self):
        """Manual cache cleanup"""
        logger.info("🧹 Manual cache cleanup requested...")
        self.cache_manager.cleanup_cache()


def create_embedding_model(persist_directory: str = "./vector_stores", 
                          max_cache_size_mb: int = 500, 
                          max_cache_age_days: int = 7) -> OpenAIEmbeddingModel:
    
    logger.info(f"Creating optimized OpenAI model (Max cache: {max_cache_size_mb}MB, Max age: {max_cache_age_days}d)")
    return OpenAIEmbeddingModel(
        persist_directory=persist_directory,
        max_cache_size_mb=max_cache_size_mb,
        max_cache_age_days=max_cache_age_days
    )