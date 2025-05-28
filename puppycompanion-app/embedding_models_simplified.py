# embedding_models_simplified.py
import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModelBase(ABC):
    """Base class for embedding models"""
    
    def __init__(self, name: str, description: str = ""):
       
        self.name = name
        self.description = description
        self.client = None
        self.vector_store = None
        self.retriever = None
    
    @abstractmethod
    def create_vector_store(self, documents: List[Document], collection_name: str, k: int = 5) -> None:
        """  Create the vector store for documents """
        pass
    
    def get_retriever(self):
        """Returns the retriever"""
        if self.retriever is None:
            raise ValueError("The vector store has not been initialized")
        return self.retriever


class OpenAIEmbeddingModel(EmbeddingModelBase):
    """Standard OpenAI embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(
            name="OpenAI Embeddings", 
            description=f"OpenAI embedding model {model_name}"
        )
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.vector_dim = 1536  # Dimension of OpenAI vectors
    
    def create_vector_store(self, documents: List[Document], collection_name: str, k: int = 5) -> None:
        """ Create the vector store for documents """
        # S'assurer que le répertoire existe
        if not os.path.exists("/tmp/qdrant_storage"):
            os.makedirs("/tmp/qdrant_storage", exist_ok=True)
        
        # Initialize Qdrant client with local storage
        logger.info(f"Creating Qdrant client with local storage at /tmp/qdrant_storage")
        self.client = qdrant_client.QdrantClient(path="/tmp/qdrant_storage")
        
        # Vérifier les collections existantes
        try:
            collections = self.client.get_collections().collections
            logger.info(f"Existing collections: {[c.name for c in collections]}")
            
            # Supprimer la collection si elle existe déjà
            if collection_name in [c.name for c in collections]:
                logger.info(f"Collection '{collection_name}' already exists, recreating it")
                self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            logger.warning(f"Could not handle existing collections: {str(e)}")
        
        # Create a collection
        logger.info(f"Creating Qdrant collection '{collection_name}'")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created successfully")
        
        # Create the vector store
        logger.info(f"Initializing Qdrant vector store for '{collection_name}'")
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings,
        )
        logger.info("Qdrant vector store initialized")
        
        # Add documents
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vector_store.add_documents(documents=documents)
        logger.info(f"[{self.name}] Vector store created with {len(documents)} documents")
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        logger.info(f"Retriever created with k={k}")