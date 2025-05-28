# embedding_models.py
import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
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
        # Initialize Qdrant client in memory
        self.client = qdrant_client.QdrantClient(":memory:")
        
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
        
        # Add documents
        self.vector_store.add_documents(documents=documents)
        logger.info(f"[{self.name}] Vector store created with {len(documents)} documents")
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})


class OpenAIWithCohereRerankModel:
    """Embedding model using OpenAI with Cohere Rerank"""
    
    def __init__(self):
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.retriever = None
        self.reranker = None
        
        # Initialisation du reranker Cohere
        if "COHERE_API_KEY" in os.environ:
            try:
                from langchain_cohere import CohereRerank
                self.reranker = CohereRerank(model="rerank-english-v2.0", top_n=5)
            except Exception as e:
                logger.error(f"Error initializing Cohere Rerank: {str(e)}")
    
    def create_vector_store(self, documents, store_name="vector_store", k=20):
        """Creates a vector store from documents"""
        # Créer le vectorstore avec OpenAI
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=store_name
        )
        
        # Créer le retriever initial
        base_retriever = db.as_retriever(search_kwargs={"k": k})
        
        # Si le reranker est disponible, l'utiliser
        if self.reranker:
            try:
                # Adapter le reranker à l'API attendue
                class RerankerWrapper:
                    def __init__(self, reranker):
                        self.reranker = reranker
                        
                    def invoke(self, inputs):
                        # Appelle rerank() au lieu de invoke()
                        return self.reranker.rerank(inputs)
                
                # Créer un wrapper pour le reranker qui fournit invoke()
                reranker_wrapper = RerankerWrapper(self.reranker)
                
                from langchain.retrievers import ContextualCompressionRetriever
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=reranker_wrapper,
                    base_retriever=base_retriever
                )
                logger.info(f"[OpenAI + Cohere Rerank] Vector store created with {len(documents)} documents")
            except Exception as e:
                logger.error(f"Error setting up Cohere reranker: {str(e)}")
                # Fallback to standard retriever
                self.retriever = base_retriever
                logger.warning("Falling back to standard retriever without reranking")
        else:
            self.retriever = base_retriever
            logger.warning("Cohere Rerank not available, using standard retriever")
            
    def get_retriever(self):
        """Returns the retriever"""
        return self.retriever


class SnowflakeArcticEmbedModel(EmbeddingModelBase):
    """ Snowflake Arctic Embed model without fine-tuning """
    
    def __init__(self):
        """Initialize the Snowflake Arctic Embed model"""
        super().__init__(
            name="Snowflake Arctic Embed",
            description="Snowflake Arctic Embed model without fine-tuning"
        )
        
        # Create embeddings with Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-l",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_dim = 1024  # Standard dimension of Snowflake Arctic Embed
    
    def create_vector_store(self, documents: List[Document], collection_name: str, k: int = 5) -> None:
        """ Create the vector store for documents """
        # Initialize Qdrant client in memory
        self.client = qdrant_client.QdrantClient(":memory:")
        
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
        
        # Add documents
        self.vector_store.add_documents(documents=documents)
        logger.info(f"[{self.name}] Vector store created with {len(documents)} documents")
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})


class SnowflakeArcticEmbedFineTunedModel(SnowflakeArcticEmbedModel):
    """Snowflake Arctic Embed model with fine-tuning"""
    
    def __init__(self, model_path: str = "Snowflake/snowflake-arctic-embed-l_finetuned"):
        """ Initialize the Snowflake Arctic Embed model with fine-tuning """
        super().__init__()
        self.name = "Snowflake Arctic Embed Fine-Tuned"
        self.description = "Snowflake Arctic Embed model with fine-tuning"
        
        # Replace embeddings with those from the fine-tuned model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


def create_embedding_model(model_type: str) -> EmbeddingModelBase:
    """ Create an embedding model according to the specified type """
    if model_type == 'openai':
        return OpenAIEmbeddingModel()
    elif model_type == 'openai_cohere':
        return OpenAIWithCohereRerankModel()
    elif model_type == 'snowflake':
        return SnowflakeArcticEmbedModel()
    elif model_type == 'snowflake_finetuned':
        return SnowflakeArcticEmbedFineTunedModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
