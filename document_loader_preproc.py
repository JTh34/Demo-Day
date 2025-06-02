# document_loader_preproc.py
import logging
import json
import pickle
from typing import List
from langchain_core.documents import Document

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_preprocessed_chunks(file_path: str) -> List[Document]:
    """Loads preprocessed chunks from a JSON or pickle file"""
    logger.info(f"Loading chunks from {file_path}")
    
    if file_path.endswith('.json'):
        # Loading from JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            serialized_chunks = json.load(f)
    elif file_path.endswith('.pkl'):
        # Loading from pickle
        with open(file_path, 'rb') as f:
            serialized_chunks = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Converting dictionaries to Document objects
    chunks = []
    for item in serialized_chunks:
        chunks.append(Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        ))
    
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks