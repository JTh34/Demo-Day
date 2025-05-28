# document_loader.py
import os
import re
import logging
from typing import List, Optional
from uuid import uuid4

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Text, ElementMetadata

from langchain_core.documents import Document

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_document_with_unstructured(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Document]:
    """ Loads a PDF document with Unstructured.io and converts it to LangChain format """
    logger.info(f"Loading PDF with Unstructured.io: {pdf_path}, pages {start_page} to {end_page}")

    page_range = None
    if start_page is not None and end_page is not None:
        page_range = list(range(start_page, end_page + 1))

    try:
        extracted_elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            include_page_breaks=False,
        )
    except Exception as e:
        logger.error(f"Error while partitioning PDF {pdf_path}: {e}")
        return []

    logger.info(f"Extraction completed: {len(extracted_elements)} raw elements extracted")

    documents = []
    for element in extracted_elements:
        # Ignore empty or non-textual elements
        if not hasattr(element, 'text') or not element.text.strip():
            continue

        current_page_number = getattr(element.metadata, 'page_number', None)

        # Page filtering
        if start_page is not None and current_page_number is not None and current_page_number < start_page:
            continue
        if end_page is not None and current_page_number is not None and current_page_number > end_page:
            continue
        
        # Build metadata for LangChain Document
        metadata = {
            "source": pdf_path,
            "page": current_page_number,
            "category": str(type(element).__name__),
            "id": str(element.id) if hasattr(element, "id") else str(uuid4()),
        }
        if hasattr(element.metadata, 'filename'):
            metadata["filename"] = element.metadata.filename
        if hasattr(element.metadata, 'filetype'):
            metadata["filetype"] = element.metadata.filetype
        if hasattr(element.metadata, 'parent_id') and element.metadata.parent_id is not None:
            metadata["parent_id"] = str(element.metadata.parent_id)

        documents.append(Document(
            page_content=element.text,
            metadata=metadata
        ))

    logger.info(f"Conversion completed: {len(documents)} LangChain documents created after filtering")
    return documents


def split_document_with_unstructured(documents: List[Document]) -> List[Document]:
    """ Splits documents into chunks using Unstructured.io features """
    if not documents:
        logger.warning("No documents to split.")
        return []
    
    logger.info(f"Splitting {len(documents)} documents into chunks with Unstructured.io")

    chunked_langchain_documents = []

    # Recreate Unstructured elements from LangChain documents
    unstructured_elements_for_chunking = []

    valid_categories = ["NarrativeText", "ListItem", "Title"]

    for doc in documents:
        if doc.metadata.get("category") not in valid_categories:
            continue

        if len(doc.page_content.strip()) < 50:
            continue
            
        # Clean the text
        cleaned_text = doc.page_content
        # Replace references like "FIGURE XX-X:" with an empty string
        cleaned_text = re.sub(r'FIGURE \d+-\d+:', '', cleaned_text)
        # Replace \xad (soft hyphen) with an empty string to fix broken text
        cleaned_text = cleaned_text.replace('\xad', ' ')
        # Reassign the cleaned text to the document
        doc.page_content = cleaned_text
        
        # Create an ElementMetadata object from the metadata dictionary
        element_meta = ElementMetadata()
        if doc.metadata.get("filename"):
            element_meta.filename = doc.metadata.get("filename")
        if doc.metadata.get("filetype"):
            element_meta.filetype = doc.metadata.get("filetype")
        if doc.metadata.get("page"):
            element_meta.page_number = doc.metadata.get("page")
        
        # Create the Text element with appropriate metadata
        element = Text(
            text=doc.page_content, 
            metadata=element_meta, 
            element_id=doc.metadata.get("id", str(uuid4()))
        )
        unstructured_elements_for_chunking.append(element)

    if not unstructured_elements_for_chunking:
        logger.warning("No Unstructured elements could be created from LangChain documents.")
        return []

    try:
        # Apply standard chunking
        chunks = chunk_elements(
            elements=unstructured_elements_for_chunking,
            max_characters=1800,
            new_after_n_chars=1500,  # To avoid chunks that are too long
            overlap=400,  # 400 characters of overlap
        )
    except Exception as e:
        logger.error(f"Error while chunking elements: {str(e)}. Returning unsplit documents.")
        return documents

    # Convert chunks to LangChain Documents
    for i, chunk_element in enumerate(chunks):
        # Chunks are Element objects (or CompositeElement)
        page = getattr(chunk_element.metadata, 'page_number', 0)
        
        metadata = {
            "source": documents[0].metadata.get("source", ""),
            "page": page,
            "chunk_index": i,
            "id": str(chunk_element.id) if hasattr(chunk_element, "id") else str(uuid4()),
            "word_count": len(chunk_element.text.split()) if hasattr(chunk_element, 'text') else 0,
        }
        # Estimate the number of tokens
        metadata["token_count_approx"] = int(metadata["word_count"] * 1.3)
        
        chunked_langchain_documents.append(Document(
            page_content=chunk_element.text if hasattr(chunk_element, 'text') else "",
            metadata=metadata
        ))

    logger.info(f"Splitting completed: {len(chunked_langchain_documents)} chunks created")
    return chunked_langchain_documents
