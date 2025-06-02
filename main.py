# main.py - FastAPI version of PuppyCompanion
import os
import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your existing modules
from rag_system import RAGSystem
from agent_workflow import AgentWorkflow

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
global_agent = None
global_qdrant_client = None
global_retriever = None
global_documents = None
initialization_completed = False

# Path to preprocessed data
PREPROCESSED_CHUNKS_PATH = "all_books_preprocessed_chunks.json"

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    tool_used: str = ""

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_log(self, message: str, log_type: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "message": message,
            "type": log_type
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(log_data))
            except:
                pass

manager = ConnectionManager()

def load_preprocessed_chunks(file_path="all_books_preprocessed_chunks.json"):
    """Load preprocessed chunks from a JSON file."""
    global global_documents
    
    if global_documents is not None:
        logger.info("Using cached document chunks")
        return global_documents
        
    logger.info(f"Loading preprocessed chunks from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        from langchain_core.documents import Document
        documents = []
        for item in data:
            doc = Document(
                page_content=item['page_content'],
                metadata=item['metadata']
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} document chunks")
        global_documents = documents
        return documents
    except Exception as e:
        logger.error(f"Error loading preprocessed chunks: {str(e)}")
        raise

def initialize_retriever(documents):
    """Create a retriever from documents using a shared Qdrant client."""
    global global_qdrant_client, global_retriever
    
    # Return existing retriever if already initialized
    if global_retriever is not None:
        logger.info("Using existing global retriever")
        return global_retriever
        
    logger.info("Creating retriever from documents")
    try:
        # Use langchain_qdrant to create a vector store
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        from langchain_openai import OpenAIEmbeddings
        
        # Create embedding object
        embeddings = OpenAIEmbeddings()
        logger.info("Created OpenAI embeddings object")
        
        # Create a persistent path for embeddings storage
        qdrant_path = "/tmp/qdrant_storage"
        logger.info(f"Using persistent Qdrant storage path: {qdrant_path}")
        
        # Create directory for Qdrant storage
        os.makedirs(qdrant_path, exist_ok=True)
        
        # Create or reuse global Qdrant client
        if global_qdrant_client is None:
            client = QdrantClient(path=qdrant_path)
            global_qdrant_client = client
            logger.info("Created new global Qdrant client with persistent storage")
        else:
            client = global_qdrant_client
            logger.info("Using existing global Qdrant client")
        
        # Check if collection already exists
        try:
            collections = client.get_collections()
            collection_exists = any(collection.name == "puppies" for collection in collections.collections)
            logger.info(f"Collection 'puppies' exists: {collection_exists}")
        except Exception as e:
            collection_exists = False
            logger.info(f"Could not check collections, assuming none exist: {e}")
        
        # OpenAI embeddings dimension
        embedding_dim = 1536
        
        # Create collection only if it doesn't exist
        if not collection_exists:
            from qdrant_client.http import models
            client.create_collection(
                collection_name="puppies",
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
            logger.info("Created new collection 'puppies'")
        else:
            logger.info("Using existing collection 'puppies'")
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="puppies",
            embedding=embeddings
        )
        
        # Add documents only if collection was just created (to avoid duplicates)
        if not collection_exists:
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        else:
            logger.info("Using existing embeddings in vector store")
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        logger.info("Created retriever")
        
        # Store global retriever
        global_retriever = retriever
        
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise

async def initialize_system():
    """Initialize the RAG system and agent"""
    global global_agent, initialization_completed
    
    if initialization_completed:
        return global_agent
    
    await manager.send_log("Starting system initialization...", "info")
    
    try:
        # Load documents
        await manager.send_log("Loading document chunks...", "info")
        documents = load_preprocessed_chunks()
        await manager.send_log(f"Loaded {len(documents)} document chunks", "success")
        
        # Create retriever
        await manager.send_log("Creating retriever...", "info")
        retriever = initialize_retriever(documents)
        await manager.send_log("Retriever ready", "success")
        
        # Create RAG system
        await manager.send_log("Setting up RAG system...", "info")
        rag_system = RAGSystem(retriever)
        rag_tool = rag_system.create_rag_tool()
        await manager.send_log("RAG system ready", "success")
        
        # Create agent workflow
        await manager.send_log("Initializing agent workflow...", "info")
        agent = AgentWorkflow(rag_tool)
        await manager.send_log("Agent workflow ready", "success")
        
        global_agent = agent
        initialization_completed = True
        
        await manager.send_log("System initialization completed!", "success")
        return agent
        
    except Exception as e:
        await manager.send_log(f"Error during initialization: {str(e)}", "error")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    try:
        await initialize_system()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise  # ⚠️ IMPORTANT: Arrêter l'application si l'initialisation échoue
    
    yield
    
    # Shutdown - cleanup if needed
    logger.info("Application shutdown")

# FastAPI app with lifespan
app = FastAPI(
    title="PuppyCompanion", 
    description="AI Assistant for Puppy Care",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def get_favicon():
    """Return a 204 No Content for favicon to avoid 404 errors"""
    from fastapi import Response
    return Response(status_code=204)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time logs"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QuestionRequest):
    """Main chat endpoint"""
    global global_agent
    
    if not initialization_completed or not global_agent:
        await manager.send_log("System not initialized, starting initialization...", "warning")
        try:
            global_agent = await initialize_system()
        except Exception as e:
            raise HTTPException(status_code=500, detail="System initialization failed")
    
    question = request.question
    await manager.send_log(f"New question: {question}", "info")
    
    try:
        # Process question with agent
        await manager.send_log("Processing with agent workflow...", "info")
        result = global_agent.process_question(question)
        
        # Extract response and metadata
        response_content = global_agent.get_final_response(result)
        
        # Parse tool usage and send detailed info to debug console
        tool_used = "Unknown"
        sources = []
        
        if "[Using RAG tool]" in response_content:
            tool_used = "RAG Tool"
            await manager.send_log("Used RAG tool - Knowledge base search", "tool")
            
            # Send detailed RAG chunks to debug console
            if "context" in result:
                await manager.send_log(f"Retrieved {len(result['context'])} chunks from knowledge base:", "info")
                for i, doc in enumerate(result["context"], 1):
                    source_name = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    chapter = doc.metadata.get('chapter', '')
                    
                    # Create detailed chunk info for console
                    if chapter:
                        chunk_header = f"Chunk {i} - {source_name} (Chapter: {chapter}, Page: {page})"
                    else:
                        chunk_header = f"Chunk {i} - {source_name} (Page: {page})"
                    
                    await manager.send_log(chunk_header, "source")
                    
                    # Send chunk content preview
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    await manager.send_log(f"Content: {content_preview}", "chunk")
                    
                    # Collect for sources array (minimal info)
                    source_info = {
                        "chunk": i,
                        "source": source_name,
                        "page": page,
                        "chapter": chapter
                    }
                    sources.append(source_info)
                    
        elif "[Using Tavily tool]" in response_content:
            tool_used = "Tavily Tool"
            await manager.send_log("Used Tavily tool - Web search", "tool")
            
            # Extract Tavily sources from response content and send to debug console
            lines = response_content.split('\n')
            tavily_sources_count = 0
            
            for line in lines:
                line_stripped = line.strip()
                # Look for Tavily source lines like "- *Source 1 - domain.com: Title*"
                if (line_stripped.startswith('- *Source') and ':' in line_stripped):
                    tavily_sources_count += 1
                    # Extract and format for debug console
                    try:
                        # Remove markdown formatting for clean display
                        clean_source = line_stripped.replace('- *', '').replace('*', '')
                        await manager.send_log(f"{clean_source}", "source")
                    except:
                        await manager.send_log(f"{line_stripped}", "source")
                        
            if tavily_sources_count > 0:
                await manager.send_log(f"Found {tavily_sources_count} web sources", "info")
            else:
                await manager.send_log("Searched the web for current information", "source")
            
        elif "out of scope" in response_content.lower():
            tool_used = "Out of Scope"
            await manager.send_log("Question outside scope (not dog-related)", "warning")
        
        # Clean response content - REMOVE ALL source references for mobile interface
        clean_response = response_content
        
        # Remove tool markers
        clean_response = clean_response.replace("[Using RAG tool]", "").replace("[Using Tavily tool]", "").strip()
        
        # Remove ALL source-related lines with comprehensive patterns
        lines = clean_response.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that are source references (comprehensive patterns)
            skip_line = False
            
            # Pattern 1: Lines starting with * containing Source/Chunk/Based on
            if (line_stripped.startswith('*') and 
                ('Chunk' in line or 'Source' in line or 'Based on' in line or 'Basé sur' in line)):
                skip_line = True
            
            # Pattern 2: Lines starting with - * containing Source/Chunk/Based on  
            if (line_stripped.startswith('- *') and 
                ('Chunk' in line or 'Source' in line or 'Based on' in line or 'Basé sur' in line)):
                skip_line = True
                
            # Pattern 3: Lines that are just chunk references like "- *Chunk 1 - filename*"
            if (line_stripped.startswith('- *Chunk') and line_stripped.endswith('*')):
                skip_line = True
                
            # Pattern 4: Lines that start with "- *Based on" 
            if line_stripped.startswith('- *Based on'):
                skip_line = True
                
            # Add line only if it's not a source reference and not empty
            if not skip_line and line_stripped:
                cleaned_lines.append(line)
        
        # Final clean response for mobile interface
        final_response = '\n'.join(cleaned_lines).strip()
        
        # Additional cleanup - remove any remaining source markers at start
        while final_response.startswith('- *') or final_response.startswith('*'):
            # Find the end of the line to remove
            if '\n' in final_response:
                final_response = final_response.split('\n', 1)[1].strip()
            else:
                final_response = ""
                break
        
        # Ensure we have a response
        if not final_response:
            final_response = "I apologize, but I couldn't generate a proper response to your question."
        
        await manager.send_log(f"Clean response ready for mobile interface", "success")
        
        return ChatResponse(
            response=final_response,
            sources=sources,  # Minimal info for API, detailed info already sent to debug console
            tool_used=tool_used
        )
        
    except Exception as e:
        await manager.send_log(f"Error processing question: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": initialization_completed,
        "timestamp": datetime.now().isoformat()
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)