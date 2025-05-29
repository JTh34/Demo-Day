# app.py - Refactored version to use the advanced agent
import os
import logging
import time
import uuid
import asyncio
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wait 2 seconds to ensure all modules are properly loaded
logger.info("Waiting 2 seconds for module initialization...")
time.sleep(2)
logger.info("Module initialization completed")

# Fix Chainlit permission issues
os.environ["CHAINLIT_AUTH_SECRET"] = "puppycompanion-secret-key"
os.environ["CHAINLIT_CONFIG_DIR"] = "/tmp/chainlit_config"
os.environ["CHAINLIT_MAX_SIZE_MB"] = "100"

# Create temporary directory
if not os.path.exists("/tmp/chainlit_config"):
    os.makedirs("/tmp/chainlit_config", exist_ok=True)

# Load environment variables
load_dotenv()

# Check OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Check Tavily API key for web search
if "TAVILY_API_KEY" not in os.environ:
    logger.warning("TAVILY_API_KEY not found in environment variables, web search will be disabled")

# Import Chainlit AFTER environment configuration
import chainlit as cl
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Import advanced system modules - use local files instead of ai_research_assistant package
from rag_system import RAGSystem
from agent_workflow import AgentWorkflow

# Path to preprocessed data
PREPROCESSED_CHUNKS_PATH = "data/preprocessed_chunks.json"
logger.info(f"Checking for chunks file at {PREPROCESSED_CHUNKS_PATH}")
if not os.path.exists(PREPROCESSED_CHUNKS_PATH):
    logger.error(f"Preprocessed chunks file not found at {PREPROCESSED_CHUNKS_PATH}")

# Global variables - Shared Qdrant client and retriever
global_qdrant_client = None
global_retriever = None
global_documents = None
global_agent = None
initialization_completed = False
initialization_lock = asyncio.Lock()

# Session management
active_sessions = set()
max_sessions = 1
session_lock = asyncio.Lock()

def load_preprocessed_chunks(file_path):
    """Load preprocessed chunks from a JSON file."""
    global global_documents
    
    if global_documents is not None:
        logger.info("Using cached document chunks")
        return global_documents
        
    logger.info(f"Loading preprocessed chunks from {file_path}")
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert to Langchain Document objects
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
        
        # Create embedding object
        embeddings = OpenAIEmbeddings()
        logger.info("Created OpenAI embeddings object")
        
        # Create a persistent path for embeddings storage
        qdrant_path = "./data/qdrant_storage"
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

def initialize_agent_workflow(retriever):
    """Initialize the advanced agent workflow."""
    global global_agent
    
    # Return existing agent if already initialized
    if global_agent is not None:
        logger.info("Using existing global agent")
        return global_agent
        
    logger.info("Initializing advanced agent workflow")
    try:
        # Create RAG system
        rag_system = RAGSystem(retriever)
        rag_tool = rag_system.create_rag_tool()
        
        # Create agent workflow that uses the RAG tool
        agent = AgentWorkflow(rag_tool)
        logger.info("Advanced agent workflow initialized successfully")
        
        # Store global agent
        global_agent = agent
        
        return agent
    except Exception as e:
        logger.error(f"Error initializing agent workflow: {str(e)}")
        raise

@cl.on_chat_start
async def start():
    """Initialize the application at session start."""
    global initialization_completed, active_sessions
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Session limitation
    async with session_lock:
        if len(active_sessions) >= max_sessions:
            logger.warning(f"Session {session_id}: Maximum sessions reached ({max_sessions}), rejecting new session")
            await cl.Message(
                content="⚠️ Maximum number of active sessions reached. Please refresh the page and try again."
            ).send()
            return
        
        active_sessions.add(session_id)
        logger.info(f"Session {session_id}: Accepted ({len(active_sessions)}/{max_sessions} active sessions)")
    
    cl.user_session.set("session_id", session_id)
    
    # Display welcome message
    await cl.Message(
        content=f"""### Welcome to PuppyCompanion 🐶

Session ID: `{session_id}`
        """
    ).send()
    
        
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        await cl.Message(
            content="⚠️ No OpenAI API key found. Please provide your API key to continue."
        ).send()
        
        api_key = await cl.AskUserMessage(
            content="Please enter your OpenAI API key:",
            timeout=300
        ).send()
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info(f"Session {session_id}: OpenAI API key provided by user")
        else:
            await cl.Message(content="No API key provided. Unable to continue.").send()
            # Remove session from active list
            async with session_lock:
                active_sessions.discard(session_id)
            return
    
    # Use lock to prevent concurrent initialization
    async with initialization_lock:
        # Initialize the model (only once globally)
        if not initialization_completed:
            logger.info(f"Session {session_id}: Starting global initialization...")
            await cl.Message(content="Initializing advanced agent... This might take a minute.").send()
            
            try:
                # Load preprocessed chunks
                documents = load_preprocessed_chunks(PREPROCESSED_CHUNKS_PATH)
                
                # Create retriever (uses global shared client)
                retriever = initialize_retriever(documents)
                
                # Initialize advanced agent workflow (create global shared agent)
                agent = initialize_agent_workflow(retriever)
                
                # Store agent in session (same as global agent)
                cl.user_session.set("agent", agent)
                
                initialization_completed = True
                logger.info(f"Session {session_id}: Global initialization completed successfully")
                
                await cl.Message(
                    content=f"Ready to help! \nI've loaded knowledge from {len(documents)} sections of puppy expertise and can also search the web for up-to-date information."
                ).send()
                
            except Exception as e:
                logger.error(f"Session {session_id}: Error during initialization: {str(e)}")
                await cl.Message(
                    content=f"There was an error initializing the model: {str(e)}"
                ).send()
                # Remove session from active list on error
                async with session_lock:
                    active_sessions.discard(session_id)
        else:
            # System already initialized, use existing global agent
            logger.info(f"Session {session_id}: Using existing global components")
            try:
                agent = global_agent
                cl.user_session.set("agent", agent)
                
                await cl.Message(
                    content="Ready to help! The system is already initialized and ready to answer your questions."
                ).send()
                logger.info(f"Session {session_id}: Session setup completed using global components")
            except Exception as e:
                logger.error(f"Session {session_id}: Error creating agent for session: {str(e)}")
                await cl.Message(
                    content=f"There was an error creating your session: {str(e)}"
                ).send()
                # Remove session from active list on error
                async with session_lock:
                    active_sessions.discard(session_id)

@cl.on_chat_end
async def end():
    """Clean up when session ends."""
    global active_sessions
    
    session_id = cl.user_session.get("session_id", "unknown")
    
    async with session_lock:
        active_sessions.discard(session_id)
        logger.info(f"Session {session_id}: Ended ({len(active_sessions)} active sessions remaining)")

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages using the advanced agent."""
    
    session_id = cl.user_session.get("session_id", "unknown")
    
    # Check if session is still active
    if session_id not in active_sessions:
        await cl.Message(
            content="Your session has expired. Please refresh the page to start a new session."
        ).send()
        return
    
    logger.info(f"Session {session_id}: Received message: {message.content}")
    
    # Get agent from session
    agent = cl.user_session.get("agent")
    
    # Check if agent is initialized
    if not agent or not initialization_completed:
        await cl.Message(
            content="The system is not yet initialized. Please wait or restart the chat."
        ).send()
        return
    
    question = message.content
    
    # Process question with advanced agent
    try:
        logger.info(f"Session {session_id}: Processing question through advanced agent workflow")
        
        # Use agent workflow to process the question
        async def process_with_agent():
            # Execute agent workflow
            result = await cl.make_async(agent.process_question)(question)
            
            # Get final response with tool and sources information
            response_data = agent.get_final_response(result)
            return response_data
        
        # Show loading indicator
        await cl.Message(content="Searching for the best answer...").send()
        
        # Get and send response
        response_data = await process_with_agent()
        
        # The response_data is always a string from get_final_response
        response_content = response_data
        
        logger.info(f"Session {session_id}: Raw response received: {response_content[:200]}...")
        
        # Check if response contains detailed RAG sources (new format)
        if "[Using RAG tool] - Based on" in response_content or "[Using RAG tool] - Basé sur" in response_content:
            # New detailed format - sources are already embedded, just clean markers
            cleaned_response = response_content.replace("[Using RAG tool]", "").strip()
            logger.info(f"Session {session_id}: Using embedded detailed RAG sources")
        elif "[Using RAG tool]" in response_content:
            # Old format with embedded sources - clean and keep as is
            cleaned_response = response_content.replace("[Using RAG tool]", "[RAG Tool]")
            logger.info(f"Session {session_id}: Using embedded RAG sources (old format)")
        elif "[Using Tavily tool]" in response_content:
            # Tavily format
            cleaned_response = response_content.replace("[Using Tavily tool]", "[Tavily Tool]")
            logger.info(f"Session {session_id}: Using Tavily source attribution")
        else:
            # Fallback - add generic RAG attribution
            cleaned_response = response_content
            if cleaned_response and not cleaned_response.startswith("["):
                cleaned_response = f"{cleaned_response}"
            logger.info(f"Session {session_id}: Using Error message")
        
        # Send the final response
        logger.info(f"Session {session_id}: Sending response")
        await cl.Message(content=cleaned_response).send()
        logger.info(f"Session {session_id}: Response sent successfully")
        
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing question with agent: {str(e)}")
        await cl.Message(
            content="I'm sorry, but an error occurred while processing your question. Please try again in a few moments."
        ).send()