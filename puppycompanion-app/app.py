# app.py - Refactored version to use the advanced agent
import os
import logging
import time
import uuid
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

# Global variables
agent = None
initialization_completed = False


def load_preprocessed_chunks(file_path):
    """Load preprocessed chunks from a JSON file."""
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
        return documents
    except Exception as e:
        logger.error(f"Error loading preprocessed chunks: {str(e)}")
        raise

def initialize_retriever(documents):
    """Create a retriever from documents."""
    logger.info("Creating retriever from documents")
    try:
        # Use langchain_qdrant to create a vector store
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        
        # Create embedding object
        embeddings = OpenAIEmbeddings()
        logger.info("Created OpenAI embeddings object")
        
        # Create a unique path for each session to avoid permission issues
        session_id = str(uuid.uuid4())
        qdrant_path = f"/tmp/qdrant_storage_{session_id}"
        logger.info(f"Using unique Qdrant storage path: {qdrant_path}")
        
        # Create directory for Qdrant storage
        os.makedirs(qdrant_path, exist_ok=True)
        
        # Use local storage
        client = QdrantClient(path=qdrant_path)
        logger.info("Created Qdrant client with local storage")
        
        # OpenAI embeddings dimension
        embedding_dim = 1536
        
        # Create collection
        from qdrant_client.http import models
        client.create_collection(
            collection_name="puppies",
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="puppies",
            embedding=embeddings
        )
        
        # Add documents
        vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        logger.info("Created retriever")
        
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise

def initialize_agent_workflow(retriever):
    """Initialize the advanced agent workflow."""
    logger.info("Initializing advanced agent workflow")
    try:
        # Create RAG system
        rag_system = RAGSystem(retriever)
        rag_tool = rag_system.create_rag_tool()
        
        # Create agent workflow that uses the RAG tool
        agent = AgentWorkflow(rag_tool)
        logger.info("Advanced agent workflow initialized successfully")
        
        return agent
    except Exception as e:
        logger.error(f"Error initializing agent workflow: {str(e)}")
        raise

@cl.on_chat_start
async def start():
    """Initialize the application at session start."""
    global agent, initialization_completed
    
    logger.info("Starting chat session")
    
    # Display welcome message
    await cl.Message(
        content="""
        ### Welcome to PuppyCompanion 🐶
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
            logger.info("OpenAI API key provided by user")
        else:
            await cl.Message(content="No API key provided. Unable to continue.").send()
            return
    
    # Initialize the model
    await cl.Message(content="Initializing advanced agent... This might take a minute.").send()
    
    try:
        # Load preprocessed chunks
        documents = load_preprocessed_chunks(PREPROCESSED_CHUNKS_PATH)
        
        # Create retriever
        retriever = initialize_retriever(documents)
        
        # Initialize advanced agent workflow
        agent = initialize_agent_workflow(retriever)
        
        initialization_completed = True
        
        await cl.Message(
            content=f"Ready to help! \nI've loaded knowledge from {len(documents)} sections of puppy expertise and can also search the web for up-to-date information."
        ).send()
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        await cl.Message(
            content=f"There was an error initializing the model: {str(e)}"
        ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages using the advanced agent."""
    global agent, initialization_completed
    
    logger.info(f"Received message: {message.content}")
    
    # Check if agent is initialized
    if not initialization_completed:
        await cl.Message(
            content="The system is not yet initialized. Please wait or restart the chat."
        ).send()
        return
    
    question = message.content
    
    # Process question with advanced agent
    try:
        logger.info("Processing question through advanced agent workflow")
        
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
        
        # Format message based on the tool used
        response_content = response_data["content"]
        tool_used = response_data.get("tool_used")
        sources = response_data.get("sources", [])
        
        logger.info(f"Preparing response with tool: {tool_used}, sources: {sources}")
        
        # Format the source information based on the tool used
        source_text = ""
        if tool_used == "rag":
            source_text = "\n\n[RAG Tool - Response from the book \"Puppies for Dummies\"]"
            logger.info("Using RAG source attribution")
        elif tool_used == "tavily":
                source_text = "\n\n[Tavily Tool - Response from the Internet]"
                logger.info("Using generic Tavily source attribution")
        
        # Send the final response with source information
        logger.info(f"Sending response with source: {source_text}")
        await cl.Message(content=response_content + source_text).send()
        logger.info("Response sent successfully")
        
    except Exception as e:
        logger.error(f"Error processing question with agent: {str(e)}")
        await cl.Message(
            content="I'm sorry, but an error occurred while processing your question. Please try again in a few moments."
        ).send()