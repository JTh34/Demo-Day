# Fichier README.md pour Hugging Face Spaces
# À placer à la racine de votre Space

---
title: PuppyCompanion
emoji: 🐶
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# PuppyCompanion 🐶

An intelligent AI assistant specialized in puppy care and training, powered by advanced RAG (Retrieval-Augmented Generation) technology and a modern FastAPI backend.

## Features

- **🐕 Specialized Knowledge Base**: Expert information from professional canine care resources
- **🧠 Smart Agent Workflow**: LangGraph-powered decision making with tool selection
- **🔍 Dual Search System**: 
  - RAG system for specialized puppy knowledge from preprocessed chunks
  - Web search integration via Tavily for up-to-date information
- **📱 Modern Mobile-like Interface**: Responsive web interface with real-time updates
- **🔧 Real-time Debug Console**: Watch the AI's decision process in real-time
- **⚡ FastAPI Backend**: High-performance async API with WebSocket support
- **📊 Detailed Source Attribution**: Shows exactly which knowledge sources were used

## How it Works

1. **Question Analysis**: The agent workflow analyzes your question and selects appropriate tools
2. **RAG Search**: For dog-related questions, searches the specialized knowledge base using Qdrant vector database
3. **Quality Evaluation**: Determines if the RAG response is comprehensive enough
4. **Web Search Fallback**: If needed, uses Tavily to search for additional current information
5. **Response Generation**: Combines knowledge sources to provide a comprehensive answer
6. **Real-time Updates**: All processing steps are visible in the debug console via WebSocket

## Usage

Simply ask any question about puppy care, training, behavior, or health. Examples:
- "How do I house train my 8-week-old puppy?"
- "What vaccination schedule should I follow?"
- "How to stop my puppy from biting furniture?"
- "Best foods for a growing German Shepherd puppy?"

## Technology Stack

- **FastAPI**: Modern async web framework for the API backend
- **LangGraph**: Agent workflow orchestration and tool selection
- **OpenAI GPT-4**: Language model for natural language processing
- **Qdrant**: Vector database for document embeddings and similarity search
- **OpenAI Embeddings**: Text embeddings for semantic search
- **Tavily**: Web search integration for real-time information
- **WebSocket**: Real-time communication for debug console
- **Docker**: Containerized deployment for HuggingFace Spaces

## Configuration

The application requires these environment variables:

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for LLM processing and embeddings

### Optional
- `TAVILY_API_KEY`: For web search functionality (highly recommended)

**⚠️ Important**: Configure these as secrets in your HuggingFace Space settings:
1. Go to your space settings
2. Add the API keys in the "Repository secrets" section
3. Never commit API keys to your repository

## Knowledge Base

The application uses a preprocessed knowledge base of professional puppy care resources:
- **File**: `all_books_preprocessed_chunks.json` (2.5MB)
- **Content**: Chunked and processed expert knowledge about puppy training, health, and behavior
- **Vector Store**: Automatically indexed in Qdrant on first startup

## API Endpoints

- `GET /`: Main web interface
- `POST /chat`: Chat API endpoint for programmatic access
- `WebSocket /ws`: Real-time debug console connection
- `GET /health`: Application health check

## Development

The application features:
- **Async Architecture**: Full async/await support for optimal performance  
- **Persistent Vector Store**: Qdrant storage persists between restarts
- **Connection Management**: WebSocket connection handling for multiple users
- **Error Handling**: Comprehensive error handling and logging
- **Health Monitoring**: Built-in health checks and monitoring

---

*Transform the chaos of new puppy ownership into confidence with AI-powered expert assistance!* 🐾