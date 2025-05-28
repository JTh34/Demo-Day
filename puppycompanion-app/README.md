---
title: puppycompanion
emoji: 🐶
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PuppyCompanion 🐶

Your intelligent assistant for puppies training, based on RAG (Retrieval-Augmented Generation) using OpenAI and Langchain.

## Features

- **Canine expertise**: Trained on veterinary and dog training sources
- **Intelligent questions**: Automatically detects if your questions concern dogs
- **Reliable information**: Uses verified sources such as "Puppies For Dummies"
- **Backup web search**: Uses Tavily to search the web when knowledge base is insufficient
- **Quality checks**: Evaluates response quality and improves them automatically
- **Safe advice**: Recognizes when to recommend a veterinary consultation

## Technologies

- LangChain for RAG implementation
- LangGraph for advanced agent workflow orchestration
- OpenAI for embeddings and LLMs
- Chainlit for the user interface
- Qdrant for vector storage
- Tavily for web search capabilities

## Architecture

PuppyCompanion uses an advanced agent architecture:
1. **Intelligent routing**: Detects if the question is about dogs
2. **RAG system**: Retrieves relevant information from a specialized knowledge base
3. **Quality evaluation**: Checks if the answer is satisfactory
4. **Web search**: Uses Tavily as a fallback if necessary
5. **Coherent final response**: Composes a final answer based on all sources
