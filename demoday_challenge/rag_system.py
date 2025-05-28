# rag_system.py
import logging
from typing import Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RAG prompt for puppy-related questions
RAG_PROMPT =    """
                You are an assistant specialized in puppy education and care.
                Your role is to help new puppy owners by answering their questions with accuracy and kindness.
                Use only the information provided in the context to formulate your answers.
                If you cannot find the information in the context, just say "I don't know".

                ### Question
                {question}

                ### Context
                {context}
                """

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

class RAGSystem:
    """RAG system for puppy-related questions"""
    
    def __init__(self, retriever, model_name: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model_name)
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.graph_rag = self._build_graph()
    
    def _build_graph(self):
        """Builds the RAG graph"""
        
        def retrieve(state):
            retrieved_docs = self.retriever.invoke(state["question"])
            return {"context": retrieved_docs}

        def generate(state):
            docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
            messages = self.rag_prompt.format_messages(
                question=state["question"], 
                context=docs_content
            )
            response = self.llm.invoke(messages)
            return {"response": response.content}

        # Build the graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    
    def process_query(self, question: str) -> Dict:
        """ Processes a query and returns the response with context """
        result = self.graph_rag.invoke({"question": question})
        
        return {
            "response": result["response"],
            "context": result["context"]
        }
    
    def create_rag_tool(self):
        """Creates a RAG tool for the agent"""
        
        # Reference to the current instance to use it in the tool
        rag_system = self
        
        @tool
        def ai_rag_tool(question: str) -> Dict:
            """MANDATORY for all questions about puppies, their behavior, education or training.
            This tool accesses a specialized knowledge base on puppies with expert and reliable information.
            Any question regarding puppy care, education, behavior or health MUST be processed by this tool.
            The input must be a complete question."""
            
            # Invoke the RAG graph
            result = rag_system.process_query(question)
            
            return {
                "messages": [HumanMessage(content=result["response"])],
                "context": result["context"]
            }
        
        return ai_rag_tool
