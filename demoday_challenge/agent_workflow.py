# agent_workflow.py
import logging
from typing import Dict, List, Any, Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Agent state for the workflow"""
    messages: Annotated[list, add_messages]
    context: List[Document]
    next_tool: str
    question: str

class AgentWorkflow:
    """Agent workflow with intelligent routing logic"""
    
    def __init__(self, rag_tool, tavily_max_results: int = 5):
        """ Initialize the agent workflow """
        self.rag_tool = rag_tool
        self.tavily_tool = TavilySearchResults(max_results=tavily_max_results)
        
        # LLMs for routing and evaluation
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
        self.evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Compile the workflow
        self.compiled_workflow = self._build_workflow()
    
    def evaluate_response_quality(self, question: str, response: str) -> bool:
        """ Evaluates if the response is satisfactory """
        prompt = f"""Evaluate if this response to "{question}" is UNSATISFACTORY:

        "{response}"

        UNSATISFACTORY CRITERIA (if ANY ONE is present, the response is UNSATISFACTORY):
        1. Contains "consult experts", "specialized training", "I'm sorry"
        2. Doesn't provide concrete steps for "how to" questions
        3. Gives general advice rather than specific methods
        4. Redirects the user without directly answering

        Quick example:
        Q: "How do I train my dog to sit?"
        UNSATISFACTORY: "Consult a professional trainer."
        SATISFACTORY: "1. Use treats... 2. Be consistent..."

        Reply only "UNSATISFACTORY" or "SATISFACTORY".
        When in doubt, choose "UNSATISFACTORY".
        """
        
        evaluation = self.evaluator_llm.invoke([SystemMessage(content=prompt)])
        result = evaluation.content.strip().upper()
        
        is_satisfactory = "UNSATISFACTORY" not in result
        logger.info(f"[Evaluation] Response rated: {'SATISFACTORY' if is_satisfactory else 'UNSATISFACTORY'}")
        
        return is_satisfactory
    
    def _build_workflow(self):
        """Builds and compiles the agent workflow"""
        
        # 1. Node for intelligent routing
        def smart_router(state):
            """Determines if the question is about dogs or not"""
            messages = state["messages"]
            last_message = [msg for msg in messages if isinstance(msg, HumanMessage)][-1]
            question = last_message.content
            
            # Prompt using reverse logic - asking if it's NOT related to dogs
            router_prompt = f"""Evaluate if this question is UNRELATED to dogs, puppies, or canine care:

            Question: "{question}"

            INDICATORS OF NON-DOG QUESTIONS (if ANY ONE is present, mark as "NOT_DOG_RELATED"):
            1. Questions about weather, time, locations, or general information
            2. Questions about other animals (cats, birds, etc.)
            3. Questions about technology, politics, or human activities
            4. Any question that doesn't explicitly mention or imply dogs/puppies/canines

            Example check:
            Q: "What is the weather in Paris today?"
            This is NOT_DOG_RELATED (about weather)

            Q: "How do I train my puppy to sit?"
            This is DOG_RELATED (explicitly about puppy training)

            Reply ONLY with "NOT_DOG_RELATED" or "DOG_RELATED".
            When in doubt, choose "NOT_DOG_RELATED".
            """
            
            router_response = self.router_llm.invoke([SystemMessage(content=router_prompt)])
            result = router_response.content.strip().upper()
            
            is_dog_related = "NOT_DOG_RELATED" not in result
            logger.info(f"[Smart Router] Question {'' if is_dog_related else 'NOT '}related to dogs")
            
            # If the question is not related to dogs, go directly to out_of_scope
            if not is_dog_related:
                return {
                    "next_tool": "out_of_scope",
                    "question": question
                }
            
            # If the question is related to dogs, go to the RAG tool
            return {
                "next_tool": "rag_tool",
                "question": question
            }
        
        # 2. Node for out-of-scope questions
        def out_of_scope(state):
            """Informs that the assistant only answers questions about dogs"""
            out_of_scope_message = AIMessage(
                content="I'm sorry, but I specialize only in canine care and puppy education. I cannot answer this question as it is outside my area of expertise. Feel free to ask me any questions about dogs and puppies!"
            )
            
            return {
                "messages": [out_of_scope_message],
                "next_tool": "final_response"
            }
        
        # 3. Node for using the RAG tool
        def use_rag_tool(state):
            """Uses the RAG tool for dog-related questions"""
            question = state["question"]
            
            # Call the RAG tool directly
            rag_result = self.rag_tool(question)
            rag_response = rag_result["messages"][0].content
            context = rag_result.get("context", [])
            
            # Evaluate the quality of the response
            is_satisfactory = self.evaluate_response_quality(question, rag_response)
            
            # Create an AI message with the response
            response_message = AIMessage(content=f"[Using RAG tool] {rag_response}")
            
            # If the response is not satisfactory, prepare to use Tavily
            next_tool = "final_response" if is_satisfactory else "need_tavily"
            
            return {
                "messages": [response_message],
                "context": context,
                "next_tool": next_tool
            }
        
        # 4. Node for using the Tavily tool
        def use_tavily_tool(state):
            """Uses the Tavily tool as a fallback for dog-related questions"""
            question = state["question"]
            
            # Call Tavily
            tavily_result = self.tavily_tool.invoke(question)
            
            # Check if we got useful results
            has_useful_results = False
            tavily_content = ""
            
            if tavily_result:
                tavily_content = "Internet search results:\n\n"
                for i, result in enumerate(tavily_result[:3], 1):
                    content = result.get('content', '')
                    if content and len(content.strip()) > 50:  # Basic check for meaningful content
                        has_useful_results = True
                        tavily_content += f"{i}. {result.get('title', 'Source')}: {content[:200]}...\n\n"
            else:
                tavily_content = "No relevant results found on the Internet."
            
            # Create a message with the results
            response_message = AIMessage(content=f"[Using Tavily tool] {tavily_content}")
            
            # Go to "don't know" only if no useful results were found
            next_tool = "final_response" if has_useful_results else "say_dont_know"
            
            return {
                "messages": [response_message],
                "next_tool": next_tool
            }
        
        # 5. Node for cases where no source has a satisfactory answer
        def say_dont_know(state):
            """Responds when no source has useful information"""
            question = state["question"]
            
            dont_know_message = AIMessage(content=f"I'm sorry, but I couldn't find specific information about '{question}' in my knowledge base or through online searches. This might be a specialized topic that requires expertise from professionals in the field of canine education.")
            
            return {
                "messages": [dont_know_message],
                "next_tool": "final_response"
            }
        
        # 6. Node for generating the final response
        def generate_final_response(state):
            """Generates a final response based on tool results"""
            messages = state["messages"]
            original_question = state["question"]
            
            # Find tool messages
            tool_responses = [msg.content for msg in messages if isinstance(msg, AIMessage)]
            
            # If no tool messages, return a default response
            if not tool_responses:
                return {"messages": [AIMessage(content="I couldn't find information about your dog-related question.")]}
            
            # Take the last tool message as the main content
            tool_content = tool_responses[-1]
            
            # Use an LLM to generate a coherent final response
            system_prompt = f"""Here are the search results for the dog-related question: "{original_question}"

            {tool_content}

            Formulate a clear, helpful, and concise response based ONLY on these results.
            Remove any text like "[Using RAG tool]" or "[Using Tavily tool]" from your response.
            If the search results contain useful information, include it in your response rather than saying "I don't know".
            Say "I don't know" only if the search results contain no useful information.
            """
            
            response = self.final_llm.invoke([SystemMessage(content=system_prompt)])
            
            return {"messages": [response]}
        
        # 7. Routing function
        def route_to_next_tool(state):
            next_tool = state["next_tool"]
            
            if next_tool == "rag_tool":
                return "use_rag_tool"
            elif next_tool == "out_of_scope": 
                return "out_of_scope"
            elif next_tool == "tavily_tool":
                return "use_tavily_tool"
            elif next_tool == "need_tavily":
                return "use_tavily_tool"
            elif next_tool == "say_dont_know":
                return "say_dont_know"
            elif next_tool == "final_response":
                return "generate_response"
            else:
                return "generate_response"
        
        # 8. Building the LangGraph
        workflow = StateGraph(AgentState)
        
        # Adding nodes
        workflow.add_node("smart_router", smart_router)
        workflow.add_node("out_of_scope", out_of_scope)
        workflow.add_node("use_rag_tool", use_rag_tool)
        workflow.add_node("use_tavily_tool", use_tavily_tool)
        workflow.add_node("say_dont_know", say_dont_know)
        workflow.add_node("generate_response", generate_final_response)
        
        # Connections
        workflow.add_edge(START, "smart_router")
        workflow.add_conditional_edges("smart_router", route_to_next_tool)
        workflow.add_edge("out_of_scope", "generate_response")
        workflow.add_conditional_edges("use_rag_tool", route_to_next_tool)
        workflow.add_conditional_edges("use_tavily_tool", route_to_next_tool)
        workflow.add_edge("say_dont_know", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile the graph
        return workflow.compile()
    
    def process_question(self, question: str):
        """ Process a question with the agent workflow """
        # Invoke the workflow
        result = self.compiled_workflow.invoke({
            "messages": HumanMessage(content=question),
            "context": [],
            "next_tool": "",
            "question": ""
        })
        
        return result
    
    def get_final_response(self, result):
        """ Extracts the final response from the workflow result """
        # Find the last AI message that is not a tool result
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("[Using"):
                return msg.content
        
        # Fallback: take the last AI message, whatever it is
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content.replace("[Using RAG tool] ", "").replace("[Using Tavily tool] ", "")
        
        return "I couldn't generate a response."
