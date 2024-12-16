from typing import Dict, List, Optional, Any
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import networkx as nx
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import AgentSlimeMoldOptimizer 
# Load environment variables from .env file
load_dotenv()


# Base Agent Configuration
class AgentState(BaseModel):
    messages: List[Any]
    current_agent: Optional[str] = None
    context: Dict = {}
    path_history: List[str] = []
    performance_metrics: Dict = {}

# Chief Agent Implementation
class ChiefAgent:
    def __init__(self, model_name: str = "gpt-4",recursion_limit: int = 10):
        self.recursion_limit = recursion_limit
        self.recursion_count = 0
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the chief agent coordinating between research, coding, and critic agents."),
            ("human", "{input}"),
            ("ai", "{agent_response}")
        ])
        
    def route_request(self, state: AgentState) -> AgentState:
        
        # Increment and check recursion count
        self.recursion_count += 1
        if self.recursion_count >= self.recursion_limit:
            state.context["task_complete"] = True
            state.current_agent = "end"
            return state
        
        if not state.path_history:  # First request
            state.current_agent = "researcher"
            state.path_history.append("researcher")
            return state
        
        last_message = state.messages[-1]
        last_agent = state.path_history[-1]
        message_content = str(last_message.content).lower()
        
        # After first complete cycle, go to human
        if len(state.path_history) == 3 and last_agent == "critic":
            state.current_agent = "human"
        # After human feedback, analyze and route accordingly
        elif last_agent == "human":
            if "done" in message_content:
                state.context["task_complete"] = True
                state.current_agent = "end"
            elif any(x in message_content for x in [
                "improve", "optimize", "better", "refactor", "implement", 
                "code", "fix", "update", "change"
            ]):
                state.current_agent = "coder"
            elif any(x in message_content for x in [
                "explain", "clarify", "research", "alternative", 
                "background", "how", "why", "what"
            ]):
                state.current_agent = "researcher"
            elif any(x in message_content for x in [
                "review", "test", "check", "verify", "evaluate", 
                "assess", "analyze", "critique"
            ]):
                state.current_agent = "critic"
            else:
                # If unclear, analyze context to make best decision
                if state.context.get("last_code_change"):
                    state.current_agent = "critic"
                elif state.context.get("needs_clarification"):
                    state.current_agent = "researcher"
                else:
                    state.current_agent = "researcher"
        # Regular routing based on context and needs
        else:
            if any(x in message_content for x in [
                "error", "bug", "fix", "implement", "code",
                "function", "class", "script"
            ]):
                state.current_agent = "coder"
                state.context["last_code_change"] = True
            elif any(x in message_content for x in [
                "unclear", "explain", "what", "how", "why",
                "alternative", "background", "context"
            ]):
                state.current_agent = "researcher"
                state.context["needs_clarification"] = True
            elif last_agent == "coder":
                state.current_agent = "critic"
            elif last_agent == "researcher":
                state.current_agent = "coder"
            else:
                state.current_agent = "human"
        
        state.path_history.append(state.current_agent)
        return state

# Specialized Agents
class ResearchAgent:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research agent specialized in analyzing queries and providing detailed information."),
            ("human", "{input}")
        ])
    
    def process(self, state: AgentState) -> AgentState:
        last_message = state.messages[-1]
        original_request = state.messages[0].content
        
        # Analyze the context and previous messages
        context_summary = self._analyze_context(state.messages)
        
        if isinstance(last_message, HumanMessage):
            if "alternative" in last_message.content.lower():
                response = self._research_alternatives(original_request, context_summary)
            else:
                response = self._research_concept(original_request, context_summary)
        else:
            response = self._deep_dive(original_request, context_summary)
        
        state.messages.append(AIMessage(content=response))
        return state
    
    def _analyze_context(self, messages):
        # Analyze message history to understand context
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        return " ".join([msg.content for msg in recent_messages])
    
    def _research_concept(self, request, context):
        # General research about the main concept/topic
        return f"Based on the request '{request}', here's what you need to know:\n" \
               f"1. Core Concept: Understanding the fundamental principles\n" \
               f"2. Common Approaches: Standard ways to solve this type of problem\n" \
               f"3. Key Considerations: Important factors to keep in mind\n" \
               f"4. Best Practices: Industry-standard approaches"
    
    def _research_alternatives(self, request, context):
        # Research alternative approaches
        return f"Here are alternative approaches to solving '{request}':\n" \
               f"1. Standard Approach: Commonly used method\n" \
               f"2. Optimized Approach: For better performance\n" \
               f"3. Alternative Paradigm: Different way of thinking\n" \
               f"4. Trade-offs: Comparing the approaches"
    
    def _deep_dive(self, request, context):
        # Detailed analysis based on previous discussion
        return f"Diving deeper into specific aspects:\n" \
               f"1. Advanced Concepts: Understanding the underlying mechanics\n" \
               f"2. Edge Cases: Important considerations\n" \
               f"3. Performance Implications: Impact on system resources\n" \
               f"4. Integration Aspects: How it fits into larger systems"

class CoderAgent:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a coding agent specialized in implementing technical solutions."),
            ("human", "{input}")
        ])
    
    def process(self, state: AgentState) -> AgentState:
        last_message = state.messages[-1]
        original_request = state.messages[0].content
        message_content = str(last_message.content).lower()
        
        if "optimize" in message_content or "improve" in message_content:
            response = self._generate_optimized_solution(original_request)
        elif "fix" in message_content or "error" in message_content:
            response = self._fix_code(message_content)
        else:
            response = self._generate_initial_solution(original_request)
        
        state.messages.append(AIMessage(content=response))
        return state
    
    def _generate_initial_solution(self, request):
        return f"Here's an implementation addressing '{request}':\n" \
               f"```python\n" \
               f"# Implementation with standard approach\n" \
               f"# Including:\n" \
               f"# - Error handling\n" \
               f"# - Input validation\n" \
               f"# - Basic documentation\n" \
               f"# - Clear variable names\n" \
               f"```"
    
    def _generate_optimized_solution(self, request):
        return f"Here's an optimized implementation:\n" \
               f"```python\n" \
               f"# Optimized implementation including:\n" \
               f"# - Performance improvements\n" \
               f"# - Memory optimization\n" \
               f"# - Advanced error handling\n" \
               f"# - Comprehensive documentation\n" \
               f"```"
    
    def _fix_code(self, error_context):
        return f"Here's the corrected implementation:\n" \
               f"```python\n" \
               f"# Fixed implementation addressing:\n" \
               f"# - Error handling\n" \
               f"# - Bug fixes\n" \
               f"# - Code improvements\n" \
               f"```"

class CriticAgent:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critic agent specialized in reviewing and providing feedback."),
            ("human", "{input}")
        ])
    
    def process(self, state: AgentState) -> AgentState:
        last_message = state.messages[-1]
        code_content = str(last_message.content)
        
        review_points = self._analyze_code(code_content)
        suggestions = self._generate_suggestions(code_content)
        
        response = f"Code Review:\n\n" \
                  f"Analysis:\n{review_points}\n\n" \
                  f"Suggestions:\n{suggestions}"
        
        state.messages.append(AIMessage(content=response))
        return state
    
    def _analyze_code(self, code):
        return "1. Code Structure and Organization\n" \
               "2. Error Handling and Edge Cases\n" \
               "3. Performance Characteristics\n" \
               "4. Code Style and Best Practices\n" \
               "5. Documentation Quality"
    
    def _generate_suggestions(self, code):
        return "1. Potential Improvements\n" \
               "2. Alternative Approaches\n" \
               "3. Performance Optimization Opportunities\n" \
               "4. Error Handling Enhancements\n" \
               "5. Documentation Updates"

# Graph Optimizer
class GraphOptimizer:
    def __init__(self, agent_state):
        self.graph = nx.DiGraph()
        self.optimizer = AgentSlimeMoldOptimizer(agent_state)
        
    def update_weights(self, path: List[str], messages: List[Any]):
        # Use optimizer's path evaluation and strengthening
        path_score = self.optimizer.evaluate_path(messages, path)
        self.optimizer.strengthen_path(path, messages)
        
        # Convert optimizer's edge weights to graph edges
        edges = self.optimizer.convert_to_edges()
        
        # Update graph with new edges and weights
        self.graph.clear()
        for source, target, weight in edges:
            self.graph.add_edge(source, target, weight=weight)

def create_system(recursion_limit: int = 10) -> StateGraph:
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("AgentWorkflow")
    
    # Initialize agents
    chief = ChiefAgent(recursion_limit=recursion_limit)
    researcher = ResearchAgent()
    coder = CoderAgent()
    critic = CriticAgent()
    
    workflow = StateGraph(AgentState)
    
    # Add logging to each node
    def log_and_process(agent_name, process_func):
        def wrapper(state: AgentState):
            logger.info(f"=== {agent_name.upper()} STEP ===")
            logger.info(f"Path history: {state.path_history}")
            logger.info(f"Last message: {state.messages[-1].content if state.messages else 'None'}")
            result = process_func(state)
            logger.info(f"{agent_name} output: {result.messages[-1].content if result.messages else 'None'}")
            return result
        return wrapper
    
    # Add nodes with logging
    workflow.add_node("route", log_and_process("router", chief.route_request))
    workflow.add_node("research", log_and_process("researcher", researcher.process))
    workflow.add_node("code", log_and_process("coder", coder.process))
    workflow.add_node("critique", log_and_process("critic", critic.process))
    workflow.add_node("human_feedback", log_and_process("human", get_human_feedback))
    
    
    # Define edge routing based on state's current_agent
    def route_next(state: AgentState) -> str:
        if state.current_agent == "end" or chief.recursion_count > recursion_limit  :
            return END
        return {
            "researcher": "research",
            "coder": "code",
            "critic": "critique",
            "human": "human_feedback"
        }.get(state.current_agent, "route")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "route",
        route_next,
        {
            "research": "research",
            "code": "code",
            "critique": "critique",
            "human_feedback": "human_feedback",
            END: END
        }
    )
    
    # Add return edges to route
    workflow.add_edge("research", "route")
    workflow.add_edge("code", "route")
    workflow.add_edge("critique", "route")
    workflow.add_edge("human_feedback", "route")
    
    # Set entry point
    workflow.set_entry_point("route")
    
    return workflow.compile()

def get_human_feedback(state: AgentState) -> AgentState:
    print("\nCurrent progress:")
    for msg in state.messages[-3:]:  # Show last few messages
        print(f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content[:200]}...")
    
    feedback = input("\nPlease provide feedback or type 'done' if satisfied: ")
    state.messages.append(HumanMessage(content=feedback))
    
    if feedback.lower() == 'done':
        state.context["task_complete"] = True
    
    return state

# Example usage
if __name__ == "__main__":
    system = create_system(recursion_limit=10)
    initial_state = AgentState(
        messages=[HumanMessage(content="Create a Python function to calculate fibonacci numbers")],
        current_agent=None,
        path_history=[],
        context={},
        performance_metrics={}
    )
    
    # Run the workflow
    result = system.invoke(initial_state)
    
    # Print results
    print("\nFinal Messages:")
    # The result is now a dict-like object, access the state attributes directly
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"AI: {msg.content[:200]}...")
        elif isinstance(msg, HumanMessage):
            print(f"Human: {msg.content}")
    
    print("\nPath History:")
    print(result["path_history"])
