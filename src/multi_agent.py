# src/multiagent.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils import get_logger, trace_function, log_state, performance_monitor

class AgentState(BaseModel):
    """State management for multi-agent system"""
    messages: List[Any] = Field(default_factory=list)
    current_agent: Optional[str] = Field(None)
    context: Dict = Field(default_factory=dict)
    path_history: List[str] = Field(default_factory=list)

class MultiAgentSystem:
    """Unified multi-agent system implementing all agent types and routing logic"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.logger = get_logger(self.__class__.__name__)
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.prompts = self._initialize_prompts()
        
        # Initialize routing tool
        self.routing_tool = Tool(
            name="route_request",
            func=self._route_based_on_path,
            description="Route requests based on agent path history"
        )
        
        self.logger.info(f"Initialized MultiAgentSystem with model: {model_name}")

    def _initialize_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Initialize prompts for all agent types"""
        return {
            "chief": ChatPromptTemplate.from_messages([
                ("system", """You are the chief agent coordinating between specialist agents:
                - Researcher: For analysis and information gathering
                - Coder: For implementation
                - Critic: For review and validation
                
                Analyze the current context and decide which agent should handle the next step."""),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ]),
            
            "researcher": ChatPromptTemplate.from_messages([
                ("system", """You are a research agent. Analyze the problem and provide:
                - Key concepts and approaches
                - Potential challenges
                - Best practices
                Keep responses focused and technical."""),
                ("human", "{input}")
            ]),
            
            "coder": ChatPromptTemplate.from_messages([
                ("system", """You are a coding agent. Provide:
                - Clean, efficient code implementations
                - Code explanations
                - Error handling
                Use Pythonic conventions and best practices."""),
                ("human", "{input}")
            ]),
            
            "critic": ChatPromptTemplate.from_messages([
                ("system", """You are a code critic agent. Analyze the code and provide:
                - Code review comments
                - Optimization suggestions
                - Potential issues
                - Best practices recommendations"""),
                ("human", "{input}")
            ])
        }

    @trace_function
    @performance_monitor(threshold_ms=100)
    def _route_based_on_path(self, state: AgentState) -> str:
        """Determine next agent based on path history"""
        last_agent = state.path_history[-1] if state.path_history else None
        
        if last_agent == "critic" and len(state.path_history) == 3:
            return "human"
        elif last_agent == "human":
            last_message = state.messages[-1].content.lower()
            if "done" in last_message:
                return "end"
            return "chief"
        elif not last_agent or last_agent == "researcher":
            return "coder"
        elif last_agent == "coder":
            return "critic"
        return "researcher"

    @trace_function
    def process_chief(self, state: AgentState) -> str:
        """Chief agent processing"""
        log_state(state, self.logger)
        
        if len(state.path_history) < 3:
            return self.routing_tool.run(state)
            
        last_message = state.messages[-1].content
        prompt_input = {
            "input": f"""
            Last message: {last_message}
            Path history: {state.path_history}
            Context: {state.context}
            
            Decide next agent: researcher, coder, critic, or human?
            """,
            "agent_scratchpad": ""
        }
        
        response = self.llm(self.prompts["chief"].format_messages(**prompt_input))
        return self._parse_agent_decision(response.content)

    @trace_function
    def process_researcher(self, state: AgentState) -> AgentState:
        """Researcher agent processing"""
        log_state(state, self.logger)
        
        last_message = state.messages[-1].content
        chain = self.prompts["researcher"] | self.llm
        
        response = chain.invoke({"input": last_message})
        state.messages.append(AIMessage(content=response.content))
        state.path_history.append("researcher")
        return state

    @trace_function
    def process_coder(self, state: AgentState) -> AgentState:
        """Coder agent processing"""
        log_state(state, self.logger)
        
        last_message = state.messages[-1].content
        chain = self.prompts["coder"] | self.llm
        
        response = chain.invoke({"input": last_message})
        state.messages.append(AIMessage(content=response.content))
        state.path_history.append("coder")
        return state

    @trace_function
    def process_critic(self, state: AgentState) -> AgentState:
        """Critic agent processing"""
        log_state(state, self.logger)
        
        last_message = state.messages[-1].content
        chain = self.prompts["critic"] | self.llm
        
        response = chain.invoke({"input": last_message})
        state.messages.append(AIMessage(content=response.content))
        state.path_history.append("critic")
        return state

    def _parse_agent_decision(self, response: str) -> str:
        """Parse the agent decision from LLM response"""
        response = response.lower()
        if "researcher" in response:
            return "researcher"
        elif "coder" in response:
            return "coder"
        elif "critic" in response:
            return "critic"
        elif "human" in response:
            return "human"
        return "researcher"  # default fallback

class WorkflowManager:
    """Manages the workflow between agents"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.memory = MemorySaver()
        self.multi_agent = MultiAgentSystem()
        self.graph = self.create_agent_graph()
        
    @trace_function
    def create_agent_graph(self):
        """Create the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("researcher", self.multi_agent.process_researcher)
        workflow.add_node("coder", self.multi_agent.process_coder)
        workflow.add_node("critic", self.multi_agent.process_critic)
        workflow.add_node("human", self.handle_human_input)
        
        # Add edges with routing
        for agent in ["researcher", "coder", "critic", "human"]:
            workflow.add_conditional_edges(
                agent,
                lambda x: self.multi_agent.process_chief(x),
                {
                    "researcher": "researcher",
                    "coder": "coder",
                    "critic": "critic",
                    "human": "human",
                    "end": END
                }
            )
        
        workflow.set_entry_point("researcher")
        return workflow.compile()

    @trace_function
    def handle_human_input(self, state: AgentState) -> AgentState:
        """Handle human input interactions"""
        print("\nCurrent conversation:")
        for msg in state.messages[-3:]:
            print(f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content[:200]}...")
        
        human_input = input("\nReview the solution. Type 'done' to complete or provide feedback: ")
        state.messages.append(HumanMessage(content=human_input))
        state.path_history.append("human")
        return state

    @trace_function
    def run_workflow(self, input_query: str) -> AgentState:
        """Run the agent workflow"""
        self.logger.info(f"Starting workflow with query: {input_query}")
        
        initial_state = AgentState(
            messages=[HumanMessage(content=input_query)],
            current_agent="researcher",
            path_history=[],
            context={}
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            raise
    

    def custom_workflow_example():
        # Initialize with custom model
        multi_agent = MultiAgentSystem(model_name="gemini-pro")
        workflow = WorkflowManager()
        
        # Multiple queries example
        queries = [
            "Implement a quick sort algorithm in Python",
            "Create a simple REST API using FastAPI",
            "Write a unit test for a database connection",
        ]
        
        results = []
        for query in queries:
            print(f"\nProcessing query: {query}")
            final_state = workflow.run_workflow(query)
            results.append(final_state)
            
        return results

if __name__ == "__main__":
    
    results = custom_workflow_example()
    
    # Analyze results
    for i, state in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Path taken: {state.path_history}")
        print(f"Number of interactions: {len(state.messages)}")
