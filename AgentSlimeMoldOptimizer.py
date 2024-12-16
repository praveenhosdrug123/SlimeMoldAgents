from typing import Dict, List, Tuple, Set, Any, heapq
import random
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import numpy as np

class AgentSlimeMoldOptimizer:
    def __init__(self, agent_state):
        self.agent_state = agent_state
        self.best_path: List[str] = []
        self.best_score = 0.0
        self.agent_types = ["researcher", "coder", "critic", "human"]
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        # Compute embeddings
        embedding1 = self.sentence_model.encode([text1], convert_to_tensor=True)
        embedding2 = self.sentence_model.encode([text2], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = np.inner(embedding1, embedding2)
        return float(similarity)
    
    def analyze_human_sentiment(self, response: str) -> float:
        # Analyze sentiment using TextBlob
        sentiment = TextBlob(response).sentiment.polarity
        
        # Check for positive keywords
        positive_keywords = ['great', 'good', 'excellent', 'done', 'perfect', 
                           'thanks', 'helpful', 'amazing', 'wonderful', 'fantastic']
        keyword_score = sum(1 for word in positive_keywords if word in response.lower()) * 0.1
        
        # Combine sentiment and keyword presence
        combined_score = (sentiment + 1) / 2 + keyword_score  # Normalize sentiment to 0-1 range
        return min(1.0, combined_score)
    
    def evaluate_path(self, messages: List[Any], path: List[str]) -> float:
        score = 0.0
        
        # Extract responses from messages
        responses = [msg.content for msg in messages if hasattr(msg, 'content')]
        
        for i, response in enumerate(responses):
            # Response quality metrics
            length_score = min(1.0, 100 / len(response))
            
            # Context coherence score using sentence transformers
            coherence_score = 0.5
            if i > 0:
                prev_msg = responses[i-1]
                similarity_score = self.get_semantic_similarity(prev_msg, response)
                coherence_score = similarity_score  # Already normalized between 0-1
            
            # Path relevance and role scoring
            role_score = 0.5
            if i < len(path):
                current_role = path[i]
                if current_role == "human":
                    # For human responses, use sentiment analysis
                    role_score = self.analyze_human_sentiment(response)
                else:
                    # For AI agents, check role-specific keywords and context
                    role_keywords = {
                        "researcher": ["analysis", "research", "study", "investigate", "findings"],
                        "coder": ["code", "implementation", "function", "class", "algorithm"],
                        "critic": ["review", "evaluate", "consider", "improve", "suggestion"]
                    }
                    
                    if current_role in role_keywords:
                        keywords = role_keywords[current_role]
                        keyword_matches = sum(1 for keyword in keywords 
                                           if keyword in response.lower())
                        role_score = min(1.0, keyword_matches * 0.2 + 0.5)
            
            # Combine scores with weights
            step_score = (
                length_score * 0.2 +      # 20% weight for length
                coherence_score * 0.5 +   # 50% weight for coherence
                role_score * 0.3          # 30% weight for role relevance
            )
            
            score += step_score / len(responses)
            
        return score
    
    def evaluate_node_transition(self, current_response: str, next_response: str, 
                               current_role: str, next_role: str) -> float:
        # Semantic coherence between responses
        semantic_score = self.get_semantic_similarity(current_response, next_response)
        
        # Role-specific evaluation
        role_transition_scores = {
            ("researcher", "coder"): lambda: self.evaluate_research_to_code(current_response, next_response),
            ("coder", "critic"): lambda: self.evaluate_code_to_critic(current_response, next_response),
            ("critic", "human"): lambda: self.evaluate_critic_to_human(current_response, next_response),
            ("human", "researcher"): lambda: self.evaluate_human_to_research(current_response, next_response)
        }
        
        transition_score = role_transition_scores.get(
            (current_role, next_role), 
            lambda: 0.5
        )()
        
        # Combine scores
        return (semantic_score * 0.6 + transition_score * 0.4)

    def evaluate_research_to_code(self, research_response: str, code_response: str) -> float:
        # Check if code implementation reflects research findings
        research_keywords = set(["should", "could", "implement", "design", "approach"])
        code_indicators = set(["def", "class", "function", "return", "import"])
        
        research_ideas = sum(1 for word in research_keywords 
                           if word in research_response.lower())
        code_implementation = sum(1 for word in code_indicators 
                                if word in code_response.lower())
        
        return min(1.0, (research_ideas + code_implementation) * 0.2)

    def evaluate_code_to_critic(self, code_response: str, critic_response: str) -> float:
        # Check if critic provides meaningful code review
        code_patterns = ["def", "class", "function", "return"]
        review_patterns = ["improve", "suggest", "review", "better", "consider"]
        
        has_code = any(pattern in code_response for pattern in code_patterns)
        has_review = any(pattern in critic_response.lower() 
                        for pattern in review_patterns)
        
        return 1.0 if (has_code and has_review) else 0.5

    def evaluate_critic_to_human(self, critic_response: str, human_response: str) -> float:
        # Evaluate if human response acknowledges critic's feedback
        sentiment_score = self.analyze_human_sentiment(human_response)
        
        # Check if human response references critic's points
        critic_keywords = set(critic_response.lower().split())
        human_keywords = set(human_response.lower().split())
        reference_score = len(critic_keywords & human_keywords) * 0.1
        
        return min(1.0, sentiment_score * 0.7 + reference_score * 0.3)

    def evaluate_human_to_research(self, human_response: str, research_response: str) -> float:
        # Check if research addresses human's input
        human_sentiment = self.analyze_human_sentiment(human_response)
        
        # Check if research response contains analytical elements
        research_indicators = ["analysis", "study", "investigate", "research", "findings"]
        research_score = sum(1 for word in research_indicators 
                           if word in research_response.lower()) * 0.2
        
        return min(1.0, human_sentiment * 0.4 + research_score * 0.6)

    def strengthen_path(self, path: List[str], messages: List[Any]):
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Initialize edge weights if needed
            if current not in self.agent_state.edge_weights:
                self.agent_state.edge_weights[current] = {}
            if next_node not in self.agent_state.edge_weights[current]:
                self.agent_state.edge_weights[current][next_node] = 0.5  # Initial weight
            
            # Get responses for current and next node
            current_response = messages[i].content if i < len(messages) else ""
            next_response = messages[i + 1].content if i + 1 < len(messages) else ""
            
            # Evaluate transition quality
            transition_score = self.evaluate_node_transition(
                current_response, next_response, current, next_node
            )
            
            # Adjust edge weight based on transition score
            current_weight = self.agent_state.edge_weights[current][next_node]
            
            if transition_score > 0.6:  # Good transition
                # Increase weight but cap at 1.0
                new_weight = min(1.0, current_weight + transition_score * 0.1)
            else:  # Poor transition
                # Decrease weight but keep minimum of 0.1
                new_weight = max(0.1, current_weight - (1 - transition_score) * 0.1)
            
            self.agent_state.edge_weights[current][next_node] = new_weight
    
    def convert_to_edges(self) -> List[Tuple[str, str, float]]:
        """Convert edge_weights dictionary to edges list format"""
        edges = []
        for source in self.agent_state.edge_weights:
            for target, weight in self.agent_state.edge_weights[source].items():
                # Convert weight to distance (inverse of weight as Dijkstra finds shortest path)
                distance = 1.0 - weight
                edges.append((source, target, distance))
        return edges

    def get_agent_index(self, agent_type: str) -> int:
        """Convert agent type to numeric index for Dijkstra"""
        return self.agent_types.index(agent_type) + 1

    def get_agent_type(self, index: int) -> str:
        """Convert numeric index back to agent type"""
        return self.agent_types[index - 1]

    def find_optimal_path(self, messages: List[Any], iterations: int = 10) -> List[str]:
        for iteration in range(iterations):
            # Convert current edge weights to format suitable for Dijkstra
            edges = self.convert_to_edges()
            n = len(self.agent_types)
            
            # Try different starting points
            start_type = random.choice(self.agent_types)
            start_index = self.get_agent_index(start_type)
            
            # Build adjacency list
            adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(1, n + 1)}
            for src, dst, w in edges:
                src_idx = self.get_agent_index(src)
                dst_idx = self.get_agent_index(dst)
                adj[src_idx].append((dst_idx, w))
            
            # Run Dijkstra's algorithm
            shortest: Dict[int, float] = {}
            visited: Set[str] = set()
            min_heap = [(0, start_index, [start_type])]
            
            while min_heap:
                weight, current_idx, current_path = heapq.heappop(min_heap)
                current_type = self.get_agent_type(current_idx)
                
                if current_type in visited:
                    continue
                    
                visited.add(current_type)
                shortest[current_idx] = weight
                
                # If we have a valid path through all agent types
                if len(current_path) == len(self.agent_types):
                    # Evaluate the path
                    path_score = self.evaluate_path(messages, current_path)
                    
                    # Update best path if better
                    if path_score > self.best_score:
                        self.best_score = path_score
                        self.best_path = current_path
                        self.strengthen_path(current_path, messages)
                    break
                
                # Explore neighbors
                for next_idx, edge_weight in adj[current_idx]:
                    next_type = self.get_agent_type(next_idx)
                    if next_type not in visited and next_type not in current_path:
                        next_path = current_path + [next_type]
                        heapq.heappush(min_heap, 
                                     (weight + edge_weight, next_idx, next_path))
            
            # Dynamic edge weight adjustment
            if self.best_path:
                # Update weights based on path performance
                for i in range(len(self.best_path) - 1):
                    current = self.best_path[i]
                    next_node = self.best_path[i + 1]
                    
                    if current not in self.agent_state.edge_weights:
                        self.agent_state.edge_weights[current] = {}
                    
                    # Increase weight for successful transitions
                    current_weight = self.agent_state.edge_weights[current].get(next_node, 0.5)
                    new_weight = min(1.0, current_weight + self.best_score * 0.1)
                    self.agent_state.edge_weights[current][next_node] = new_weight
                    
                    # Decrease weights for unused edges
                    for target in self.agent_state.edge_weights[current]:
                        if target != next_node:
                            current_weight = self.agent_state.edge_weights[current][target]
                            new_weight = max(0.1, current_weight - 0.05)
                            self.agent_state.edge_weights[current][target] = new_weight
        
        return self.best_path
    
    def visualize_graph(self, title="Agent Network Visualization"):
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.agent_types:
            G.add_node(node)
        
        # Add edges with weights
        for source, edges in self.agent_state.edge_weights.items():
            for target, weight in edges.items():
                G.add_edge(source, target, weight=weight)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] * 2 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                             alpha=0.6, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        return plt

def initialize_edge_weights() -> Dict[str, Dict[str, float]]:
    return {
        "researcher": {"coder": 0.5, "critic": 0.3, "human": 0.2},
        "coder": {"critic": 0.6, "researcher": 0.2, "human": 0.2},
        "critic": {"human": 0.5, "coder": 0.3, "researcher": 0.2},
        "human": {"researcher": 0.4, "coder": 0.4, "critic": 0.2}
    }
