import pytest
import networkx as nx
import sys  # Provides access to Python's path handling
import os   # Provides functions for interacting with the operating system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs.conversation_graph import Message,ConversationNode,ConversationDAG,ConversationManager
from collections import defaultdict, deque
from typing import Dict, List, Set, Any , Callable
from sentence_transformers import CrossEncoder
from utils.debug_utils import DebugUtils

# Initialize DebugUtils
debug_utils = DebugUtils()
logger = debug_utils.get_logger(__name__)


class SlimeMoldGraph:
    def __init__(self, manager: ConversationManager):
        self.manager = manager
        self.adjacency_list = {}
        self.node_level_ranks = {}
        self.root_content = {}
        self.level_based_content = defaultdict(list)
        self.node_safety_flags = {}
        self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base', show_progress_bar=False)
        self.logger.info("SlimeMoldGraph initialized")

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def build_graph_structure(self):
        """Build the core graph structure excluding chief nodes"""
        for virtual_node_id, graph in self.manager.conversation_graphs.items():
            self.logger.debug(f"Processing virtual node: {virtual_node_id}")
            self.adjacency_list[virtual_node_id] = {}
            
            # BFS to build connections skipping chief nodes
            queue = deque([(graph.root, 0)])
            while queue:
                node, level = queue.popleft()
                
                if node.message.agent_type != 'chief':
                    self.logger.debug(f"Processing non-chief node at level {level}: {node.message.node_id}")

                    # Store level-based content
                    self.level_based_content[level].append(node.message.content)
                    
                    # Store node in adjacency list
                    if node.message.node_id not in self.adjacency_list[virtual_node_id]:
                        self.adjacency_list[virtual_node_id][node.message.node_id] = []
                    
                    # Connect to non-chief children
                    for child in node.children:
                        if child.message.agent_type != 'chief':
                            self.adjacency_list[virtual_node_id][node.message.node_id].append(child.message.node_id)
                
                # Add children to queue
                for child in node.children:
                    queue.append((child, level + 1))
            self.logger.info(f"Processed {processed_nodes} nodes for virtual node {virtual_node_id}")


    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def compute_node_ranks(self):
        """Compute ranks for nodes based on level-wise content similarity to the root node"""

        self.logger.debug("Starting node rank computation")
        
        for virtual_node_id in self.adjacency_list:
            self.logger.debug(f"Computing ranks for virtual node: {virtual_node_id}")
            queue = deque([(self.manager.conversation_graphs[virtual_node_id].root, 0)])
            visited = set()
            processed_nodes = 0
            while queue:
                node, level = queue.popleft()
                if node.message.node_id in visited or node.message.agent_type == 'chief':
                    continue
                    
                visited.add(node.message.node_id)
                processed_nodes += 1
                
                if node.message.agent_type != 'chief':
                    # Get parent content for comparison
                    parent_content = ""
                    if node.parent and node.parent.message.agent_type != 'chief':
                        parent_content = node.parent.message.content
                    
                    # Compare with level-based content
                    current_content = node.message.content
                    level_contents = self.level_based_content[level]
                    
                    self.logger.debug(f"Computing similarity scores for node {node.message.node_id} at level {level}")  
                    # Compute similarity scores
                    pairs = [(current_content, content) for content in level_contents]
                    scores = self.cross_encoder.predict(pairs)
                    
                    # Store rank
                    self.node_level_ranks[node.message.node_id] = {
                        'level': level,
                        'similarity_score': float(scores.mean()),
                        'parent_similarity': float(self.cross_encoder.predict([(current_content, parent_content)])[0])
                        if parent_content else 0.0
                    }
                
                for child in node.children:
                    queue.append((child, level + 1))
            
            self.logger.info(f"Processed {processed_nodes} nodes for rank computation in virtual node {virtual_node_id}")
    
    @DebugUtils.trace_function
    def check_safety(self):
        self.logger.debug("Starting safety checks")
        checked_nodes = 0
        """Apply safety checks to all nodes"""
        for virtual_node_id in self.adjacency_list:
            for node_id in self.adjacency_list[virtual_node_id]:
                node = self.find_node_by_id(virtual_node_id, node_id)
                if node:
                    # Implement safety check logic here
                    self.node_safety_flags[node_id] = True  # placeholder
                    self.logger.debug(f"Checking safety for node: {node_id}")
                    self.node_safety_flags[node_id] = True
                    checked_nodes += 1
    
    @DebugUtils.trace_function
    def find_node_by_id(self, virtual_node_id: str, node_id: str) -> ConversationNode:
        """Helper method to find a node by its ID"""
        graph = self.manager.conversation_graphs.get(virtual_node_id)
        self.logger.debug(f"Searching for node {node_id} in virtual node {virtual_node_id}")
        if not graph:
            self.logger.warning(f"Virtual node {virtual_node_id} not found")
            return None
            
        queue = deque([graph.root])
        while queue:
            node = queue.popleft()
            if node.message.node_id == node_id:
                self.logger.debug(f"Found node {node_id} after checking {nodes_checked} nodes")
                return node
            queue.extend(node.children)
        self.logger.warning(f"Node {node_id} not found after checking {nodes_checked} nodes")
        return None

@DebugUtils.trace_function
@DebugUtils.performance_monitor(threshold_ms=2000)
def load_virtual_nodes_with_descendants(graphml_path, manager: ConversationManager):
    logger.debug(f"Loading virtual nodes from: {graphml_path}")

    try:
        G = nx.read_graphml(graphml_path)
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Error loading GraphML file: {str(e)}")
        raise

    @DebugUtils.trace_function
    def bfs_collect_with_levels(start_node):
        logger.debug(f"Starting BFS collection from node: {start_node}")
        level_map = defaultdict(dict)
        node_metadata = {}
        visited = set()
        queue = deque([(start_node, 0, None)])  # (node, level, parent)
        
        while queue:
            current_node, current_level, parent = queue.popleft()
            
            if current_node not in visited:
                visited.add(current_node)
                
                # Store node in level map with parent info
                level_map[current_level][current_node] = {
                    'parent': parent,
                    'edge_type': G.get_edge_data(parent, current_node).get('edge_type', '') if parent else None
                }
                
                # Store node metadata
                node_data = G.nodes[current_node]
                node_metadata[current_node] = {
                    'content': node_data.get('content', ''),
                    'agent_type': node_data.get('agent_type', ''),
                    'timestamp': node_data.get('timestamp', ''),
                    'is_virtual': node_data.get('is_virtual', 'False')
                }
                
                # Add children to queue with next level
                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, current_level + 1, current_node))
        
        return {
            'level_map': level_map,
            'node_metadata': node_metadata
        }

    # Process virtual nodes
    virtual_nodes_processed = 0
    for node_id, node_data in G.nodes(data=True):
        # The boolean flag is stored as astring in the conversation graphml file.
        if node_data.get('is_virtual') == "True":
            logger.debug(f"Processing virtual node: {node_id}")
            # Collect level-aware structure
            hierarchy = bfs_collect_with_levels(node_id)
            
            # Create virtual graph with level awareness
            virtual_content = node_data.get('content', '')
            virtual_graph = ConversationDAG(virtual_content, node_id)
            
            # Set root node
            virtual_message = Message(
                node_id=node_id,
                content=virtual_content,
                agent_type="virtual",
                timestamp=""
            )
            virtual_message.is_virtual = True
            virtual_graph.root.message = virtual_message
            virtual_nodes_processed += 1
            # Build graph level by level
            for level in sorted(hierarchy['level_map'].keys())[1:]:  # Skip level 0 (root)
                level_nodes = hierarchy['level_map'][level]
                
                for node_id, node_info in level_nodes.items():
                    parent_id = node_info['parent']
                    node_metadata = hierarchy['node_metadata'][node_id]
                    
                    message = Message(
                        node_id=node_id,
                        content=node_metadata['content'],
                        agent_type=node_metadata['agent_type'],
                        timestamp=node_metadata['timestamp']
                    )
                    
                    virtual_graph.add_node(
                        message=message,
                        parent_id=parent_id,
                        edge_type=node_info['edge_type']
                    )
            
            manager.conversation_graphs[node_id] = virtual_graph
    logger.info(f"Processed {virtual_nodes_processed} virtual nodes")
    return manager

@DebugUtils.trace_function
def display_virtual_node_hierarchy(manager: ConversationManager):
    logger.debug("Starting hierarchy display")
    """
    Display the hierarchical structure of virtual nodes and their descendants
    with detailed information about relationships and metadata.
    """
    def get_truncated_content(content: str, max_length: int = 50) -> str:
        """Helper function to truncate long content"""
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content

    print("\n=== Virtual Node Hierarchy Analysis ===\n")
    
    if not manager.conversation_graphs:
        print("No virtual nodes found in the manager.")
        return
    
    for virtual_node_id, graph in manager.conversation_graphs.items():
        print(f"\n{'='*80}")
        print(f"Virtual Node: {virtual_node_id}")
        print(f"{'='*80}")
        
        # Create level-based structure for display
        levels = defaultdict(list)
        node_queue = deque([(graph.root, 0)])
        visited = set()
        
        while node_queue:
            current_node, level = node_queue.popleft()
            if current_node and current_node.message.node_id not in visited:
                visited.add(current_node.message.node_id)
                levels[level].append(current_node)
                
                for child in current_node.children:
                    node_queue.append((child, level + 1))
        
        # Display level by level
        for level, nodes in sorted(levels.items()):
            print(f"\nLevel {level}:")
            print("-" * 80)
            
            for node in nodes:
                msg = node.message
                indent = "  " * level
                
                # Node basic info
                print(f"\n{indent}Node ID: {msg.node_id}")
                print(f"\n{indent}Content: {get_truncated_content(msg.content)}")
                print(f"\n{indent}Agent Type: {msg.agent_type}")
                print(f"\n{indent}Timestamp: {msg.timestamp}")
                
                # Parent relationship
                if hasattr(node, 'parent') and node.parent:
                    print(f"\n{indent}Parent: {node.parent.message.node_id}")
                
                # Child relationships
                if node.children:
                    child_ids = [child.message.node_id for child in node.children]
                    print(f"\n{indent}Children: {child_ids}")
                
                print(f"{indent}{'-' * 40}")
        
        # Summary statistics
        total_nodes = len(visited)
        max_depth = len(levels) - 1
        nodes_per_level = {level: len(nodes) for level, nodes in levels.items()}
        
        print(f"\nSummary for Virtual Node {virtual_node_id}:")
        print(f"\nTotal Nodes: {total_nodes}")
        print(f"\nMaximum Depth: {max_depth}")
        print(f"\nNodes per Level: {dict(nodes_per_level)}")



# Usage example:


manager = ConversationManager()
manager = load_virtual_nodes_with_descendants('merged_conversation_graph.graphml', manager1)

# Usage
display_virtual_node_hierarchy(manager)
