import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import defaultdict
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


class Message:
    def __init__(self,node_id:str,content:str,agent_type:str,timestamp:str):
        self.node_id = node_id
        self.content = content
        self.agent_type = agent_type
        self.timestamp= timestamp
        self.content_embedding = model.encode(content)
        self.is_virtual = False


class ConversationNode:
    def __init__(self, message: Message):
        self.message = message
        self.children: List[ConversationNode] = []
        self.edge_types: Dict[str, str] = {}  # child_id -> edge_type
        self.edge_weight: Dict[str,float] ={}
        
    def add_child(self, child_node: 'ConversationNode', edge_type: str):
        self.children.append(child_node)
        self.edge_types[child_node.message.node_id] = edge_type
        self.edge_weight[child_node.message.node_id] = 0.5

class ConversationDAG:
    def __init__(self, root_query: str, root_id: str):
        self.root = ConversationNode(Message(
            node_id=root_id,
            content=root_query,
            agent_type="query",
            timestamp=""
        ))
        self.nodes: Dict[str, ConversationNode] = {root_id: self.root}
        
    def add_node(self, message: Message, parent_id: str, edge_type: str):
        if message.node_id not in self.nodes:
            node = ConversationNode(message)
            self.nodes[message.node_id] = node
            
        if parent_id in self.nodes:
            parent = self.nodes[parent_id]
            parent.add_child(self.nodes[message.node_id], edge_type)
    
    def has_node(self, node_id):
        """Check if a node with given ID exists in the graph"""
        return node_id in self.nodes

    def get_all_paths(self) -> List[List[ConversationNode]]:
        """Returns all possible paths from root to leaf nodes."""
        paths = []
        
        def dfs(node: ConversationNode, current_path: List[ConversationNode]):
            if not node.children:  # Leaf node
                paths.append(current_path[:])
                return
            
            for child in node.children:
                current_path.append(child)
                dfs(child, current_path)
                current_path.pop()
        
        dfs(self.root, [self.root])
        return paths
    
    def get_branch(self, node_id):
        """Returns the path (node IDs) from root to given node"""
        if node_id not in self.nodes:
            return []
            
        path = []
        current_id = node_id
        
        # Build path from node to root
        while current_id:
            path.append(current_id)
            current_id = None
            # Find parent
            for potential_parent_id, potential_parent in self.nodes.items():
                if any(child.message.node_id == path[-1] for child in potential_parent.children):
                    current_id = potential_parent_id
                    break
                    
        # Reverse to get root-to-node path
        path.reverse()
        return path

    def get_leaf_nodes(self) -> List[ConversationNode]:
        """Returns all leaf nodes (nodes with no children)."""
        # This function stays the same as it doesn't depend on parent/root relationships
        return [node for node in self.nodes.values() if not node.children]

    def get_siblings(self, node_id: str) -> List[ConversationNode]:
        """Returns siblings of the specified node (nodes sharing the same parent)."""
        """Returns list of sibling nodes for given node_id"""
        # Find parent
        parent = None
        parent_node = None
        for potential_parent in self.nodes.values():
            if any(child.message.node_id == node_id for child in potential_parent.children):
                parent_node = potential_parent
                break
        
        if not parent_node:
            return []
            
        # Return all children except the node itself
        return [child for child in parent_node.children 
                if child.message.node_id != node_id]

class UnionFind:
    def __init__(self, root_list):
        self.parent = {}  # renamed from par for clarity
        self.rank = {}
        
        for root in root_list:
            self.parent[root.node_id] = root
            self.rank[root.node_id] = 0
    
    def find(self, node):
        # Finds the root of node
        if node.node_id != self.parent[node.node_id].node_id:
            self.parent[node.node_id] = self.find(self.parent[node.node_id])
        return self.parent[node.node_id]

    def union(self, node1, node2):
        parent1, parent2 = self.find(node1), self.find(node2)
        similarity = float(model.similarity(parent1.content_embedding, parent2.content_embedding))
        print(f"Similarity between {node1.node_id} and {node2.node_id}: {similarity}")
        if similarity < 0.92:           
            return False

        # Use node_ids for rank comparison
        if self.rank[parent1.node_id] > self.rank[parent2.node_id]:
            self.parent[parent2.node_id] = parent1
        elif self.rank[parent1.node_id] < self.rank[parent2.node_id]:
            self.parent[parent1.node_id] = parent2
        else:
            self.parent[parent1.node_id] = parent2
            self.rank[parent2.node_id] += 1
            
        return True

class ConversationManager:
    def __init__(self):
        #print("Stack trace at ConversationManager initialization:")
        #traceback.print_stack()
        self.conversation_graphs: Dict[str, ConversationDAG] = {}
        
    def create_from_networkx(self, nx_graph: nx.DiGraph):
        # First pass: Create all conversations
        for node, data in nx_graph.nodes(data=True):
            if data.get('type') == 'initial_query':  # Changed from agent_type to type
                graph = ConversationDAG(data.get('content', ''), node)
                self.conversation_graphs[node] = graph
        
        # Second pass: Build conversation trees
        for source, target, edge_data in nx_graph.edges(data=True):
            edge_type = edge_data.get('type', '')  # Changed from edge_type to type
            
            # Find which graph this edge belongs to
            for graph in self.conversation_graphs.values():
                if graph.has_node(source):
                    # Get target node data
                    target_data = nx_graph.nodes[target]
                    message = Message(
                        node_id=target,
                        content=target_data.get('content', ''),
                        agent_type=target_data.get('agent', ''),  # Changed to match GraphML
                        timestamp=target_data.get('timestamp', '')
                    )
                    graph.add_node(message, source, edge_type)
                    break

    def get_conversation_path(self, root_id: str) -> List[tuple]:
        """Get the sequence of messages in the conversation"""
        if root_id not in self.conversation_graphs:
            return []
            
        path = []
        def dfs(node: ConversationNode,parent_weight: float = 0.0):

            path.append((
                node.message.agent_type,
                node.message.content,
                node.message.timestamp
            ))
            # Sort children by timestamp if available
            sorted_children = sorted(node.children, key=lambda x: x.message.timestamp)
            for child in sorted_children:
                child_edge_weight = node.edge_weight[child.message.node_id]
                dfs(child,parent_weight+child_edge_weight)
                
        dfs(self.conversation_graphs[root_id].root)
        return path

    def merge_similar_conversations(self):
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        print("\n--- Starting merge process at {current_time}---")
        
        # Get unmerged nodes
        unmerged_nodes = []
        for conv_id, graph in self.conversation_graphs.items():
            if conv_id.startswith("virtual"):
                continue
            is_merged = any(
                graph.root.message.node_id in vgraph.nodes
                for vid, vgraph in self.conversation_graphs.items()
                if vid.startswith("virtual")
            )
            if not is_merged:
                unmerged_nodes.append(graph.root.message)
        
        print(f"Found unmerged nodes: {[node.node_id for node in unmerged_nodes]}")
        
        if not unmerged_nodes:
            print("No unmerged nodes found")
            return

        # Check existing virtual nodes
        virtual_nodes = [vid for vid in self.conversation_graphs.keys() if vid.startswith("virtual")]
        print(f"Found existing virtual nodes: {virtual_nodes}")

        # First, try to merge with existing virtual nodes
        for node in unmerged_nodes[:]:  # Create a copy to iterate
            for vid, vgraph in self.conversation_graphs.items():
                if not vid.startswith("virtual"):
                    continue
                
                # Get the root messages of conversations in this virtual cluster
                cluster_roots = [n.message for n in vgraph.root.children]
                for existing_root in cluster_roots:
                    similarity = float(model.similarity(node.content_embedding, existing_root.content_embedding))
                    print(f"Checking {node.node_id} with virtual cluster root {existing_root.node_id}: {similarity}")
                    
                    if similarity >= 0.92:
                        # Add to existing virtual cluster
                        print(f"Adding {node.node_id} to existing virtual cluster {vid}")
                        original_graph = self.conversation_graphs[node.node_id]
                        vgraph.nodes[node.node_id] = original_graph.root
                        vgraph.root.children.append(original_graph.root)
                        unmerged_nodes.remove(node)
                        break
                if node not in unmerged_nodes:  # If node was merged, stop checking other virtual clusters
                    break

        print(f"Nodes still unmerged after virtual check: {[node.node_id for node in unmerged_nodes]}")

        # If there are still unmerged nodes, create new clusters
        if unmerged_nodes:
            print("Creating new clusters for remaining unmerged nodes")
            uf = UnionFind(unmerged_nodes)
            
            # Compare all pairs of remaining unmerged nodes
            for i, node1 in enumerate(unmerged_nodes):
                for node2 in unmerged_nodes[i+1:]:
                    similarity = float(model.similarity(node1.content_embedding, node2.content_embedding))
                    print(f"Similarity between {node1.node_id} and {node2.node_id}: {similarity}")
                    if similarity >= 0.92:
                        uf.union(node1, node2)

            # Group nodes by their root in the UnionFind structure
            clusters = {}
            for node in unmerged_nodes:
                root = uf.find(node)
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(node)
            
            # Create virtual nodes for each new cluster with more than one conversation
            for parent_id, cluster_nodes in clusters.items():
                if len(cluster_nodes) > 1:
                    virtual_content = " | ".join(node.content for node in cluster_nodes)
                    virtual_id = f"virtual_{'_'.join(node.node_id for node in cluster_nodes)}"
                    
                    virtual_message = Message(
                        node_id=virtual_id,
                        content=virtual_content,
                        agent_type="virtual",
                        timestamp=""
                    )
                    virtual_message.is_virtual = True
                    
                    virtual_message.content_embedding = np.mean([node.content_embedding for node in cluster_nodes], axis=0)
                    
                    virtual_graph = ConversationDAG(virtual_content, virtual_id)
                    virtual_graph.root.message = virtual_message
                    
                    # Calculate weights based on tree sizes
                    all_nodes = set()
                    for node in cluster_nodes:
                        graph_nodes = self.conversation_graphs[node.node_id].nodes
                        all_nodes.update(graph_nodes)
                    total_nodes = len(all_nodes)
                    
                    for node in cluster_nodes:
                        weight = len(self.conversation_graphs[node.node_id].nodes) / total_nodes
                        virtual_graph.add_node(node, virtual_id, "merged")
                        virtual_graph.root.edge_weight[node.node_id] = weight
                    
                    self.conversation_graphs[virtual_id] = virtual_graph

        print("--- Merge process complete ---\n")
    
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert the conversation structure to a NetworkX DiGraph"""
        G = nx.DiGraph()
        
        print("\nConverting to NetworkX graph...")
        print(f"Processing {len(self.conversation_graphs)} conversation graphs")
        
        # Add all nodes and their edges
        for graph_id, graph in self.conversation_graphs.items():
            print(f"\nProcessing graph {graph_id} with {len(graph.nodes)} nodes")
            for node_id, node in graph.nodes.items():
                print(f"Adding node {node_id}")
                # Convert numpy array to list for GraphML compatibility
                content_embedding = node.message.content_embedding.tolist() if hasattr(node.message, 'content_embedding') else []
                # Add node with attributes from Message class
                node_attrs = {
                    'node_id':str(node.message.node_id),
                    'content': str(node.message.content),
                    'agent_type': str(node.message.agent_type),
                    'timestamp': str(node.message.timestamp),
                    'content_embedding': str(content_embedding),
                    'is_virtual': str(node.message.is_virtual)
                }
                
                # Add node with all attributes
                G.add_node(node_id, **node_attrs)
                
                # Add edges from this node to its children
                for child in node.children:
                    print(f"Adding edge {node_id} -> {child.message.node_id}")
                    edge_attrs = {
                        'edge_type': str(node.edge_types.get(child.message.node_id, "")),
                        'weight': str(node.edge_weight.get(child.message.node_id, 0.5))
                    }
                    
                    G.add_edge(node_id, child.message.node_id, **edge_attrs)
        
        print(f"\nCreated graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G

def print_conversation_flows(nx_graph: nx.DiGraph):
    # Initialize and create conversation graphs
    manager = ConversationManager()
    manager.create_from_networkx(nx_graph)
    
    # Print each conversation flow
    for root_id in manager.conversation_graphs:
        print("\n" + "="*80)
        print(f"Conversation Flow for Query: {manager.conversation_graphs[root_id].root.message.content}")
        print("="*80)
        
        path = manager.get_conversation_path(root_id)
        for agent, content, timestamp in path:
            if agent:
                print(f"\n{agent.upper()}: {content}")
            else:
                print(f"\nQUERY: {content}")


# Usage:
def main():

    graph = nx.read_graphml("test_conversation_graph.graphml")
    print("\nOriginal graph:")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print("Sample node data:", next(iter(graph.nodes(data=True))))
    print("Sample edge data:", next(iter(graph.edges(data=True))))

    manager = ConversationManager()
    manager.create_from_networkx(graph)
    manager.merge_similar_conversations()
    merged_graph = manager.to_networkx()

    print("\nMerged graph:")
    print(f"Nodes: {len(merged_graph.nodes)}")
    print(f"Edges: {len(merged_graph.edges)}")

    nx.write_graphml(merged_graph, "merged_conversation_graph.graphml")
if __name__ == "__main__":
    main()
