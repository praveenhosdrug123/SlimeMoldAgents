# test_fixtures.py
import pytest
import networkx as nx
import sys  # Provides access to Python's path handling
import os   # Provides functions for interacting with the operating system
from pathlib import Path

# This complex line modifies Python's import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs.conversation_graph import ConversationDAG, Message,ConversationManager 
from src.slime_mold import load_virtual_nodes_with_descendants
import tempfile

@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    
    # Add virtual node
    G.add_node("virtual_1", 
               content="Virtual cluster of questions about Python",
               agent_type="virtual",
               is_virtual="True",
               timestamp="2024-01-01")
    
    # Add regular nodes
    G.add_node("q1", 
               content="How do I use Python lists?",
               agent_type="user",
               is_virtual="False",
               timestamp="2024-01-01")
    
    G.add_node("a1",
               content="Python lists are created using square brackets",
               agent_type="assistant",
               is_virtual="False",
               timestamp="2024-01-01")
    
    # Add edges
    G.add_edge("virtual_1", "q1", edge_type="contains", weight="0.5")
    G.add_edge("q1", "a1", edge_type="response", weight="0.5")
    
    return G

@pytest.fixture
def multi_virtual_graph(simple_graph):
    G = simple_graph.copy()
    G.add_node("virtual_2",
               content="Another virtual cluster",
               agent_type="virtual",
               is_virtual="True",
               timestamp="2024-01-01")
    G.add_node("q3",
               content="Another question",
               agent_type="user",
               is_virtual="False",
               timestamp="2024-01-01")
    G.add_edge("virtual_2", "q3", edge_type="contains", weight="0.5")
    return G

@pytest.fixture
def cyclic_graph(simple_graph):
    G = simple_graph.copy()
    G.add_edge("a1", "q1", edge_type="related", weight="0.5")
    return G

@pytest.fixture
def temp_graphml_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.graphml"
        yield path

@pytest.fixture
def conversation_manager():
    return ConversationManager()


def test_load_simple_virtual_node(simple_graph, temp_graphml_path, conversation_manager):
    nx.write_graphml(simple_graph, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    
    assert "virtual_1" in manager.conversation_graphs
    virtual_graph = manager.conversation_graphs["virtual_1"]
    
    # Test virtual node properties
    assert virtual_graph.root.message.is_virtual == True
    assert virtual_graph.root.message.agent_type == "virtual"
    
    # Test structure
    assert len(virtual_graph.nodes) == 3
    assert "q1" in virtual_graph.nodes
    assert virtual_graph.nodes["q1"].children[0].message.node_id == "a1"
    
    # Test edge properties
    assert virtual_graph.root.edge_types["q1"] == "contains"
    assert virtual_graph.root.edge_weight["q1"] == 0.5

def test_empty_graph(temp_graphml_path, conversation_manager):
    G = nx.DiGraph()
    nx.write_graphml(G, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    assert len(manager.conversation_graphs) == 0

def test_no_virtual_nodes(temp_graphml_path, conversation_manager):
    G = nx.DiGraph()
    G.add_node("q1", 
               content="Regular question",
               agent_type="user",
               is_virtual="False")
    nx.write_graphml(G, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    assert len(manager.conversation_graphs) == 0

def test_multiple_virtual_nodes(multi_virtual_graph, temp_graphml_path, conversation_manager):
    nx.write_graphml(multi_virtual_graph, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    assert len(manager.conversation_graphs) == 2
    assert "virtual_1" in manager.conversation_graphs
    assert "virtual_2" in manager.conversation_graphs
    
    # Test second virtual node structure
    virtual_2 = manager.conversation_graphs["virtual_2"]
    assert len(virtual_2.nodes) == 2
    assert "q3" in virtual_2.nodes

def test_cyclic_graph(cyclic_graph, temp_graphml_path, conversation_manager):
    nx.write_graphml(cyclic_graph, temp_graphml_path)
    
    # Add debug prints
    print("\nDebug cyclic graph structure:")
    print("Edges:", cyclic_graph.edges())
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    print("\nManager graphs:", manager.conversation_graphs.keys())
    
    assert "virtual_1" in manager.conversation_graphs
    virtual_graph = manager.conversation_graphs["virtual_1"]
    
    print("\nVirtual graph nodes:", virtual_graph.nodes.keys())
    
    # Should handle cycles properly
    assert len(virtual_graph.nodes) == 3
    
    # Test cycle is preserved
    q1_node = virtual_graph.nodes["q1"]
    a1_node = virtual_graph.nodes["a1"]
    
    print("\nq1_node children:", [child.message.node_id for child in q1_node.children])
    print("a1_node children:", [child.message.node_id for child in a1_node.children])
    print([child.message.node_id for child in a1_node.children])
    assert not any(child.message.node_id == "q1" for child in a1_node.children)

def test_node_attributes(simple_graph, temp_graphml_path, conversation_manager):
    nx.write_graphml(simple_graph, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    virtual_graph = manager.conversation_graphs["virtual_1"]
    
    # Test node attributes are preserved
    q1_node = virtual_graph.nodes["q1"]
    assert q1_node.message.content == "How do I use Python lists?"
    assert q1_node.message.agent_type == "user"
    assert q1_node.message.timestamp == "2024-01-01"

def test_edge_attributes(simple_graph, temp_graphml_path, conversation_manager):
    nx.write_graphml(simple_graph, temp_graphml_path)
    
    manager = load_virtual_nodes_with_descendants(temp_graphml_path, conversation_manager)
    virtual_graph = manager.conversation_graphs["virtual_1"]
    
    # Test edge attributes
    assert virtual_graph.root.edge_types["q1"] == "contains"
    assert virtual_graph.root.edge_weight["q1"] == 0.5
    
    q1_node = virtual_graph.nodes["q1"]
    assert q1_node.edge_types["a1"] == "response"
    assert q1_node.edge_weight["a1"] == 0.5

def verify_graph_integrity(manager: ConversationManager):
    """
    Verify the integrity of the loaded graphs
    """
    print("\n=== Graph Integrity Verification ===\n")
    
    for virtual_node_id, graph in manager.conversation_graphs.items():
        print(f"\nChecking graph for virtual node: {virtual_node_id}")
        
        # Check for cycles
        visited = set()
        cycle_found = False
        
        def check_cycles(node, path_set):
            nonlocal cycle_found
            if node.message.node_id in path_set:
                cycle_found = True
                return
            
            path_set.add(node.message.node_id)
            for child in node.children:
                check_cycles(child, path_set.copy())
        
        check_cycles(graph.root, set())
        print(f"Cycle check: {'Failed' if cycle_found else 'Passed'}")
        
        # Check parent-child consistency
        inconsistencies = []
        def check_consistency(node):
            for child in node.children:
                if hasattr(child, 'parent'):
                    if child.parent != node:
                        inconsistencies.append(
                            f"Parent-child mismatch: {node.message.node_id} -> {child.message.node_id}"
                        )
                check_consistency(child)
        
        check_consistency(graph.root)
        print(f"Parent-child consistency: {'Failed' if inconsistencies else 'Passed'}")
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
