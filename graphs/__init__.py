# graphs/__init__.py

from typing import Dict, List, Optional, Set
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

# Core classes
from .conversation_graph import (
    Message,
    ConversationNode,
    ConversationDAG,
    UnionFind,
    ConversationManager
)

# Utility functions
def create_conversation_graph(root_query: str, root_id: str) -> ConversationDAG:
    """Helper function to create a new conversation graph"""
    return ConversationDAG(root_query, root_id)

def load_graph_from_file(filepath: str) -> nx.DiGraph:
    """Load a conversation graph from a GraphML file"""
    return nx.read_graphml(filepath)

def save_graph_to_file(graph: nx.DiGraph, filepath: str) -> None:
    """Save a conversation graph to a GraphML file"""
    nx.write_graphml(graph, filepath)

# Export specific components
__all__ = [
    'Message',
    'ConversationNode',
    'ConversationDAG',
    'UnionFind',
    'ConversationManager',
    'create_conversation_graph',
    'load_graph_from_file',
    'save_graph_to_file',
]

# Version info
__version__ = '0.1.0'
