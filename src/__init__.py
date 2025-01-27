# src/__init__.py

from .multiagent import MultiAgentSystem, WorkflowManager, AgentState
from .safetyframework import SafetyFramework, SafetyViolation
from .slime_mold import SlimeMoldGraph, load_virtual_nodes_with_descendants, display_virtual_node_hierarchy
from utils.debug_utils import DebugUtils

# Initialize debug utils
debug_utils = DebugUtils()
logger = debug_utils.get_logger(__name__)

# Export commonly used classes and functions
__all__ = [
    # MultiAgent components
    'MultiAgentSystem',
    'WorkflowManager',
    'AgentState',
    
    # Safety components
    'SafetyFramework',
    'SafetyViolation',
    
    # SlimeMold components
    'SlimeMoldGraph',
    'load_virtual_nodes_with_descendants',
    'display_virtual_node_hierarchy'
]

# Version information
__version__ = '1.0.0'

logger.debug(f"Source package initialized. Version: {__version__}")
