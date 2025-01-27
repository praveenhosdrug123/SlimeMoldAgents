# utils/__init__.py

from .debug_utils import DebugUtils
from .data_manager import DataManager

# Initialize global instances
debug_utils = DebugUtils()
data_manager = DataManager()

# Export commonly used functions and decorators for easier access
get_logger = debug_utils.get_logger
trace_function = debug_utils.trace_function
log_state = debug_utils.log_state
performance_monitor = debug_utils.performance_monitor
setup_logging = debug_utils.setup_logging

# Data manager common functions
store_interaction = data_manager.store_interaction
save_graph = data_manager.save_graph
load_graph = data_manager.load_graph
query_conversation_tree = data_manager.query_conversation_tree
visualize_graph = data_manager.visualize_graph

# Define what should be available when using "from utils import *"
__all__ = [
    # Debug utilities
    'DebugUtils',
    'debug_utils',
    'get_logger',
    'trace_function',
    'log_state',
    'performance_monitor',
    'setup_logging',
    
    # Data manager
    'DataManager',
    'data_manager',
    'store_interaction',
    'save_graph',
    'load_graph',
    'query_conversation_tree',
    'visualize_graph'
]

# Version information
__version__ = '1.0.0'

# Initialize logging for the utils package
logger = debug_utils.get_logger(__name__)
logger.debug(f"Utils package initialized. Version: {__version__}")
