import networkx as nx
from datetime import datetime
import uuid
import matplotlib.pyplot as plt
import numpy as np
from utils.debug_utils import DebugUtils

class DataManager:
    def __init__(self):
        # Initialize debug utils and logger
        self.debug = DebugUtils()
        self.logger = self.debug.get_logger(__name__)
        self.logger.info("Initializing DataManager")
        
        self.graph = nx.MultiDiGraph()
        self.current_session = str(uuid.uuid4())
        self.logger.debug(f"Created new session: {self.current_session}")

    @DebugUtils.trace_function
    def store_interaction(self, interaction_data: dict):
        """Store interaction data in the graph."""
        try:
            self.logger.debug(f"Processing interaction data: {interaction_data}")
            
            # Create unique IDs for nodes
            query_id = f"query_{self.current_session}"
            message_id = f"message_{uuid.uuid4()}"
            
            self.logger.debug(f"Generated IDs - Query: {query_id}, Message: {message_id}")
            
            # Add initial query node if it doesn't exist
            if not self.graph.has_node(query_id):
                self.logger.debug(f"Adding new query node: {query_id}")
                self.graph.add_node(query_id, 
                    content=interaction_data.get("query", ""),
                    timestamp=interaction_data.get("timestamp", datetime.now().isoformat()),
                    type='initial_query'
                )
            
            # Add interaction node
            self.logger.debug(f"Adding interaction node: {message_id}")
            self.graph.add_node(message_id,
                content=interaction_data.get("content", ""),
                agent=interaction_data.get("agent", "unknown"),
                timestamp=interaction_data.get("timestamp", datetime.now().isoformat()),
                path_step=len(self.graph.nodes())
            )
            
            # Add edges
            self._add_edges(query_id, message_id)
            
        except Exception as e:
            self.logger.error(f"Error in store_interaction: {str(e)}")
            raise

    @DebugUtils.trace_function
    def _add_edges(self, query_id: str, message_id: str):
        """Helper method to add edges."""
        try:
            # Add relationship to initial query
            self.graph.add_edge(query_id, message_id, 
                type='GENERATED',
                timestamp=datetime.now().isoformat()
            )
            
            # Add relationship to previous message if exists
            nodes = list(self.graph.nodes())
            if len(nodes) > 2:
                prev_message_id = nodes[-2]
                self.logger.debug(f"Adding edge to previous message: {prev_message_id} -> {message_id}")
                self.graph.add_edge(prev_message_id, message_id,
                    type='LEADS_TO',
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            self.logger.error(f"Error in _add_edges: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=1000)
    def save_graph(self, filename="conversation_graph.graphml"):
        """Save the graph to a file."""
        try:
            self.logger.info(f"Saving graph to {filename}")
            nx.write_graphml(self.graph, filename)
            self.logger.debug(f"Graph saved successfully. Node count: {len(self.graph.nodes())}")
        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}")
            raise

    @DebugUtils.trace_function
    def load_graph(self, filename="conversation_graph.graphml"):
        """Load the graph from a file."""
        try:
            self.logger.info(f"Loading graph from {filename}")
            self.graph = nx.read_graphml(filename)
            self.logger.debug(f"Graph loaded successfully. Node count: {len(self.graph.nodes())}")
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            raise

    @DebugUtils.trace_function
    def query_conversation_tree(self, initial_query):
        """Query the conversation tree."""
        try:
            self.logger.debug(f"Querying conversation tree for: {initial_query}")
            result = nx.bfs_tree(self.graph, f"query_{self.current_session}")
            self.logger.debug(f"Query result node count: {len(result.nodes())}")
            return result
        except Exception as e:
            self.logger.error(f"Error querying conversation tree: {str(e)}")
            raise

    @DebugUtils.trace_function
    @DebugUtils.performance_monitor(threshold_ms=2000)
    def visualize_graph(self):
        """Enhanced NetworkX visualization with debugging."""
        try:
            self.logger.info("Starting graph visualization")
            plt.figure(figsize=(24, 14))
            
            # Layout calculation with error handling
            try:
                self.logger.debug("Attempting to use graphviz layout")
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot', args='-Grankdir=TB')
            except Exception as layout_error:
                self.logger.warning(f"Graphviz layout failed, using spring layout: {layout_error}")
                pos = nx.spring_layout(self.graph, k=2, iterations=100, scale=2)
            
            self._draw_graph_elements(pos)
            
            self.logger.info("Graph visualization completed")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            raise

    def _draw_graph_elements(self, pos):
        """Helper method for drawing graph elements."""
        try:
            self.logger.debug("Drawing graph elements")
            
            # Node separation
            query_nodes = [node for node in self.graph.nodes() if 'query' in node]
            chief_nodes = [node for node in self.graph.nodes() if self.graph.nodes[node].get('agent') == 'chief']
            message_nodes = [node for node in self.graph.nodes() if 'message' in node 
                           and self.graph.nodes[node].get('agent') != 'chief']
            
            self.logger.debug(f"Node counts - Query: {len(query_nodes)}, Chief: {len(chief_nodes)}, Message: {len(message_nodes)}")
            
            # Draw elements
            self._draw_nodes_and_edges(pos, query_nodes, chief_nodes, message_nodes)
            self._draw_labels_and_legend(pos)
            
        except Exception as e:
            self.logger.error(f"Error drawing graph elements: {str(e)}")
            raise

    def _draw_nodes_and_edges(self, pos, query_nodes, chief_nodes, message_nodes):
        """Draw nodes and edges with debugging."""
        try:
            self.logger.debug("Drawing nodes and edges")
            # [Rest of the existing visualization code remains the same]
            
        except Exception as e:
            self.logger.error(f"Error drawing nodes and edges: {str(e)}")
            raise

    def _draw_labels_and_legend(self, pos):
        """Draw labels and legend with debugging."""
        try:
            self.logger.debug("Drawing labels and legend")
            # [Rest of the existing visualization code remains the same]
            
        except Exception as e:
            self.logger.error(f"Error drawing labels and legend: {str(e)}")
            raise
    
def generate_test_data():
    data_manager = DataManager()
    
    # Query 1: Sorting Algorithm (Quick Sort)
    query1_flow = {
    "query": "Implement a quicksort algorithm in Python",
    "flow": [
        {
            "agent": "chief",
            "content": "This requires algorithm understanding first. Routing to researcher for theoretical background.",
            "route_to": "researcher"
        },
        {
            "agent": "researcher",
            "content": "QuickSort is a divide-and-conquer algorithm that works by selecting a 'pivot' element and partitioning the array around it. Key characteristics:\n- Average time complexity: O(n log n)\n- In-place sorting\n- Unstable sort\n- Efficient for large datasets",
            "feedback_to": "chief"
        },
        {
            "agent": "chief",
            "content": "Good theoretical foundation established. Routing to coder for implementation.",
            "route_to": "coder"
        },
        {
            "agent": "coder",
            "content": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```",
            "feedback_to": "chief"
        },
        {
            "agent": "chief",
            "content": "Implementation provided. Routing to critic for code review and quality assessment.",
            "route_to": "critic"
        },
        {
            "agent": "critic",
            "content": "Code Review:\n1. Implementation is clear but not in-place\n2. Pivot selection could be improved\n3. No handling for edge cases\n4. Suggest adding type hints and documentation",
            "feedback_to": "chief"
        },
        {
            "agent": "chief",
            "content": "Review indicates several improvements needed. Should we route back to coder for improvements or is this sufficient for the current requirements?",
            "route_to": "human"
        }
    ]
            }

        # Query 2: Merge Sort
    query2_flow = {
        "query": "Implement a merge sort algorithm in Python",
        "flow": [
            {
                "agent": "chief",
                "content": "Starting with theoretical background. Routing to researcher for algorithm explanation.",
                "route_to": "researcher"
            },
            {
                "agent": "researcher",
                "content": "MergeSort is a divide-and-conquer algorithm that recursively splits the array and merges sorted subarrays. Key characteristics:\n- Guaranteed O(n log n) complexity\n- Stable sort\n- Requires extra space\n- Good for linked lists",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Algorithm characteristics clear. Routing to coder for implementation.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\ndef mergesort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = mergesort(arr[:mid])\n    right = mergesort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Implementation complete. Routing to critic for code review.",
                "route_to": "critic"
            },
            {
                "agent": "critic",
                "content": "Code Review:\n1. Clean implementation but space inefficient\n2. Good stability maintenance\n3. Consider in-place implementation\n4. Add input validation",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Review suggests optimization opportunities. Should we proceed with improvements or is this implementation sufficient?",
                "route_to": "human"
            }
        ]
                }       


        # Query 3: Database Connection
    query3_flow = {
        "query": "Create a PostgreSQL database connection class with connection pooling",
        "flow": [
            {
                "agent": "chief",
                "content": "Database connection requires best practices research first. Routing to researcher.",
                "route_to": "researcher"
            },
            {
                "agent": "researcher",
                "content": "For PostgreSQL connection pooling, we should use psycopg2 with connection pooling libraries like psycopg2-pool or SQLAlchemy. Key considerations:\n1. Connection lifecycle\n2. Pool size management\n3. Error handling\n4. Resource cleanup",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Research complete. Routing to coder for initial implementation.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\nfrom psycopg2 import pool\n\nclass DatabasePool:\n    def __init__(self, minconn=1, maxconn=10, **kwargs):\n        self.pool = pool.SimpleConnectionPool(\n            minconn, maxconn,\n            **kwargs\n        )\n    \n    def get_connection(self):\n        return self.pool.getconn()\n    \n    def return_connection(self, conn):\n        self.pool.putconn(conn)\n    \n    def close_pool(self):\n        self.pool.closeall()\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Basic implementation provided. Routing to critic for review.",
                "route_to": "critic"
            },
            {
                "agent": "critic",
                "content": "Code Review:\n1. Need context manager implementation\n2. Add error handling\n3. Consider connection timeout\n4. Add connection validation\n5. Missing documentation",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Critical improvements needed. Routing back to coder for updates.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\n# Updated implementation with context manager\nclass DatabasePool:\n    def __init__(self, minconn=1, maxconn=10, **kwargs):\n        self.pool = pool.SimpleConnectionPool(\n            minconn, maxconn,\n            **kwargs\n        )\n    \n    def __enter__(self):\n        self.conn = self.pool.getconn()\n        return self.conn\n    \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.pool.putconn(self.conn)\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Updated implementation addresses context management. Should we proceed with additional improvements?",
                "route_to": "human"
            }
        ]
                }

        # Query 4: API Integration
    query4_flow = {
        "query": "Create a REST API client for OpenWeatherMap",
        "flow": [
            {
                "agent": "chief",
                "content": "API integration requires understanding requirements first. Routing to researcher.",
                "route_to": "researcher"
            },
            {
                "agent": "researcher",
                "content": "OpenWeatherMap API requires:\n1. API key authentication\n2. Rate limiting handling\n3. JSON response parsing\n4. Error handling for different HTTP status codes",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Requirements gathered. Routing to coder for implementation.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\nimport requests\n\nclass WeatherClient:\n    def __init__(self, api_key):\n        self.api_key = api_key\n        self.base_url = 'https://api.openweathermap.org/data/2.5'\n    \n    def get_weather(self, city):\n        response = requests.get(\n            f'{self.base_url}/weather',\n            params={\n                'q': city,\n                'appid': self.api_key\n            }\n        )\n        response.raise_for_status()\n        return response.json()\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Basic implementation complete. Routing to critic for review.",
                "route_to": "critic"
            },
            {
                "agent": "critic",
                "content": "Code Review:\n1. Add retry mechanism\n2. Implement rate limiting\n3. Add response validation\n4. Consider adding caching\n5. Need better error handling",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Several critical improvements suggested. Should we implement these enhancements?",
                "route_to": "human"
            }
        ]
                }

        # Query 5: Data Processing
    query5_flow = {
        "query": "Create a data processing pipeline for CSV files using pandas",
        "flow": [
            {
                "agent": "chief",
                "content": "Data processing pipeline needs requirements analysis. Routing to researcher.",
                "route_to": "researcher"
            },
            {
                "agent": "researcher",
                "content": "Data processing pipeline should handle:\n1. Data validation\n2. Cleaning\n3. Transformation\n4. Error logging\n5. Large file processing",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Requirements clear. Routing to coder for initial implementation.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\nimport pandas as pd\nfrom typing import List, Dict\n\nclass DataPipeline:\n    def __init__(self, filename: str):\n        self.df = pd.read_csv(filename)\n    \n    def clean_data(self) -> None:\n        self.df.dropna(inplace=True)\n        self.df.drop_duplicates(inplace=True)\n    \n    def transform_data(self, operations: List[Dict]) -> None:\n        for op in operations:\n            if op['type'] == 'rename':\n                self.df.rename(columns=op['mapping'], inplace=True)\n            elif op['type'] == 'filter':\n                self.df = self.df.query(op['condition'])\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Initial implementation complete. Routing to critic for review.",
                "route_to": "critic"
            },
            {
                "agent": "critic",
                "content": "Code Review:\n1. Add data validation\n2. Implement chunking for large files\n3. Add progress tracking\n4. Include data type validation\n5. Add logging mechanism",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Performance and validation improvements needed. Routing back to coder.",
                "route_to": "coder"
            },
            {
                "agent": "coder",
                "content": "```python\n# Updated with chunking and validation\nclass DataPipeline:\n    def __init__(self, filename: str, chunk_size: int = 10000):\n        self.filename = filename\n        self.chunk_size = chunk_size\n    \n    def process_in_chunks(self):\n        for chunk in pd.read_csv(self.filename, chunksize=self.chunk_size):\n            yield self.process_chunk(chunk)\n```",
                "feedback_to": "chief"
            },
            {
                "agent": "chief",
                "content": "Chunking implementation added. Should we address the remaining improvement suggestions?",
                "route_to": "human"
            }
        ]
                }
    query6_flow = {
        "query": "Could you help me understand and write a QuickSort implementation in Python? Please explain the algorithm as well.",
            "flow": [
            {
             "agent": "chief",
            "content": "This is a comprehensive request involving both theoretical understanding and practical implementation. Let's start with a thorough theoretical foundation from our researcher before moving to implementation. Routing to researcher.",
            "route_to": "researcher"
            },
            {
        "agent": "researcher",
        "content": "QuickSort is one of the most efficient and widely used sorting algorithms. Let me break down its key aspects:\n\nTheoretical Foundation:\n1. Algorithm Paradigm: Divide-and-conquer\n2. Time Complexity:\n - Average case: O(n log n)\n - Worst case: O(n²) when poorly pivoted\n - Best case: O(n log n)\n3. Space Complexity: O(log n) due to recursion stack\n\nKey Characteristics:\n- In-place sorting capability\n- Unstable sort (doesn't preserve order of equal elements)\n- Excellent cache performance\n- Highly practical for large datasets\n\nCore Concept:\n1. Select a 'pivot' element\n2. Partition elements around pivot (smaller left, larger right)\n3. Recursively apply to sub-arrays\n4. Combine results\n\nCommon Optimization Techniques:\n- Median-of-three pivot selection\n- Switching to insertion sort for small subarrays\n- Tail recursion elimination",
        "feedback_to": "chief"
             },
            {
        "agent": "chief",
        "content": "Excellent theoretical background provided. Now we need a practical implementation incorporating these concepts. Routing to coder for a robust implementation.",
        "route_to": "coder"
             },
            {
        "agent": "coder",
        "content": "python\nfrom typing import List, Optional\n\ndef quicksort(arr: List[int], left: Optional[int] = None, right: Optional[int] = None) -> List[int]:\n    \"\"\"Implementation of QuickSort algorithm with in-place partitioning.\n    \n    Args:\n        arr: List of integers to be sorted\n        left: Starting index of the partition (None for full array)\n        right: Ending index of the partition (None for full array)\n    \n    Returns:\n        Sorted list of integers\n    \"\"\"\n    # Initialize left and right for first call\n    if left is None:\n        left = 0\n    if right is None:\n        right = len(arr) - 1\n    \n    def partition(low: int, high: int) -> int:\n        # Use median-of-three for pivot selection\n        mid = (low + high) // 2\n        pivot_candidates = [\n            (arr[low], low),\n            (arr[mid], mid),\n            (arr[high], high)\n        ]\n        _, pivot_idx = sorted(pivot_candidates, key=lambda x: x[0])[1]\n        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]\n        pivot = arr[high]\n        \n        i = low - 1\n        for j in range(low, high):\n            if arr[j] <= pivot:\n                i += 1\n                arr[i], arr[j] = arr[j], arr[i]\n        \n        arr[i + 1], arr[high] = arr[high], arr[i + 1]\n        return i + 1\n    \n    def _quicksort(low: int, high: int):\n        if low < high:\n            # Use insertion sort for small arrays\n            if high - low < 10:\n                for i in range(low + 1, high + 1):\n                    key = arr[i]\n                    j = i - 1\n                    while j >= low and arr[j] > key:\n                        arr[j + 1] = arr[j]\n                        j -= 1\n                    arr[j + 1] = key\n                return\n            \n            pi = partition(low, high)\n            _quicksort(low, pi - 1)\n            _quicksort(pi + 1, high)\n    \n    _quicksort(left, right)\n    return arr\n",
        "feedback_to": "chief"
            },
            {
        "agent": "chief",
        "content": "Implementation looks comprehensive. Let's have our critic review the code for quality, performance, and potential improvements. Routing to critic.",
        "route_to": "critic"
            },
            {
        "agent": "critic",
        "content": "Detailed Code Review:\n\nStrengths:\n1. Well-documented with type hints and docstrings\n2. Implements several optimizations:\n - Median-of-three pivot selection\n - Insertion sort for small subarrays\n - In-place partitioning\n3. Proper handling of edge cases\n4. Good variable naming and code structure\n\nPotential Improvements:\n1. Error Handling:\n - Could add input validation for non-numeric elements\n - Consider handling empty or None input\n\n2. Performance Optimizations:\n - Could implement tail recursion elimination\n - Might benefit from parallel processing for large arrays\n\n3. Flexibility Enhancements:\n - Consider adding comparison key parameter\n - Could make it generic for different data types\n\n4. Testing Considerations:\n - Add property-based testing\n - Include edge case tests\n - Benchmark against standard library sort\n\nOverall: Strong implementation that balances readability with performance.",
        "feedback_to": "chief"
            },
            {
        "agent": "chief",
        "content": "The implementation meets high quality standards but has room for some enhancements. Should we proceed with improvements based on the critic's suggestions, or is this implementation sufficient for your needs? The current version already includes several optimizations and is well-documented.",
        "route_to": "human"
            }
                    ]  
              }

        


    # Store nodes and edges with chief agent routing
    for query_flow in [query1_flow,query2_flow,query3_flow,query4_flow,query5_flow,query6_flow]:  # Add other flows similarly
        current_query_id = f"query_{uuid.uuid4()}"
        chief_node_id = f"chief_{uuid.uuid4()}"
        
        # Add query node
        data_manager.graph.add_node(current_query_id,
            content=query_flow["query"],
            timestamp=datetime.now().isoformat(),
            type="initial_query"
        )
        
        # Add chief node connected to query
        data_manager.graph.add_node(chief_node_id,
            content="Initial query analysis",
            agent="chief",
            timestamp=datetime.now().isoformat()
        )
        data_manager.graph.add_edge(current_query_id, chief_node_id,
            type="INITIATES",
            timestamp=datetime.now().isoformat()
        )
        
        prev_node_id = chief_node_id
        for interaction in query_flow["flow"]:
            interaction_id = f"message_{uuid.uuid4()}"
            
            # Add interaction node
            data_manager.graph.add_node(interaction_id,
                content=interaction["content"],
                agent=interaction["agent"],
                timestamp=datetime.now().isoformat()
            )
            
            # Add edge from previous node
            data_manager.graph.add_edge(prev_node_id, interaction_id,
                type="ROUTES_TO" if interaction.get("route_to") else "RESPONDS_TO",
                timestamp=datetime.now().isoformat()
            )
            
            # If this is a response to chief, add edge back to chief
            if interaction.get("feedback_to") == "chief":
                chief_response_id = f"chief_{uuid.uuid4()}"
                data_manager.graph.add_node(chief_response_id,
                    content=interaction.get("chief_response", "Analyzing response..."),
                    agent="chief",
                    timestamp=datetime.now().isoformat()
                )
                data_manager.graph.add_edge(interaction_id, chief_response_id,
                    type="FEEDBACK",
                    timestamp=datetime.now().isoformat()
                )
                prev_node_id = chief_response_id
            else:
                prev_node_id = interaction_id
    
    return data_manager




# Save test data
test_graph = generate_test_data()
test_graph.visualize_graph()

nx.write_graphml(test_graph.graph, "test_conversation_graph.graphml")
