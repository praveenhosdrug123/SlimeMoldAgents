# utils/debug_utils.py
import logging
import time
import traceback
import json
import os
from functools import wraps
from typing import Callable, Any, Optional
from datetime import datetime

class DebugUtils:
    """Unified debugging and logging utility class"""
    
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DebugUtils, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DebugUtils._is_initialized:
            self.setup_logging()
            DebugUtils._is_initialized = True
    
    @staticmethod
    def setup_logging(log_dir: str = "logs", 
                     level: int = logging.DEBUG, 
                     retention_days: int = 7) -> None:
        """Configure logging settings with rotation and cleanup"""
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Clean old log files
        DebugUtils._cleanup_old_logs(log_dir, retention_days)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging initialized. Log file: {log_file}")
    
    @staticmethod
    def _cleanup_old_logs(log_dir: str, retention_days: int) -> None:
        """Remove log files older than retention_days"""
        current_time = time.time()
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath):
                file_time = os.path.getmtime(filepath)
                if (current_time - file_time) > (retention_days * 86400):
                    try:
                        os.remove(filepath)
                        print(f"Removed old log file: {filename}")
                    except Exception as e:
                        print(f"Error removing {filename}: {e}")
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        return logging.getLogger(name)
    
    @staticmethod
    def trace_function(func: Callable) -> Callable:
        """Decorator for function tracing"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the class if it's a method, otherwise use function name
            if hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(func.__name__)
            
            start_time = time.time()
            
            # Create trace ID for this function call
            trace_id = f"{func.__name__}_{datetime.now().strftime('%H%M%S%f')}"
            
            # Log entry
            logger.debug(f"[{trace_id}] Entering {func.__name__}")
            
            # Log args (excluding self for methods)
            if len(args) > 1:
                logger.debug(f"[{trace_id}] Args: {args[1:]}")
            if kwargs:
                logger.debug(f"[{trace_id}] Kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"[{trace_id}] Exiting {func.__name__}. Execution time: {execution_time:.2f}s")
                return result
            except Exception as e:
                logger.error(f"[{trace_id}] Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
        return wrapper
    
    @staticmethod
    def log_state(state: Any, logger: logging.Logger) -> None:
        """Log the current state"""
        try:
            state_dict = {
                "current_agent": getattr(state, 'current_agent', None),
                "path_history": getattr(state, 'path_history', []),
                "context": getattr(state, 'context', {}),
                "message_count": len(getattr(state, 'messages', []))
            }
            logger.debug(f"Current State: {json.dumps(state_dict, indent=2)}")
        except Exception as e:
            logger.error(f"Error logging state: {str(e)}")
    
    @staticmethod
    def performance_monitor(threshold_ms: float = 1000):
        """Decorator to monitor function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                logger = logging.getLogger(func.__name__)
                if execution_time > threshold_ms:
                    logger.warning(
                        f"Performance warning: {func.__name__} took {execution_time:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                
                return result
            return wrapper
        return decorator
