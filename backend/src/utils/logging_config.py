"""
logging_config.py

Logging configuration for the healthcare data generation system.

This module provides centralized logging setup for consistent logging
across all modules in the project.
"""

import logging
import logging.config
from pathlib import Path
import sys
from datetime import datetime

from config.settings import LOGS_DIR, LOGGING_CONFIG

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None: # type: ignore
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file name
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"healthcare_system_{timestamp}.log"
    
    log_path = LOGS_DIR / log_file
    
    # Create custom logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'console': {
                'format': '[%(levelname)8s] %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'formatter': 'console',
                'class': 'logging.StreamHandler',
                'stream': sys.stdout
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'detailed',
                'class': 'logging.FileHandler',
                'filename': str(log_path),
                'mode': 'w',  # Overwrite each time
            },
            'file_append': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': str(LOGS_DIR / 'healthcare_system.log'),
                'mode': 'a',  # Append to main log file
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file', 'file_append'],
                'level': 'DEBUG',
                'propagate': False
            },
            # Specific loggers can be configured here
            'src.data_processing': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.validation': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.models': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.api': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_path}")
    logger.info(f"Log directory: {LOGS_DIR}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_function_start(logger: logging.Logger, function_name: str, **kwargs) -> None:
    """
    Log the start of a function with parameters.
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        **kwargs: Function parameters to log
    """
    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "no parameters"
    logger.debug(f"Starting {function_name}({params_str})")

def log_function_end(logger: logging.Logger, function_name: str, result=None) -> None:
    """
    Log the end of a function with result info.
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        result: Function result (optional)
    """
    if result is not None:
        if hasattr(result, '__len__'):
            logger.debug(f"Completed {function_name} - Result size: {len(result)}")
        else:
            logger.debug(f"Completed {function_name} - Result: {type(result).__name__}")
    else:
        logger.debug(f"Completed {function_name}")

def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """
    Log basic information about a pandas DataFrame.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to use in log message
    """
    logger.info(f"{name}: {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"{name} columns: {list(df.columns)}")
    
    # Log memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    logger.debug(f"{name} memory usage: {memory_mb:.2f} MB")

def log_processing_step(logger: logging.Logger, step: str, input_size: int, output_size: int) -> None:
    """
    Log a processing step with input/output sizes.
    
    Args:
        logger: Logger instance
        step: Description of the processing step
        input_size: Size before processing
        output_size: Size after processing
    """
    change = output_size - input_size
    change_pct = (change / input_size * 100) if input_size > 0 else 0
    
    logger.info(f"{step}: {input_size} → {output_size} ({change:+d}, {change_pct:+.1f}%)")

class LoggingContext:
    """
    Context manager for function-level logging.
    """
    
    def __init__(self, logger: logging.Logger, function_name: str, **kwargs):
        self.logger = logger
        self.function_name = function_name
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_function_start(self.logger, self.function_name, **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time # type: ignore
        
        if exc_type is None:
            self.logger.debug(f"Completed {self.function_name} in {duration.total_seconds():.2f}s")
        else:
            self.logger.error(f"Error in {self.function_name} after {duration.total_seconds():.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions

def logged_function(logger: logging.Logger):
    """
    Decorator to automatically log function entry and exit.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingContext(logger, func.__name__, **kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage functions for testing
def test_logging():
    """Test function to verify logging setup."""
    setup_logging()
    
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test context manager
    with LoggingContext(logger, "test_function", param1="value1", param2=42):
        logger.info("Inside context manager")
    
    print("Logging test completed. Check the logs directory for output files.")