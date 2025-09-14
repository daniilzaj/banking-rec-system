import logging
import sys
from logging import FileHandler
from pythonjsonlogger import jsonlogger
from .config import settings

def setup_logger(log_filepath: str):
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console Handler (for seeing logs in the terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (for writing JSON logs to a file)
    file_handler = FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)