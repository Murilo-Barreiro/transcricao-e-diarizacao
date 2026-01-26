import logging
import sys
from pathlib import Path
from config.settings import Config

def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Configura sistema de logging
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    # Evitar duplicação de handlers
    if logger.handlers:
        return logger
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(Config.LOG_LEVEL)
    console_formatter = logging.Formatter(Config.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # Handler para arquivo
    log_file = Config.LOGS_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger