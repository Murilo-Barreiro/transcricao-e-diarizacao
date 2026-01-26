import os
from pathlib import Path
import torch

class Config:
    """Configurações globais do projeto"""
    # Tokens e API Keys
    HF_TOKEN = "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"
    
    # Diretórios
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # Criar diretórios se não existirem
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "transcriptions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "diarizations").mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Configurações de Modelo
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    # Configurações de Áudio
    SAMPLE_RATE = 16000
    DEFAULT_LANGUAGE = "pt"
    
    # Configurações de Diarização
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    NUM_SPEAKERS = None
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"