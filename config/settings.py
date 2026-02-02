import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configurações globais do projeto"""
    HF_TOKEN = os.getenv("HF_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUBSPOT_TOKEN = os.getenv("HUBSPOT_TOKEN")
    
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = OUTPUT_DIR / "logs"
    STATS_DIR = OUTPUT_DIR / "stats"
    PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"
    INPUT_DIR = PROJECT_ROOT / "data" / "input"
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "transcriptions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "diarizations").mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    DEFAULT_TRANSCRIPTION_MODEL = 'turbo'
    DEFAULT_SUMMARIZATION_MODEL = 'gpt-5-mini'

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    SAMPLE_RATE = 16000
    DEFAULT_LANGUAGE = "pt"
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    NUM_SPEAKERS = 2

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"