import warnings
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.core.summarizer import OpenAIModel
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

from config.settings import Config

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

prompt_ata_score = Config.PROMPTS_DIR / "main.txt"
context_ativos = Config.PROMPTS_DIR / "context_ativos.txt"

def main(audio_path: str):
    audio_path = Path(audio_path)
    if not Path(audio_path).exists():
        logger.error(f"Arquivo não encontrado: {audio_path}")
        sys.exit(1)
    try:
        #Tracrição e diarização
        processor = AudioProcessor(num_speakers=Config.NUM_SPEAKERS, model_name=Config.DEFAULT_TRANSCRIPTION_MODEL, use_diarization=True, hf_token=Config.HF_TOKEN)
        _, diarized_path = processor.process(file_path=audio_path, output_name=audio_path.name)
        processor.cleanup()

        #Criação de ata e score da reunião
        summarizer = OpenAIModel(Config.OPENAI_API_KEY, Config.DEFAULT_SUMMARIZATION_MODEL)
        summarizer.generate(prompt_ata_score, diarized_path, {'context_ativos':context_ativos})

    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main(Config.INPUT_DIR / "sample1")