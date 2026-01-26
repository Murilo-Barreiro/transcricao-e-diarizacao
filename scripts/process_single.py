import warnings
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.utils.logging_config import setup_logging

from src.utils.display import display_results

logger = setup_logging(__name__)

from config.settings import Config

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

def main(audio_path: str, model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], num_speakers=int|None):
    audio_path = Path(audio_path)
    # Validar arquivo existe
    if not Path(audio_path).exists():
        logger.error(f"Arquivo não encontrado: {audio_path}")
        sys.exit(1)
    
    # Processar
    try:
        processor = AudioProcessor(num_speakers=num_speakers, model_name=model, use_diarization=True, hf_token=Config.HF_TOKEN)
        result = processor.process(audio_path=audio_path, output_name=audio_path.name)
        
        display_results(result)
        
        processor.cleanup()
        
    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main(audio_path=r'data/audio.wav', model='base', num_speakers = 2)