import warnings
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.core.summarize import GeminiModel
from src.utils.logging_config import setup_logging
from config.settings import Config
from src.utils.display import display_results

logger = setup_logging(__name__)

from config.settings import Config

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

def main(
    audio_path: str, transcption_model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], summarize_model: str,  num_speakers=int | None,
) -> None:

    audio_path = Path(audio_path)
    # Validar arquivo existe
    if not Path(audio_path).exists():
        logger.error(f"Arquivo não encontrado: {audio_path}")
        sys.exit(1)
    
    # Processar
    try:
        processor = AudioProcessor(num_speakers=num_speakers, model_name=transcption_model, use_diarization=True, hf_token=Config.HF_TOKEN)
        output_name = audio_path.name
        result, outout_path = processor.process(audio_path=audio_path, output_name=output_name)
        
        display_results(result)
        
        processor.cleanup()

        summarize_processor = GeminiModel(api_key="AIzaSyCA7M5z6wClaqk6ZCtabOT1bsPSCqjhHaw", model_name=summarize_model)
        config = summarize_processor.configure(
            temperature=0.1,
            top_p=0.95,
            top_k=40,
        )

        summarize_processor.generate(
            prompt=Config.PROJECT_ROOT / "src/prompts/criacao_ata.txt",
            diarized_text=outout_path,
            extra_files={"context_ativos": r"data/context/context_ativos.txt"},
            config=config
        )

    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main(audio_path=r'data/audio.wav', transcption_model='tiny', num_speakers = 2, summarize_model='gemini-2.5-flash')