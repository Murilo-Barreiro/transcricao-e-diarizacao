import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.core.summarizer import OpenAIModel, GeminiModel
from src.utils.cli_args import build_parser, parse_extra
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

from config.settings import Config

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")


def build_summarizer(provider: str):
    """Instancia o modelo de sumarização conforme o provider."""
    if provider == "openai":
        return OpenAIModel(Config.OPENAI_API_KEY, Config.DEFAULT_SUMMARIZATION_MODEL)
    if provider == "gemini":
        return GeminiModel(Config.GEMINI_API_KEY, Config.DEFAULT_GEMINI_MODEL)
    raise ValueError(f"Provider desconhecido: {provider}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Arquivo não encontrado: {audio_path}")
        sys.exit(1)

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        logger.error(f"Prompt não encontrado: {prompt_path}")
        sys.exit(1)

    try:
        extra_files = parse_extra(args.extra)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    output_name = args.output or audio_path.stem

    try:
        # Transcrição e diarização
        processor = AudioProcessor(
            num_speakers=args.num_speakers,
            model_name=Config.DEFAULT_TRANSCRIPTION_MODEL,
            use_diarization=True,
            hf_token=Config.HF_TOKEN,
        )
        _, diarized_path = processor.process(
            file_path=audio_path,
            output_name=output_name,
            language=args.language,
        )
        processor.cleanup()

        # Sumarização
        summarizer = build_summarizer(args.provider)
        summarizer.generate(prompt_path, diarized_path, extra_files)

    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
