import argparse
import sys
from pathlib import Path
from time import perf_counter
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from src.core.processor import AudioProcessor
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def benchmark_model(audio_path: str, model_name: str) -> dict:
    """Benchmarca um modelo específico"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testando modelo: {model_name}")
    logger.info(f"{'='*60}")
    
    processor = AudioProcessor(
        model_name=model_name,
        use_diarization=False  # Desabilitar para benchmark puro
    )
    
    start_time = perf_counter()
    result = processor.process(audio_path)
    end_time = perf_counter()
    
    processor.cleanup()
    
    return {
        "model": model_name,
        "audio_duration": result.duration_sec,
        "processing_time": end_time - start_time,
        "rtf": (end_time - start_time) / result.duration_sec,  # Real Time Factor
        "language": result.language,
        "num_segments": len(result.segments)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de modelos Whisper"
    )
    parser.add_argument(
        "--audio",
        "-a",
        required=True,
        help="Caminho do arquivo de áudio"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["tiny", "base", "small"],
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Modelos a testar (default: tiny base small)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Arquivo para salvar resultados JSON"
    )
    
    args = parser.parse_args()
    
    # Validar arquivo
    if not Path(args.audio).exists():
        logger.error(f"Arquivo não encontrado: {args.audio}")
        sys.exit(1)
    
    # Executar benchmarks
    results = []
    
    for model in args.models:
        try:
            result = benchmark_model(args.audio, model)
            results.append(result)
        except Exception as e:
            logger.error(f"Erro ao testar {model}: {e}")
    
    # Exibir resultados
    print("\n" + "="*80)
    print("RESULTADOS DO BENCHMARK")
    print("="*80)
    print(f"{'Modelo':<15} {'Duração':<12} {'Tempo Proc.':<15} {'RTF':<10} {'Segmentos'}")
    print("-"*80)
    
    for r in results:
        print(
            f"{r['model']:<15} "
            f"{r['audio_duration']:<12.2f} "
            f"{r['processing_time']:<15.2f} "
            f"{r['rtf']:<10.2f} "
            f"{r['num_segments']}"
        )
    
    print("="*80)
    print(f"\nRTF = Real Time Factor (< 1.0 = mais rápido que tempo real)")
    
    # Salvar em JSON se solicitado
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()