import argparse
import sys
from pathlib import Path
from time import perf_counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Processa múltiplos arquivos de áudio em batch"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Diretório com arquivos de áudio"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*.wav",
        help="Padrão de arquivos a processar (default: *.wav)"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Modelo Whisper a usar (default: base)"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Desabilitar diarização de speakers"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continuar processamento mesmo se houver erros"
    )
    
    args = parser.parse_args()
    
    # Validar diretório
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Diretório não encontrado: {input_dir}")
        sys.exit(1)
    
    # Encontrar arquivos
    audio_files = list(input_dir.glob(args.pattern))
    if not audio_files:
        logger.error(f"Nenhum arquivo encontrado com padrão: {args.pattern}")
        sys.exit(1)
    
    logger.info(f"Encontrados {len(audio_files)} arquivos para processar")
    
    # Processar arquivos
    processor = AudioProcessor(
        model_name=args.model,
        use_diarization=not args.no_diarization
    )
    
    results = []
    errors = []
    start_time = perf_counter()
    
    try:
        for audio_file in tqdm(audio_files, desc="Processando arquivos"):
            try:
                result = processor.process(str(audio_file))
                results.append(result)
                logger.info(f"✅ Sucesso: {audio_file.name}")
            
            except Exception as e:
                error_msg = f"❌ Erro em {audio_file.name}: {e}"
                logger.error(error_msg)
                errors.append((audio_file.name, str(e)))
                
                if not args.continue_on_error:
                    raise
    
    finally:
        processor.cleanup()
    
    # Resumo final
    total_time = perf_counter() - start_time
    
    print("\n" + "="*60)
    print("RESUMO DO BATCH")
    print("="*60)
    print(f"Total de arquivos: {len(audio_files)}")
    print(f"Processados com sucesso: {len(results)}")
    print(f"Erros: {len(errors)}")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Tempo médio por arquivo: {total_time/len(audio_files):.2f}s")
    
    if errors:
        print("\n❌ Arquivos com erro:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
    
    print("="*60)


if __name__ == "__main__":
    main()