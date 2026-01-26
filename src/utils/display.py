
def display_results(result) -> None:
    print("\n" + "="*60)
    print("RESUMO DO PROCESSAMENTO")
    print("="*60)
    print(f"Arquivo: {result.audio_path}")
    print(f"Duração: {result.duration_sec:.2f}s")
    print(f"Idioma: {result.language}")
    print(f"Modelo: {result.model}")
    print(f"Speakers: {result.num_speakers}")
    print(f"Segmentos: {len(result.segments)}")
    print(f"Tempo total: {result.transcription_time_sec:.2f}s")
    print("="*60)