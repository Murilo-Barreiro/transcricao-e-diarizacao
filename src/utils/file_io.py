import json
from pathlib import Path
from typing import List
from src.models.schemas import TranscriptionSegment, ProcessingResult
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

def save_transcription(segments: List[TranscriptionSegment],output_path: Path,metadata: dict = None) -> None:
    """Salva transcrição em arquivo de texto"""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(
                f"[{segment.start:.2f} - {segment.end:.2f}] "
                f"{segment.text}\n"
            )
        
        if metadata:
            f.write("\n\n=== METADADOS ===\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Transcrição salva: {output_path}")


def save_diarized_transcription(segments: List[TranscriptionSegment], output_path: Path, metadata: dict = None) -> None:
    """Salva transcrição com diarização formatada por speaker"""
    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        
        for segment in segments:
            if segment.speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                f.write(f"--- {segment.speaker} ---\n")
                current_speaker = segment.speaker
            
            f.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n")
        
        if metadata:
            f.write("\n\n=== ESTATÍSTICAS ===\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Transcrição com diarização salva: {output_path}")


def save_json_result(result: ProcessingResult, output_path: Path) -> None:
    """Salva resultado em JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultado JSON salvo: {output_path}")


def append_to_results_log(result: ProcessingResult, log_path: Path) -> None:
    """Adiciona resultado ao log acumulado"""
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False)
        f.write("\n")
    
    logger.debug(f"Resultado adicionado ao log: {log_path}")
