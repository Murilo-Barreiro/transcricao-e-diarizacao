from typing import List
from src.models.schemas import SpeakerSegment

def assign_speaker_to_segment(
    segment_start: float,
    segment_end: float,
    speaker_segments: List[SpeakerSegment]
) -> str:
    """
    Atribui speaker ao segmento baseado em overlap máximo
    
    Args:
        segment_start: Início do segmento
        segment_end: Fim do segmento
        speaker_segments: Lista de segmentos de speakers
        
    Returns:
        Nome do speaker atribuído
    """
    max_overlap = 0
    assigned_speaker = "DESCONHECIDO"
    
    for sp_seg in speaker_segments:
        overlap_start = max(segment_start, sp_seg.start)
        overlap_end = min(segment_end, sp_seg.end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_speaker = sp_seg.speaker
    
    return assigned_speaker


def calculate_speaker_statistics(
    speaker_segments: List[SpeakerSegment]
) -> dict:
    """
    Calcula estatísticas sobre os speakers
    
    Args:
        speaker_segments: Lista de segmentos
        
    Returns:
        Dicionário com estatísticas
    """
    speakers = {}
    
    for segment in speaker_segments:
        if segment.speaker not in speakers:
            speakers[segment.speaker] = {
                "total_time": 0,
                "num_segments": 0
            }
        
        speakers[segment.speaker]["total_time"] += segment.duration
        speakers[segment.speaker]["num_segments"] += 1
    
    return {
        "num_speakers": len(speakers),
        "speakers": speakers
    }