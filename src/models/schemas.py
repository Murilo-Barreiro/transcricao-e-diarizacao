from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class SpeakerSegment:
    """Representa um segmento de fala de um speaker"""
    start: float
    end: float
    speaker: str
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionSegment:
    """Representa um segmento de transcrição"""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProcessingResult:
    """Resultado completo do processamento"""
    audio_path: str
    model: str
    duration_sec: float
    language: str
    transcription_time_sec: float
    diarization_enabled: bool
    num_speakers: int
    device: str
    diarized_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    segments: List[TranscriptionSegment] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "model": self.model,
            "duration_sec": self.duration_sec,
            "language": self.language,
            "transcription_time_sec": self.transcription_time_sec,
            "diarization_enabled": self.diarization_enabled,
            "num_speakers": self.num_speakers,
            "device": self.device,
            "diarized_path": self.diarized_path,
            "timestamp": self.timestamp,
            "num_segments": len(self.segments),
        }

@dataclass
class SummarizationResult:
    """Resultado da sumarização"""
    summary_path: str
    model: str
    summary_time_sec: float
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "summary_path": self.summary_path,
            "model": self.model,
            "summary_time_sec": self.summary_time_sec,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": self.timestamp
        }