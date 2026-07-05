from typing import List, Tuple
from pathlib import Path
import torch
from faster_whisper import WhisperModel

from config.settings import Config
from src.models.schemas import TranscriptionSegment
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class WhisperTranscriber:
    """Gerencia transcrição de áudio usando Whisper"""
    def __init__(self, model_name:str, device:str = None, compute_type:str = None):
        """
        Inicializa o transcritor
        
        Args:
            model_name: Nome do modelo Whisper
            device: Device para processamento
            compute_type: Tipo de computação
        """
        self.model_name = model_name
        self.device = device or Config.DEVICE
        self.compute_type = compute_type or Config.COMPUTE_TYPE
        self.model = None
        
        logger.info(
            f"WhisperTranscriber inicializado: "
            f"model={self.model_name}, device={self.device}"
        )
    
    def load_model(self) -> None:
        """Carrega o modelo Whisper"""
        if self.model is not None:
            logger.debug("Modelo já carregado")
            return
        
        try:
            logger.info(f"Carregando modelo Whisper: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: str | None = None) -> Tuple[List[TranscriptionSegment], dict]:
        """
        Transcreve arquivo de áudio
        
        Args:
            audio_path: Caminho do arquivo
            language: Código do idioma (None = auto-detect)
            
        Returns:
            Tupla (lista de segmentos, info do áudio)
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        if self.model is None:
            self.load_model()

        logger.info(f"Iniciando transcrição: {audio_path} (language={language})")
        
        try:
            segments, info = self.model.transcribe(audio_path,language=language,)
            # Converter para schema
            transcription_segments = []
            for segment in segments:
                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip()
                    )
                )
            
            logger.info(
                f"Transcrição concluída: {len(transcription_segments)} segmentos, "
                f"duração={info.duration:.2f}s, idioma={info.language}"
            )
            
            return transcription_segments, info
            
        except Exception as e:
            logger.error(f"Erro durante transcrição: {e}")
            raise
    
    def unload_model(self) -> None:
        """Libera modelo da memória"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Modelo descarregado da memória")