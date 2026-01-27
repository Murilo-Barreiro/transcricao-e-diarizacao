from typing import List
from pathlib import Path
import torch
import torchaudio

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
from pyannote.audio import Pipeline

from config.settings import Config
from src.models.schemas import SpeakerSegment
from src.utils.audio import resample_audio
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class AudioDiarizer:
    """Gerencia diarização de áudio usando Pyannote"""
    def __init__(self,hf_token: str = None, device:str = None, num_speakers: int = None,):

        self.hf_token = hf_token or Config.HF_TOKEN
        self.device = torch.device(
            device if device else Config.DEVICE
        )
        self.num_speakers = num_speakers
        self.pipeline = None
        
        if not self.hf_token:
            raise ValueError(
                "Token HuggingFace não fornecido. "
                "Configure HF_TOKEN no .env"
            )
        
        logger.info(
            f"AudioDiarizer inicializado: device={self.device}, "
            f"num_speakers = {self.num_speakers}"
        )
    
    def load_pipeline(self) -> None:
        """Carrega o pipeline de diarização"""
        if self.pipeline is not None:
            logger.debug("Pipeline já carregado")
            return
        
        try:
            logger.info("Carregando pipeline de diarização...")
            self.pipeline = Pipeline.from_pretrained(
                Config.DIARIZATION_MODEL,
                token=self.hf_token
            )
            self.pipeline.to(self.device)
            logger.info(f"Pipeline carregado em {self.device}")
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline: {e}")
            raise
    
    def process(self, audio_path: str) -> List[SpeakerSegment]:
        """Processa áudio e identifica speakers"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        if self.pipeline is None:
            self.load_pipeline()
        
        logger.info(f"Processando diarização: {audio_path}")
        
        try:
            # Carregar áudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample se necessário
            waveform = resample_audio(
                waveform,
                sample_rate,
                Config.SAMPLE_RATE,
                str(self.device)
            )
            
            # Configurar parâmetros de diarização
            diarization_params = {
                "waveform": waveform,
                "sample_rate": Config.SAMPLE_RATE,
                "min_speakers": self.num_speakers,
                "max_speakers": self.num_speakers
            }
            
            # Executar diarização
            outputs = self.pipeline(diarization_params)
            
            # Extrair anotações
            if hasattr(outputs, "speaker_diarization"):
                diarization = outputs.speaker_diarization
            elif hasattr(outputs, "annotation"):
                diarization = outputs.annotation
            else:
                raise ValueError("Formato de saída do pipeline desconhecido")
            
            # Converter para nossa estrutura
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker
                    )
                )
            
            num_speakers = len(set(seg.speaker for seg in speaker_segments))
            logger.info(
                f"Diarização concluída: {num_speakers} speakers, "
                f"{len(speaker_segments)} segmentos"
            )
            
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Erro durante diarização: {e}")
            raise
        
        finally:
            # Liberar memória GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_pipeline(self) -> None:
        """Libera pipeline da memória"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Pipeline descarregado da memória")