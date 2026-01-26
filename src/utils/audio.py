import torch
import torchaudio
from pathlib import Path
from typing import Tuple
from config.settings import Config
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Carrega e processa arquivo de áudio
    
    Args:
        audio_path: Caminho do arquivo
        
    Returns:
        Tupla (waveform, sample_rate)
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
    
    logger.info(f"Carregando áudio: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    return waveform, sample_rate


def resample_audio(
    waveform: torch.Tensor,
    orig_sample_rate: int,
    target_sample_rate: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Reamostra áudio para taxa de amostragem alvo
    
    Args:
        waveform: Tensor de áudio
        orig_sample_rate: Taxa original
        target_sample_rate: Taxa alvo
        device: Device para processamento
        
    Returns:
        Waveform reamostrado
    """
    if orig_sample_rate == target_sample_rate:
        return waveform.to(device)
    
    logger.info(f"Resampling: {orig_sample_rate}Hz -> {target_sample_rate}Hz")
    resampler = torchaudio.transforms.Resample(
        orig_sample_rate, 
        target_sample_rate
    ).to(device)
    
    return resampler(waveform.to(device))


def validate_audio_format(audio_path: str) -> bool:
    """
    Valida se o formato de áudio é suportado
    
    Args:
        audio_path: Caminho do arquivo
        
    Returns:
        True se válido
    """
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    suffix = Path(audio_path).suffix.lower()
    
    if suffix not in supported_formats:
        raise ValueError(
            f"Formato não suportado: {suffix}. "
            f"Formatos suportados: {supported_formats}"
        )
    
    return True