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


from pathlib import Path
import subprocess


def validate_file(file_path: str) -> str:
    """Valida o formato do arquivo.
    Se for áudio, retorna o próprio path.
    Se for vídeo, extrai o áudio em mp3 e retorna o novo path.
    """
    audio_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    video_formats = {'.mp4', '.mkv'}

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in audio_formats:
        return str(file_path)

    elif suffix in video_formats:
        output_path = file_path.with_suffix(".mp3")

        command = [
            "ffmpeg",
            "-y",
            "-i", str(file_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            str(output_path)
        ]

        subprocess.run(command, check=True)

        return str(output_path)

    else:
        supported = sorted(audio_formats | video_formats)
        raise ValueError(
            f"Formato não suportado: {suffix}. "
            f"Formatos suportados: {supported}"
        )
    