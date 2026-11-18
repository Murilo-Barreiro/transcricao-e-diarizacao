import os
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import List, Dict, Optional

import torch
import torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings conhecidos
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class Config:
    """Configurações centralizadas"""
    HF_TOKEN = os.getenv("HF_TOKEN", "")  # Usar variável de ambiente
    OUTPUT_DIR = Path("outputs")
    MODEL = "base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 16000
    COMPUTE_TYPE = "float16"

Config.OUTPUT_DIR.mkdir(exist_ok=True)


class AudioDiarizer:
    """Classe para gerenciar diarização de áudio"""
    
    def __init__(self, hf_token: str, device: str = "cuda"):
        if not hf_token:
            raise ValueError("Token HuggingFace não fornecido. Configure a variável HF_TOKEN no .env")
        
        self.hf_token = hf_token
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
    def load_pipeline(self):
        """Carrega o pipeline de diarização"""
        try:
            logger.info("Carregando pipeline de diarização...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            self.pipeline.to(self.device)
            logger.info(f"Pipeline carregado em {self.device}")
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline: {e}")
            raise
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Processa áudio e retorna segmentos de falantes
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Lista de dicionários com start, end, speaker
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        if self.pipeline is None:
            self.load_pipeline()
        
        logger.info(f"Processando diarização: {audio_path}")
        
        try:
            # Carregar áudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample se necessário
            if sample_rate != Config.SAMPLE_RATE:
                logger.info(f"Resampling de {sample_rate}Hz para {Config.SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(
                    sample_rate, Config.SAMPLE_RATE
                ).to(self.device)
                waveform = resampler(waveform.to(self.device))
            else:
                waveform = waveform.to(self.device)
            
            # Diarização
            outputs = self.pipeline({
                "waveform": waveform,
                "sample_rate": Config.SAMPLE_RATE
            })
            
            # Extrair anotações
            if hasattr(outputs, "speaker_diarization"):
                diarization = outputs.speaker_diarization
            elif hasattr(outputs, "annotation"):
                diarization = outputs.annotation
            else:
                raise ValueError("Formato de saída do pipeline desconhecido")
            
            # Converter para lista de segmentos
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            logger.info(f"Encontrados {len(set(s['speaker'] for s in speaker_segments))} falantes")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Erro durante diarização: {e}")
            raise
        finally:
            # Liberar memória GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class WhisperTranscriber:
    """Classe para gerenciar transcrição com Whisper"""
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def load_model(self):
        """Carrega o modelo Whisper"""
        try:
            logger.info(f"Carregando modelo Whisper: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=Config.COMPUTE_TYPE
            )
            logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: str = "pt") -> tuple:
        """
        Transcreve áudio
        
        Returns:
            Tupla (segments_list, info)
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        if self.model is None:
            self.load_model()
        
        logger.info(f"Iniciando transcrição: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(audio_path, language=language)
            segments_list = list(segments)
            logger.info(f"Transcrição concluída: {len(segments_list)} segmentos")
            return segments_list, info
        except Exception as e:
            logger.error(f"Erro durante transcrição: {e}")
            raise


def assign_speaker(
    segment_start: float,
    segment_end: float,
    speaker_segments: List[Dict]
) -> str:
    """
    Atribui falante ao segmento baseado em overlap máximo
    
    Args:
        segment_start: Início do segmento (segundos)
        segment_end: Fim do segmento (segundos)
        speaker_segments: Lista de segmentos de falantes
        
    Returns:
        Nome do falante atribuído
    """
    max_overlap = 0
    assigned_speaker = "DESCONHECIDO"
    
    for sp_seg in speaker_segments:
        overlap_start = max(segment_start, sp_seg["start"])
        overlap_end = min(segment_end, sp_seg["end"])
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_speaker = sp_seg["speaker"]
    
    return assigned_speaker


def save_transcription(
    segments: List,
    model_name: str,
    info,
    transcription_time: float,
    output_file: Path
):
    """Salva transcrição simples"""
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n")
        
        f.write(f"\n\n=== INFORMAÇÕES ===\n")
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Tempo de transcrição: {transcription_time:.2f}s\n")
    
    logger.info(f"Transcrição salva em: {output_file}")


def save_diarized_transcription(
    segments: List,
    speaker_segments: List[Dict],
    model_name: str,
    info,
    transcription_time: float,
    output_file: Path
):
    """Salva transcrição com diarização"""
    num_speakers = len(set(s['speaker'] for s in speaker_segments))
    
    with open(output_file, "w", encoding="utf-8") as f:
        current_speaker = None
        
        for segment in segments:
            speaker = assign_speaker(segment.start, segment.end, speaker_segments)
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                f.write(f"--- {speaker} ---\n")
                current_speaker = speaker
            
            f.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n")
        
        f.write(f"\n\n=== ESTATÍSTICAS ===\n")
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Número de falantes: {num_speakers}\n")
        f.write(f"Duração do áudio: {info.duration:.2f}s\n")
        f.write(f"Idioma detectado: {info.language}\n")
        f.write(f"Tempo de processamento: {transcription_time:.2f}s\n")
    
    logger.info(f"Transcrição com diarização salva em: {output_file}")


def process_audio_file(
    audio_path: str,
    model_name: str = "base",
    use_diarization: bool = True,
    hf_token: Optional[str] = None
) -> Dict:
    """
    Processa arquivo de áudio completo
    
    Args:
        audio_path: Caminho do arquivo de áudio
        model_name: Nome do modelo Whisper
        use_diarization: Se deve fazer diarização
        hf_token: Token HuggingFace
        
    Returns:
        Dicionário com resultados e métricas
    """
    start_time = perf_counter()
    
    # Validações
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
    
    # Transcrição
    transcriber = WhisperTranscriber(model_name, Config.DEVICE)
    segments, info = transcriber.transcribe(audio_path)
    
    # Diarização
    speaker_segments = []
    num_speakers = 0
    
    if use_diarization:
        token = hf_token or Config.HF_TOKEN
        diarizer = AudioDiarizer(token, Config.DEVICE)
        speaker_segments = diarizer.process_audio(audio_path)
        num_speakers = len(set(s['speaker'] for s in speaker_segments))
    
    # Salvar resultados
    out_file = Config.OUTPUT_DIR / f"transcription_{model_name}.txt"
    save_transcription(segments, model_name, info, perf_counter() - start_time, out_file)
    
    if use_diarization:
        out_file_diarized = Config.OUTPUT_DIR / f"transcription_{model_name}_diarized.txt"
        save_diarized_transcription(
            segments, speaker_segments, model_name, info,
            perf_counter() - start_time, out_file_diarized
        )
    
    # Métricas
    total_time = perf_counter() - start_time
    file_size_kb = out_file.stat().st_size // 1024
    
    result = {
        "model": model_name,
        "duration_sec": info.duration,
        "language": info.language,
        "file_kb": file_size_kb,
        "transcription_time_sec": total_time,
        "diarization_enabled": use_diarization,
        "num_speakers": num_speakers,
        "device": str(Config.DEVICE)
    }
    
    logger.info(f"Processamento concluído em {total_time:.2f}s")
    logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


def main(audio_file:str):
    start_time = perf_counter()
    
    try:
        result = process_audio_file(
            audio_path=audio_file,
            model_name=Config.MODEL,
            use_diarization=True,
            hf_token=Config.HF_TOKEN
        )
        
        # Salvar resultados em JSON
        results_file = Config.OUTPUT_DIR / "results.json"
        with open(results_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
        
        total_time = perf_counter() - start_time
        logger.info(f"✅ Tempo total: {int(total_time//60)}min {int(total_time%60)}seg")
        
    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main("audio.wav")