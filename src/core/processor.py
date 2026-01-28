from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

from config.settings import Config

from src.core.transcriber import WhisperTranscriber
from src.core.diarizer import AudioDiarizer
from src.core.speaker_assignment import assign_speaker_to_segment

from src.models.schemas import ProcessingResult

from src.utils.files import save_transcription,save_diarized_transcription,save_json_result,append_to_results_log
from src.utils.audio import validate_audio_format
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class AudioProcessor:
    """Orquestra todo o processamento de áudio"""
    def __init__(self, model_name:str = None, use_diarization:bool = True, hf_token:str = None, num_speakers:int | None = None):
        self.model_name = model_name
        self.use_diarization = use_diarization
        self.num_speakers = num_speakers
        self.hf_token = hf_token
        
        # Inicializar componentes
        self.transcriber = WhisperTranscriber(model_name=self.model_name)
        self.diarizer = None
        
        if self.use_diarization:
            if self.num_speakers is None:
                self.num_speakers = 2
            self.diarizer = AudioDiarizer(hf_token=self.hf_token, num_speakers=self.num_speakers)
        
        logger.info(
            f"AudioProcessor inicializado: model={self.model_name}, "
            f"diarization={self.use_diarization}"
        )
    
    def process(self, audio_path:str, output_name:str = None) -> Tuple[ProcessingResult, Path]:
        """Processa arquivo de áudio completo"""
        start_time = perf_counter()
        
        # Validações
        validate_audio_format(audio_path)
        audio_path_obj = Path(audio_path)
        
        if output_name is None:
            output_name = audio_path_obj.stem
        
        logger.info(f"========== Processando: {audio_path} ==========")
        
        # 1. Transcrição
        logger.info("Etapa 1/3: Transcrição")
        segments, info = self.transcriber.transcribe(audio_path)
        
        # 2. Diarização (opcional)
        speaker_segments = []
        num_speakers = self.num_speakers
        
        if self.use_diarization and self.diarizer:
            logger.info("Etapa 2/3: Diarização")
            speaker_segments = self.diarizer.process(audio_path)
            
            # Atribuir speakers aos segmentos
            logger.info("Etapa 3/3: Atribuindo speakers aos segmentos")
            for segment in segments:
                segment.speaker = assign_speaker_to_segment(
                    segment.start,
                    segment.end,
                    speaker_segments
                )
        else:
            logger.info("Diarização desabilitada, pulando etapas 2 e 3")
        
        # 3. Salvar resultados
        output_path = self._save_outputs(segments, speaker_segments, output_name, info, perf_counter() - start_time)
        
        # 4. Criar resultado
        result = ProcessingResult(
            audio_path=str(audio_path),
            model=self.model_name,
            duration_sec=info.duration,
            language=info.language,
            transcription_time_sec=perf_counter() - start_time,
            diarization_enabled=self.use_diarization,
            num_speakers=num_speakers,
            device=Config.DEVICE,
            segments=segments
        )
        
        # Salvar resultado em JSON e log
        json_path = Config.OUTPUT_DIR / f"{output_name}_result.json"
        save_json_result(result, json_path)
        
        log_path = Config.OUTPUT_DIR / "processing_log.jsonl"
        append_to_results_log(result, log_path)
        
        total_time = perf_counter() - start_time
        logger.info(
            f"========== Processamento concluído em {total_time:.2f}s =========="
        )
        
        return result, output_path
    
    def _save_outputs(self, segments, speaker_segments, output_name, info, processing_time) -> Path:
        """Salva todos os arquivos de saída"""
        metadata = {
            "Modelo": self.model_name,
            "Duração": f"{info.duration:.2f}s",
            "Idioma": info.language,
            "Tempo de processamento": f"{processing_time:.2f}s"
        }
        
        # Transcrição simples
        transcription_path = (
            Config.OUTPUT_DIR / "transcriptions" / f"{output_name}.txt"
        )
        save_transcription(segments, transcription_path, metadata)
        
        # Transcrição com diarização
        if self.use_diarization and speaker_segments:
            num_speakers = len(set(seg.speaker for seg in speaker_segments))
            metadata["Número de falantes"] = num_speakers
            
            diarized_path = (
                Config.OUTPUT_DIR / "diarizations" / f"{output_name}_diarized.txt"
            )
            save_diarized_transcription(segments, diarized_path, metadata)

        return diarized_path
    
    def cleanup(self) -> None:
        """Libera recursos"""
        self.transcriber.unload_model()
        if self.diarizer:
            self.diarizer.unload_pipeline()
        logger.info("Recursos liberados")