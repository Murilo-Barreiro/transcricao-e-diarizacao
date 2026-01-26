import os
import json
import warnings
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchaudio

from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
from pyannote.audio import Pipeline

hf_token = "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"

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def diarizar_audio(audio_path, hf_token):
    print(f"\n--- Iniciando transcrição para: {audio_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    ).to(device)

    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)

    outputs = pipeline({"waveform": waveform, "sample_rate": 16000, "num_speakers":2})

    if hasattr(outputs, "speaker_diarization"):
        diarization = outputs.speaker_diarization
    elif hasattr(outputs, "annotation"):
        diarization = outputs.annotation

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
            
    return speaker_segments

def atribuir_falante(segment_start, segment_end, speaker_segments):
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

def processa_modelo(modelo, usar_diarizacao=True, hf_token=None):
    print(f"\n--- Iniciando transcrição: {modelo} ---")
    start = datetime.now()
    model = WhisperModel(
        modelo,
        device="cuda",
        compute_type="float16"
    )
    out_file = OUTPUT_DIR / f"transcription_{modelo}_GPU.txt"
    out_file_diarized = OUTPUT_DIR / f"transcription_{modelo}_GPU_diarized.txt"

    segments, info = model.transcribe(r"tests/audio2.wav", language="pt")
    segments_list = list(segments)
    
    # Diarização
    speaker_segments = []
    num_speakers = 0

    speaker_segments = diarizar_audio(r"tests/audio2.wav", hf_token)
    num_speakers = len(set([s['speaker'] for s in speaker_segments]))
    
    # Salva transcrição
    with open(out_file, "w", encoding="utf-8") as f:
        for segment in segments_list:
            f.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n")
        f.write(f"\n\nModelo: {modelo}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Transcription completed in {datetime.now() - start} seconds\n")
    
    # Salva transcrição com diarização
    if usar_diarizacao:
        with open(out_file_diarized, "w", encoding="utf-8") as f:
            current_speaker = None
            for segment in segments_list:
                speaker = atribuir_falante(segment.start, segment.end, speaker_segments)
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n\n")
                    f.write(f"--- {speaker} ---\n")
                    current_speaker = speaker
                
                f.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n")
            
            f.write(f"\n\n=== ESTATÍSTICAS ===\n")
            f.write(f"Modelo: {modelo}\n")
            f.write(f"Número de falantes: {num_speakers}\n")
            f.write(f"Info: {info}\n")
            f.write(f"Transcription completed in {datetime.now() - start} seconds\n")

    transcription_time = datetime.now() - start
    file_size_kb = out_file.stat().st_size // 1024

    result = {
        "model": modelo,
        "duration_sec": info.duration,
        "language": info.language,
        "file_kb": file_size_kb,
        "transcription_time": str(transcription_time),
        "transcription_time_sec": transcription_time.total_seconds(),
        "diarization_enabled": usar_diarizacao,
        "num_speakers": num_speakers if usar_diarizacao else 0
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))
    return result

def main():
    if not Path(r"tests/audio2.wav").exists():
        print("Arquivo 'audio.wav' não encontrado.")
        return

    result = processa_modelo(
        modelo='base',
        usar_diarizacao=True,
        hf_token=hf_token
    )
    
    with open(OUTPUT_DIR / "results.json", "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    main()