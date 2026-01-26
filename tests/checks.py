# def diarizar_audio(audio_path, hf_token):
#     print("Iniciando diarização...")
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         token=hf_token
#     )
#     if torch.cuda.is_available():
#         pipeline.to(torch.device("cuda"))
#     diarization = pipeline(audio_path)
#     speaker_segments = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         speaker_segments.append({
#             "start": turn.start,
#             "end": turn.end,
#             "speaker": speaker
#         })
#     print(f"Diarização concluída. Encontrados {len(set([s['speaker'] for s in speaker_segments]))} falantes.")

    
    # return speaker_segments

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())