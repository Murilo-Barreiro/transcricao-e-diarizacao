import os
import json
import warnings
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchaudio
from openai import OpenAI

from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
from pyannote.audio import Pipeline

import os
from google import genai

hf_token = "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"

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OUTPUT_DIR = Path(r"tests/outputs")
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
    return result, out_file_diarized

def gerar_resumo_gpt(diarizaed_text_file: str, contexto_ativos_file_path) -> str:
    
    def carregar_arquivo (arquivo):
        with open(arquivo, "r", encoding="utf-8") as f:
            return f.read()
    
    client = OpenAI(api_key="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")
    
    prompt = f"""
        Você é um especialista em geração de atas de reuniões no contexto do mercado financeiro brasileiro.
        Seu objetivo é analisar uma transcrição diarizada de uma reunião entre assessor(es) de investimentos e cliente(s) e gerar uma ata estruturada e detalhada.
        A transcrição real a ser analisada será fornecida após estas instruções.
        ----------------------------------------
        REGRAS GERAIS
        ----------------------------------------
        1. Identifique corretamente todos os participantes.
        2. Classifique cada participante como:
        - Assessor
        - Cliente
        - Outro participante relevante (se houver)
        3. Caso os papéis não estejam explícitos, infira com base no contexto das falas.
        4. Considere que pode haver múltiplos assessores e/ou múltiplos clientes.
        5. Ignore ruídos de transcrição, vícios de linguagem e repetições irrelevantes.
        6. O contexto macroeconômico e de ativos deve ser considerado sempre que citado.
        ----------------------------------------
        ESTRUTURA OBRIGATÓRIA DA ATA
        ----------------------------------------
        A ata deve conter exatamente as seguintes seções:
        1. Introdução  
        2. Assessor(es)  
        3. Cliente(s)  
        4. Próximos Passos  
        ----------------------------------------
        1. INTRODUÇÃO
        ----------------------------------------
        - Descreva um panorama geral da reunião.
        - Informe o tema central.
        - Liste os participantes identificados e seus papéis.
        - Contextualize os principais assuntos discutidos.
        - Seja descritivo, mas objetivo.
        ----------------------------------------
        2. ASSESSOR(ES)
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Liste recomendações, análises, justificativas e alertas feitos.
        - Destaque argumentos técnicos utilizados (CDI, IPCA, Selic, Ibovespa etc., quando aplicável).
        - Cite estratégias propostas (alocação, rebalanceamento, proteção, diversificação etc.).
        - Se houver mais de um assessor, organize por nome.
        ----------------------------------------
        3. CLIENTE(S)
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Liste dúvidas, preocupações, objeções e feedbacks.
        - Destaque perfil de risco mencionado.
        - Registre preferências por classes de ativos.
        - Identifique metas financeiras ou horizonte de investimento.
        - Se houver mais de um cliente, organizar por nome.
        ----------------------------------------
        4. PRÓXIMOS PASSOS
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Cada bullet deve começar com um emoji que represente ação.
        - Transforme os pontos discutidos em ações concretas.
        - Inclua:
        - Ajustes de carteira
        - Envio de materiais
        - Agendamento de retorno
        - Análises complementares
        - Justificativas baseadas em indicadores
        - Os próximos passos devem ser práticos e executáveis.
        ----------------------------------------
        CONTEXTOS DE APOIO
        ----------------------------------------

        Contexto sobre ativos:
        {carregar_arquivo(contexto_ativos_file_path)}

        ----------------------------------------
        IMPORTANTE
        ----------------------------------------

        - Utilize linguagem profissional e formal.
        - Não invente informações que não estejam na transcrição.
        - Caso algum dado não esteja claro, registre como "não especificado na reunião".
        - Priorize clareza, estrutura e profundidade analítica.
        """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": carregar_arquivo(diarizaed_text_file)
            }
        ]
    )

    with open(r"tests/outputs/ata_gerada_gpt.txt", "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)

    print("Resumo gerado e salvo em 'ata_gerada_gpt.txt'")

def gerar_resumo_gemini(diarizaed_text_file: str, contexto_ativos_file_path) -> str:
    # Configuração da API - Certifique-se de definir a variável de ambiente ou colar a chave aqui
    client = genai.Client(api_key="AIzaSyCA7M5z6wClaqk6ZCtabOT1bsPSCqjhHaw")

    def carregar_arquivo(caminho):
        """Auxiliar para ler os arquivos de texto."""
        with open(caminho, 'r', encoding='utf-8') as f:
            return f.read()

    # Inicializa o modelo (Flash é recomendado para benchmarks de velocidade)
    prompt = f"""
        Você é um especialista em geração de atas de reuniões no contexto do mercado financeiro brasileiro.
        Seu objetivo é analisar uma transcrição diarizada de uma reunião entre assessor(es) de investimentos e cliente(s) e gerar uma ata estruturada e detalhada.
        A transcrição real a ser analisada será fornecida após estas instruções.
        ----------------------------------------
        REGRAS GERAIS
        ----------------------------------------
        1. Identifique corretamente todos os participantes.
        2. Classifique cada participante como:
        - Assessor
        - Cliente
        - Outro participante relevante (se houver)
        3. Caso os papéis não estejam explícitos, infira com base no contexto das falas.
        4. Considere que pode haver múltiplos assessores e/ou múltiplos clientes.
        5. Ignore ruídos de transcrição, vícios de linguagem e repetições irrelevantes.
        6. O contexto macroeconômico e de ativos deve ser considerado sempre que citado.
        ----------------------------------------
        ESTRUTURA OBRIGATÓRIA DA ATA
        ----------------------------------------
        A ata deve conter exatamente as seguintes seções:
        1. Introdução  
        2. Assessor(es)  
        3. Cliente(s)  
        4. Próximos Passos  
        ----------------------------------------
        1. INTRODUÇÃO
        ----------------------------------------
        - Descreva um panorama geral da reunião.
        - Informe o tema central.
        - Liste os participantes identificados e seus papéis.
        - Contextualize os principais assuntos discutidos.
        - Seja descritivo, mas objetivo.
        ----------------------------------------
        2. ASSESSOR(ES)
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Liste recomendações, análises, justificativas e alertas feitos.
        - Destaque argumentos técnicos utilizados (CDI, IPCA, Selic, Ibovespa etc., quando aplicável).
        - Cite estratégias propostas (alocação, rebalanceamento, proteção, diversificação etc.).
        - Se houver mais de um assessor, organize por nome.
        ----------------------------------------
        3. CLIENTE(S)
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Liste dúvidas, preocupações, objeções e feedbacks.
        - Destaque perfil de risco mencionado.
        - Registre preferências por classes de ativos.
        - Identifique metas financeiras ou horizonte de investimento.
        - Se houver mais de um cliente, organizar por nome.
        ----------------------------------------
        4. PRÓXIMOS PASSOS
        ----------------------------------------
        - Apresente no mínimo 6 bullet points.
        - Cada bullet deve começar com um emoji que represente ação.
        - Transforme os pontos discutidos em ações concretas.
        - Inclua:
        - Ajustes de carteira
        - Envio de materiais
        - Agendamento de retorno
        - Análises complementares
        - Justificativas baseadas em indicadores
        - Os próximos passos devem ser práticos e executáveis.
        ----------------------------------------
        CONTEXTOS DE APOIO
        ----------------------------------------

        Contexto sobre ativos:
        {carregar_arquivo(contexto_ativos_file_path)}

        ----------------------------------------
        TRANSCRIÇÃO PARA ANÁLISE
        ----------------------------------------
        {carregar_arquivo(diarizaed_text_file)}

        ----------------------------------------
        IMPORTANTE
        ----------------------------------------

        - Utilize linguagem profissional e formal.
        - Não invente informações que não estejam na transcrição.
        - Caso algum dado não esteja claro, registre como "não especificado na reunião".
        - Priorize clareza, estrutura e profundidade analítica.
        """

    # Nova forma de chamada no SDK unificado
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    with open(r"tests/outputs/ata_gerada_gemini.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Resumo gerado e salvo em 'ata_gerada_gemini.txt'")

def main():
    # Transcrever e Diarizar 
    if not Path(r"tests/audio2.wav").exists():
        print("Arquivo 'audio.wav' não encontrado.")
        return

    result, output_path = processa_modelo(
        modelo='base',
        usar_diarizacao=True,
        hf_token=hf_token
    )
    
    with open(OUTPUT_DIR / "results.json", "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")

    #Gerar ata de reunião
    gerar_resumo_gemini(rf"{output_path}.txt", r"tests/contexto_ativos.txt")
    # gerar_resumo_gpt(r"tests/transcription_base_GPU_diarized.txt", r"tests/contexto_ativos.txt")
    
if __name__ == "__main__":
    main()