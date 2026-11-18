# Sistema de Transcrição e Diarização de Áudio

Sistema para transcrição automática de reuniões entre assessores de investimentos e clientes, com identificação de falantes (diarização) e geração de atas/resumos usando IA.

## Funcionalidades

- **Transcrição de áudio** usando Whisper (faster-whisper)
- **Diarização** (identificação de falantes) usando Pyannote
- **Geração de atas e resumos** usando OpenAI GPT ou Google Gemini
- Suporte a múltiplos formatos de áudio e vídeo
- Processamento em lote de múltiplos arquivos
- Logs detalhados de processamento

## Requisitos

- Python 3.10+
- CUDA (opcional, para processamento em GPU)
- FFmpeg (para conversão de vídeo)
- Tokens de API:
  - HuggingFace (para diarização)
  - OpenAI ou Google Gemini (para geração de resumos)

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/Murilo-Barreiro/transcricao-e-diarizacao.git
cd transcricao-e-diarizacao
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

Para processar em GPU, instale o torch do índice do PyTorch antes das demais
dependências, para não baixar a build de CPU por cima:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Instalar FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Baixe em https://ffmpeg.org/download.html e adicione ao PATH

### 5. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
HF_TOKEN=seu_token_huggingface
OPENAI_API_KEY=sua_chave_openai
GEMINI_API_KEY=sua_chave_gemini
```

**Como obter os tokens:**
- HuggingFace: https://huggingface.co/settings/tokens
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://aistudio.google.com/app/apikey

## Estrutura do Projeto

```
├── config/
│   └── settings.py          # Configurações globais
├── src/
│   ├── core/
│   │   ├── processor.py     # Orquestrador principal
│   │   ├── transcriber.py   # Transcrição com Whisper
│   │   ├── diarizer.py      # Diarização com Pyannote
│   │   ├── summarizer.py    # Geração de resumos
│   │   ├── summarize.py     # Motor genérico de sumarização
│   │   └── speaker_assignment.py
│   ├── models/
│   │   └── schemas.py       # Modelos de dados
│   └── utils/
│       ├── audio.py         # Utilitários de áudio
│       ├── cli_args.py      # Argumentos de linha de comando
│       ├── files.py         # Manipulação de arquivos
│       └── logging_config.py
├── scripts/
│   ├── process_single.py    # Processar um arquivo
│   ├── process_batch.py     # Processar múltiplos arquivos
│   └── benchmark.py         # Benchmark de modelos
├── tests/                   # Testes de CLI, plumbing e saída
├── data/
│   ├── input/               # Arquivos de entrada
│   └── prompts/             # Prompts para IA
└── outputs/
    ├── transcriptions/      # Transcrições puras
    ├── diarizations/        # Transcrições com falantes
    ├── summaries/           # Resumos/atas gerados
    ├── logs/                # Logs de processamento
    └── stats/               # Estatísticas JSON
```

## Uso

### Processar um único arquivo

```bash
python scripts/process_single.py
```

Edite o arquivo `scripts/process_single.py` para configurar:
- Caminho do áudio
- Número de falantes
- Modelo de transcrição (tiny, base, small, medium, large, turbo)

### Processar múltiplos arquivos

```bash
python scripts/process_batch.py --input-dir /caminho/dos/audios
```

Opções:
```bash
--input-dir, -i      Diretório com arquivos de áudio (obrigatório)
--pattern, -p        Padrão de arquivos (default: *.wav)
--model, -m          Modelo Whisper (default: base)
--no-diarization     Desabilitar diarização
--continue-on-error  Continuar se houver erros
```

### Benchmark de modelos

```bash
python scripts/benchmark.py --audio /caminho/audio.wav --models tiny base small
```

## Modelos Whisper Disponíveis

| Modelo | Tamanho | VRAM  | Velocidade | Qualidade |
|--------|---------|-------|------------|-----------|
| tiny   | 39 MB   | ~1 GB | Muito rápido | Básica   |
| base   | 74 MB   | ~1 GB | Rápido     | Boa      |
| small  | 244 MB  | ~2 GB | Médio      | Muito boa |
| medium | 769 MB  | ~5 GB | Lento      | Excelente |
| large  | 1550 MB | ~10 GB| Muito lento| Máxima   |
| turbo  | 809 MB  | ~6 GB | Rápido     | Excelente|

**Recomendação:** Use `base` para testes e `turbo` ou `small` para produção.

## Formatos Suportados

### Áudio
- WAV, MP3, FLAC, OGG, M4A

### Vídeo
O sistema extrai automaticamente o áudio de:
- MP4, MKV

## Exemplo de Código

```python
from src.core.processor import AudioProcessor
from config.settings import Config

# Criar processador
processor = AudioProcessor(
    model_name='base',           # Modelo Whisper
    use_diarization=True,        # Ativar diarização
    num_speakers=2,              # Número de falantes
    hf_token=Config.HF_TOKEN
)

# Processar áudio
result, diarized_path = processor.process(
    file_path='audio.wav',
    output_name='reuniao_2024'
)

# Limpar recursos
processor.cleanup()

print(f"Duração: {result.duration_sec}s")
print(f"Transcrição: {diarized_path}")
```

## Configurações

Edite `config/settings.py` para ajustar:

```python
DEFAULT_TRANSCRIPTION_MODEL = 'turbo'  # Modelo padrão
DEFAULT_SUMMARIZATION_MODEL = 'gpt-4o-mini'  # Modelo para resumos
NUM_SPEAKERS = 2  # Número de falantes
DEFAULT_LANGUAGE = "pt"  # Idioma
```

## Outputs

Após o processamento, você encontrará:

```
outputs/
├── transcriptions/
│   └── reuniao_2024.txt          # Transcrição simples
├── diarizations/
│   └── reuniao_2024_diarized.txt # Transcrição com falantes
├── atas/
│   └── ata_reuniao_2024.txt      # Resumo gerado por IA
├── stats/
│   └── reuniao_2024_stats.json   # Estatísticas
└── logs/
    └── processing_log.jsonl      # Logs de processamento
```

## Licença

Este projeto é de uso interno.

## Suporte

Para dúvidas ou problemas, abra uma issue no repositório.

## Uso (CLI)

Perfil genérico (detecção automática de idioma e falantes):

```bash
python scripts/process_single.py \
  --audio caminho/do/arquivo.mp4 \
  --prompt prompts/generic_summary.txt
```

Perfil ata (reunião, valores fixos e contexto extra — arquivos locais, não versionados):

```bash
python scripts/process_single.py \
  --audio data/input/sample1 \
  --prompt data/prompts/main.txt \
  --extra context_ativos=data/prompts/context_ativos.txt \
  --num-speakers 2 --language pt --provider openai
```

Flags: `--output` (nome-base da saída), `--provider {openai,gemini}`,
`--num-speakers {auto|N}`, `--language {auto|código}`. Saída em `outputs/summaries/`.
