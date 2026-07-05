# Design: motor genérico de transcrição, diarização e sumarização

Data: 2026-07-05 · Branch: `develop`

## Objetivo

Transformar o projeto de um "gerador de ata de reunião" num motor genérico de
áudio/vídeo para insight. A sumarização deve atender qualquer conteúdo (palestra,
podcast, entrevista, aula) a partir de um prompt genérico, mantendo o perfil de ata
como uma das aplicações possíveis. O repositório deve passar a rodar de ponta a ponta
para quem o clona (bastando as chaves no `.env`).

Base de contexto: `docs/HANDOFF.md`.

## Decisões de design (fechadas no brainstorming)

1. **CLI com flags explícitas**, sem preset escondido. O perfil "ata" é apenas uma
   combinação específica de flags; o perfil "genérico" usa os defaults.
2. **`--provider {openai,gemini}`** (default `openai`): fia o Gemini na CLI e corrige a
   inconsistência do nome de saída fixo `"gemini"` no summarizer.
3. **Saída neutra fixa**: sempre `outputs/summaries/summary_<output>_<modelo>.txt`. O
   nome-base vem de `--output` (ou do nome do áudio, se omitido). Nada de `atas/ata_` no
   código.

## Manter os DOIS perfis (sem remover a ata)

- **Perfil ata**: `--prompt data/prompts/main.txt --extra context_ativos=data/prompts/context_ativos.txt --num-speakers 2 --language pt --provider openai`.
- **Perfil genérico**: `--audio <arquivo> --prompt prompts/generic_summary.txt` (resto nos defaults `auto`).

## Mudanças por arquivo

### `scripts/process_single.py` — reescrever `main()` com `argparse`

Flags:

| Flag | Obrigatória | Default | Efeito |
|------|-------------|---------|--------|
| `--audio` | sim | — | caminho do áudio/vídeo |
| `--prompt` | sim | — | caminho do arquivo de prompt |
| `--output` | não | nome do áudio | nome-base da saída |
| `--extra` | não (repetível) | — | `placeholder=arquivo` (ex.: `context_ativos=data/prompts/context_ativos.txt`) |
| `--provider` | não | `openai` | `openai` \| `gemini` |
| `--num-speakers` | não | `auto` | inteiro ou `auto` (→ `None`) |
| `--language` | não | `auto` | código (ex. `pt`) ou `auto` (→ `None`) |

Comportamento:
- Remover as globais fixas `prompt_ata_score = main.txt` e `context_ativos` (linhas ~20-21)
  e o input fixo `Config.INPUT_DIR / "sample1"` (linha ~43).
- `--num-speakers`: aceitar a string `auto` (→ `None`) ou um inteiro. Validar entrada.
- `--language`: aceitar `auto` (→ `None`) ou string de idioma.
- `--extra`: repetível; cada valor no formato `chave=caminho`, montado num dict
  `{placeholder: caminho}` passado a `summarizer.generate(...)`. Erro claro se faltar `=`.
- `--provider`: selecionar `OpenAIModel` (com `Config.OPENAI_API_KEY` /
  `Config.DEFAULT_SUMMARIZATION_MODEL`) ou `GeminiModel` (com `Config.GEMINI_API_KEY`).
  Escolher um modelo default de Gemini razoável (ex.: constante em Config) quando provider=gemini.
- `num_speakers` vai para `AudioProcessor(num_speakers=...)`; `language` vai para
  `processor.process(..., language=...)`.
- Manter o guard de arquivo inexistente e o `try/except` com log + `sys.exit(1)`.

### `src/core/processor.py` — deixar `num_speakers` e `language` fluírem

- Linhas ~35-36: **remover** a coerção `if self.num_speakers is None: self.num_speakers = 2`.
  Passar `num_speakers` (possivelmente `None`) direto ao `AudioDiarizer`, que já plumba
  `min_speakers`/`max_speakers` e deixa o Pyannote auto-detectar quando `None`.
- `process()`: adicionar parâmetro `language: str | None = None` e repassar em
  `self.transcriber.transcribe(audio_path, language=language)` (linha ~58).
- Atenção: `_save_outputs` usa `speaker_segments` para nomear o arquivo diarizado. Com
  `num_speakers=None` o fluxo continua igual (o Pyannote retorna os segmentos); apenas o
  número de falantes passa a ser detectado. Conferir que `metadata["Número de falantes"]`
  continua correto (já é derivado de `set(seg.speaker ...)`).

### `src/core/transcriber.py` — idioma automático

- Linha ~52: trocar default `language: str = "pt"` → `language: str | None = None`.
- Remover a linha redundante `language = language`.
- Docstring já diz "None = auto-detect"; agora o comportamento bate com o docstring.

### `src/core/summarizer.py` — neutralizar saída e corrigir Gemini

- `save_results` (linha ~30): `Config.OUTPUT_DIR / f"atas/ata_{audio_name}_{model_name}.txt"`
  → `Config.OUTPUT_DIR / f"summaries/summary_{audio_name}_{model_name}.txt"`.
- Mensagem `print(... outputs/atas)` (linha ~39) → `outputs/summaries`.
- `SummarizationResult(summary_path="outputs/atas/")` nas linhas ~68 e ~128 →
  `"outputs/summaries/"` (nos dois modelos).
- Linha ~136 (Gemini): `save_results(response.text, result, "gemini", self.model_name)`
  → usar `diarized_text.name` como `audio_name`, igual ao OpenAIModel (linha ~76). Assim o
  nome do arquivo reflete o áudio, não o provedor.

### `config/settings.py` — defaults genéricos e diretório de saída

- `NUM_SPEAKERS = 2` → `NUM_SPEAKERS = None` (auto por padrão; a ata força via CLI).
- `DEFAULT_LANGUAGE = "pt"` → `DEFAULT_LANGUAGE = None` (auto por padrão).
- Adicionar `(OUTPUT_DIR / "summaries").mkdir(exist_ok=True)` no bloco de `mkdir`.
- Adicionar constante de modelo Gemini default (para o `--provider gemini`), ex.:
  `DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"` (ou valor equivalente já usado no projeto).
- `HUBSPOT_TOKEN` permanece como está: lido de env, opcional, fora do perfil genérico.

### `prompts/generic_summary.txt` — novo, versionado

- Criar a pasta `prompts/` na raiz (FORA de `data/`, que é gitignored).
- Conteúdo: instrução neutra em português, usando o placeholder `<arquivo_input>` para
  receber a transcrição diarizada. Deve pedir: temas abordados, pontos principais,
  conclusões ou decisões, e trechos marcantes. Sem qualquer referência a reunião/ata/CRM.

## Verificação

O pipeline real depende de modelos pesados (faster-whisper, Pyannote) e chaves de API,
então a verificação principal é **manual, ponta a ponta**:

1. **Perfil genérico**: rodar num áudio/vídeo público de tema qualquer, sem `--num-speakers`
   nem `--language`, e obter transcrição diarizada + resumo coerente em
   `outputs/summaries/summary_...`.
2. **Perfil ata (não-regressão)**: rodar com as flags da ata e confirmar que a saída é
   equivalente à de antes (mesmo prompt, `context_ativos`, 2 falantes, idioma pt).

Testes unitários leves (sem rede, executáveis por quem clona):
- Parsing das flags: `--num-speakers auto` → `None`, valor inteiro preservado;
  `--language auto` → `None`; `--extra a=1 --extra b=2` → `{"a": "1", "b": "2"}`;
  `--extra` mal-formado (sem `=`) gera erro claro.
- `prepare_prompt`: substitui `<arquivo_input>` e os placeholders de `extra_files`.

## Critérios de aceite

- Rodar o pipeline num vídeo/áudio público sem informar falantes nem idioma e obter
  transcrição diarizada + resumo coerente.
- O perfil de ata continua funcionando exatamente como antes (sem regressão).
- Repositório com prompt genérico versionado e script parametrizado, executável por quem
  clona (bastando as chaves no `.env`).
- Nada corporativo versionado: `main.txt`, `context_ativos.txt` e `.env` seguem gitignored;
  `HUBSPOT_TOKEN` continua opcional e fora do perfil genérico.

## Cuidados

- Não commitar prompts/contexto corporativos nem dado de reunião real.
- Ao mexer em `num_speakers`/`language`, garantir que a ata ainda passa os valores fixos
  corretos (2 e pt) via CLI, para não regredir o caso original.
