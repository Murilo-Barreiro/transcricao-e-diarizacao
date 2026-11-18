# Motor Genérico de Transcrição/Diarização/Sumarização — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o pipeline de "gerador de ata" num motor genérico de áudio/vídeo, parametrizado por CLI, mantendo o perfil de ata sem regressão.

**Architecture:** Extrair a lógica de parsing de CLI para um módulo leve e testável (`src/utils/cli_args.py`); deixar `num_speakers` e `language` fluírem como `None` (auto) pelo core; neutralizar os nomes de saída da sumarização; e reescrever `scripts/process_single.py` como CLI com flags explícitas (`--audio`, `--prompt`, `--output`, `--extra`, `--provider`, `--num-speakers`, `--language`). Um prompt genérico versionado torna o repo executável por quem clona.

**Tech Stack:** Python 3, argparse, stdlib `unittest` (sem pytest — não é dependência do projeto), faster-whisper, Pyannote, OpenAI e Google GenAI SDKs.

## Global Constraints

- Manter os DOIS perfis: **ata** (via flags `--prompt data/prompts/main.txt --extra context_ativos=... --num-speakers 2 --language pt --provider openai`) e **genérico** (defaults `auto`).
- Nada corporativo versionado: `data/` (inclui `main.txt`, `context_ativos.txt`) e `.env` seguem gitignored. `HUBSPOT_TOKEN` continua opcional e fora do perfil genérico.
- Provider default: `openai`. Modelo Gemini default: `gemini-2.5-flash`.
- Saída neutra fixa: `outputs/summaries/summary_<nome>_<modelo>.txt`. Nenhuma string `atas`/`ata_` no código.
- Testes com stdlib `unittest`, executáveis via `python tests/<arquivo>.py -v` a partir da raiz do repo (cada arquivo insere a raiz no `sys.path`).
- Working dir da raiz do repo: `c:/Users/mukkk/transcricao-e-diarizacao`. Branch: `develop`.

---

## File Structure

- **Create** `src/utils/cli_args.py` — funções puras de parsing + `build_parser()`. Sem imports pesados (só `argparse`).
- **Create** `tests/test_cli_args.py` — testes unitários do parsing (rápidos, sem torch).
- **Create** `tests/test_core_plumbing.py` — testes de assinatura/fluxo de `num_speakers` e `language`.
- **Create** `tests/test_summary_output.py` — testes do caminho neutro e defaults do Config.
- **Create** `prompts/generic_summary.txt` — prompt genérico versionado.
- **Modify** `src/core/transcriber.py` — default `language=None`.
- **Modify** `src/core/processor.py` — remover coerção para 2; `process()` aceita e repassa `language`.
- **Modify** `src/core/summarizer.py` — helper `summary_output_path`, saída neutra, fix nome Gemini.
- **Modify** `config/settings.py` — `NUM_SPEAKERS=None`, `DEFAULT_LANGUAGE=None`, `DEFAULT_GEMINI_MODEL`, mkdir `summaries`/`stats`.
- **Modify** `scripts/process_single.py` — reescrever com argparse e seleção de provider.
- **Modify** `README.md` — bloco curto de uso da CLI.

---

## Task 1: Módulo de parsing de CLI (`cli_args.py`)

**Files:**
- Create: `src/utils/cli_args.py`
- Test: `tests/test_cli_args.py`

**Interfaces:**
- Produces:
  - `parse_num_speakers(value: str) -> int | None` — `"auto"`→`None`; inteiro ≥1; erro `argparse.ArgumentTypeError` caso contrário.
  - `parse_language(value: str) -> str | None` — `"auto"`→`None`; senão a própria string.
  - `parse_extra(pairs: list[str]) -> dict[str, str]` — cada item `"chave=caminho"`; erro `ValueError` se faltar `=` ou chave vazia.
  - `build_parser() -> argparse.ArgumentParser` — parser com todas as flags; `--num-speakers`/`--language` já convertidos via `type=`.

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_cli_args.py`:

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cli_args import (
    parse_num_speakers,
    parse_language,
    parse_extra,
    build_parser,
)


class TestParseNumSpeakers(unittest.TestCase):
    def test_auto_becomes_none(self):
        self.assertIsNone(parse_num_speakers("auto"))

    def test_integer_preserved(self):
        self.assertEqual(parse_num_speakers("2"), 2)

    def test_invalid_raises(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_num_speakers("dois")

    def test_zero_raises(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_num_speakers("0")


class TestParseLanguage(unittest.TestCase):
    def test_auto_becomes_none(self):
        self.assertIsNone(parse_language("auto"))

    def test_code_preserved(self):
        self.assertEqual(parse_language("pt"), "pt")


class TestParseExtra(unittest.TestCase):
    def test_multiple_pairs(self):
        result = parse_extra(["a=1.txt", "b=2.txt"])
        self.assertEqual(result, {"a": "1.txt", "b": "2.txt"})

    def test_empty_list(self):
        self.assertEqual(parse_extra([]), {})

    def test_missing_equals_raises(self):
        with self.assertRaises(ValueError):
            parse_extra(["semigual"])

    def test_empty_key_raises(self):
        with self.assertRaises(ValueError):
            parse_extra(["=x.txt"])

    def test_path_with_equals_kept(self):
        # split apenas no primeiro '='
        self.assertEqual(parse_extra(["k=a=b"]), {"k": "a=b"})


class TestBuildParser(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["--audio", "a.mp3", "--prompt", "p.txt"])
        self.assertEqual(args.audio, "a.mp3")
        self.assertEqual(args.prompt, "p.txt")
        self.assertIsNone(args.output)
        self.assertEqual(args.extra, [])
        self.assertEqual(args.provider, "openai")
        self.assertIsNone(args.num_speakers)
        self.assertIsNone(args.language)

    def test_full_args(self):
        args = build_parser().parse_args([
            "--audio", "a", "--prompt", "p",
            "--num-speakers", "2", "--language", "pt",
            "--extra", "context_ativos=x.txt",
            "--provider", "gemini",
            "--output", "reuniao",
        ])
        self.assertEqual(args.num_speakers, 2)
        self.assertEqual(args.language, "pt")
        self.assertEqual(args.extra, ["context_ativos=x.txt"])
        self.assertEqual(args.provider, "gemini")
        self.assertEqual(args.output, "reuniao")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `python tests/test_cli_args.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.utils.cli_args'`.

- [ ] **Step 3: Implementar o módulo**

Create `src/utils/cli_args.py`:

```python
import argparse


def parse_num_speakers(value: str) -> int | None:
    """Converte o argumento --num-speakers: 'auto' -> None, senão inteiro >= 1."""
    if value == "auto":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--num-speakers deve ser 'auto' ou inteiro, recebido: {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--num-speakers deve ser >= 1, recebido: {n}"
        )
    return n


def parse_language(value: str) -> str | None:
    """Converte o argumento --language: 'auto' -> None, senão o próprio código."""
    return None if value == "auto" else value


def parse_extra(pairs: list[str]) -> dict[str, str]:
    """Converte pares 'chave=caminho' num dict de placeholders extras."""
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"--extra deve ser 'chave=caminho', recebido: {pair!r}"
            )
        key, path = pair.split("=", 1)
        if not key:
            raise ValueError(f"--extra com chave vazia: {pair!r}")
        result[key] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser da CLI de processamento."""
    parser = argparse.ArgumentParser(
        description="Motor de transcrição, diarização e sumarização de áudio/vídeo."
    )
    parser.add_argument("--audio", required=True, help="Caminho do arquivo de áudio/vídeo")
    parser.add_argument("--prompt", required=True, help="Caminho do arquivo de prompt")
    parser.add_argument(
        "--output", default=None,
        help="Nome-base da saída (default: nome do arquivo de áudio)",
    )
    parser.add_argument(
        "--extra", action="append", default=[], metavar="CHAVE=CAMINHO",
        help="Arquivo extra para placeholder do prompt, formato chave=caminho (repetível)",
    )
    parser.add_argument(
        "--provider", choices=["openai", "gemini"], default="openai",
        help="Provedor de sumarização (default: openai)",
    )
    parser.add_argument(
        "--num-speakers", dest="num_speakers", type=parse_num_speakers, default="auto",
        help="Número de falantes ou 'auto' para detectar (default: auto)",
    )
    parser.add_argument(
        "--language", type=parse_language, default="auto",
        help="Idioma (ex.: pt) ou 'auto' para detectar (default: auto)",
    )
    return parser
```

Nota: `argparse` aplica a função `type=` também ao default quando ele é string, então `default="auto"` vira `None` via `parse_num_speakers`/`parse_language`.

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `python tests/test_cli_args.py -v`
Expected: PASS (todos os testes OK).

- [ ] **Step 5: Commit**

```bash
git add src/utils/cli_args.py tests/test_cli_args.py
git commit -m "feat: módulo de parsing de CLI (cli_args) com testes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Idioma e nº de falantes automáticos no core

**Files:**
- Modify: `src/core/transcriber.py:52` e `:69`
- Modify: `src/core/processor.py:34-36` e `:44` e `:58`
- Test: `tests/test_core_plumbing.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces:
  - `WhisperTranscriber.transcribe(self, audio_path, language: str | None = None)` — default `None` (auto-detect).
  - `AudioProcessor.process(self, file_path, output_name=None, language: str | None = None)` — repassa `language` ao transcritor; `num_speakers=None` flui até o diarizer.

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_core_plumbing.py`:

```python
import sys
import inspect
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.transcriber import WhisperTranscriber
from src.core.processor import AudioProcessor


class TestLanguageDefaults(unittest.TestCase):
    def test_transcribe_default_language_is_none(self):
        sig = inspect.signature(WhisperTranscriber.transcribe)
        self.assertIsNone(sig.parameters["language"].default)

    def test_process_accepts_language_none_default(self):
        sig = inspect.signature(AudioProcessor.process)
        self.assertIn("language", sig.parameters)
        self.assertIsNone(sig.parameters["language"].default)


class TestNumSpeakersFlow(unittest.TestCase):
    def test_none_flows_to_diarizer(self):
        p = AudioProcessor(
            num_speakers=None, model_name="tiny",
            use_diarization=True, hf_token="fake",
        )
        self.assertIsNone(p.num_speakers)
        self.assertIsNone(p.diarizer.num_speakers)

    def test_fixed_value_preserved(self):
        p = AudioProcessor(
            num_speakers=2, model_name="tiny",
            use_diarization=True, hf_token="fake",
        )
        self.assertEqual(p.num_speakers, 2)
        self.assertEqual(p.diarizer.num_speakers, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `python tests/test_core_plumbing.py -v`
Expected: FAIL — `test_transcribe_default_language_is_none` falha (default é `"pt"`); `test_none_flows_to_diarizer` falha (`num_speakers` vira 2); `test_process_accepts_language_none_default` falha (parâmetro `language` inexistente).

- [ ] **Step 3: Ajustar `transcriber.py`**

Em `src/core/transcriber.py`, trocar a assinatura (linha ~52) e remover a linha redundante (linha ~69).

Antes:
```python
    def transcribe(self,audio_path: str,language: str = "pt") -> Tuple[List[TranscriptionSegment], dict]:
```
Depois:
```python
    def transcribe(self, audio_path: str, language: str | None = None) -> Tuple[List[TranscriptionSegment], dict]:
```

Remover esta linha (era `language = language`, sem efeito):
```python
        language = language
```

- [ ] **Step 4: Ajustar `processor.py`**

Em `src/core/processor.py`, remover a coerção para 2 no `__init__` (linhas ~34-37).

Antes:
```python
        if self.use_diarization:
            if self.num_speakers is None:
                self.num_speakers = 2
            self.diarizer = AudioDiarizer(hf_token=self.hf_token, num_speakers=self.num_speakers)
```
Depois:
```python
        if self.use_diarization:
            self.diarizer = AudioDiarizer(hf_token=self.hf_token, num_speakers=self.num_speakers)
```

Adicionar `language` ao `process()` (assinatura na linha ~44) e repassar na chamada ao transcritor (linha ~58).

Antes:
```python
    def process(self, file_path:str, output_name:str = None) -> Tuple[ProcessingResult, str]:
```
Depois:
```python
    def process(self, file_path:str, output_name:str = None, language: str | None = None) -> Tuple[ProcessingResult, str]:
```

Antes:
```python
        segments, info = self.transcriber.transcribe(audio_path)
```
Depois:
```python
        segments, info = self.transcriber.transcribe(audio_path, language=language)
```

- [ ] **Step 5: Rodar os testes e confirmar que passam**

Run: `python tests/test_core_plumbing.py -v`
Expected: PASS (4 testes OK).

- [ ] **Step 6: Commit**

```bash
git add src/core/transcriber.py src/core/processor.py tests/test_core_plumbing.py
git commit -m "feat: idioma e nº de falantes automáticos (None flui pelo core)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Saída neutra + defaults do Config

**Files:**
- Modify: `config/settings.py:23-27` (mkdir), `:35` (`DEFAULT_LANGUAGE`), `:37` (`NUM_SPEAKERS`), e nova constante `DEFAULT_GEMINI_MODEL`
- Modify: `src/core/summarizer.py:28-39` (helper + saída), `:68`, `:128`, `:136`
- Test: `tests/test_summary_output.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces:
  - `summarizer.summary_output_path(audio_name: str, model_name: str) -> Path` → `outputs/summaries/summary_<audio_name>_<model_name>.txt`.
  - `Config.NUM_SPEAKERS is None`, `Config.DEFAULT_LANGUAGE is None`, `Config.DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"`, diretório `outputs/summaries/` criado.

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_summary_output.py`:

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from src.core.summarizer import summary_output_path


class TestSummaryOutputPath(unittest.TestCase):
    def test_under_summaries_dir(self):
        p = summary_output_path("meu_audio", "gpt-5-mini")
        self.assertEqual(p.parent.name, "summaries")

    def test_neutral_filename(self):
        p = summary_output_path("meu_audio", "gpt-5-mini")
        self.assertEqual(p.name, "summary_meu_audio_gpt-5-mini.txt")

    def test_no_atas_reference(self):
        self.assertNotIn("atas", str(summary_output_path("x", "y")))


class TestConfigDefaults(unittest.TestCase):
    def test_num_speakers_none(self):
        self.assertIsNone(Config.NUM_SPEAKERS)

    def test_language_none(self):
        self.assertIsNone(Config.DEFAULT_LANGUAGE)

    def test_gemini_model(self):
        self.assertEqual(Config.DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")

    def test_summaries_dir_created(self):
        self.assertTrue((Config.OUTPUT_DIR / "summaries").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `python tests/test_summary_output.py -v`
Expected: FAIL — `ImportError` de `summary_output_path` (ainda não existe).

- [ ] **Step 3: Ajustar `config/settings.py`**

No bloco de `mkdir` (linhas ~23-27), acrescentar `summaries` e `stats`:

Antes:
```python
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "transcriptions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "diarizations").mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
```
Depois:
```python
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "transcriptions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "diarizations").mkdir(exist_ok=True)
    (OUTPUT_DIR / "summaries").mkdir(exist_ok=True)
    STATS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
```

Trocar os defaults (linhas ~35 e ~37) e adicionar o modelo Gemini default (junto de `DEFAULT_SUMMARIZATION_MODEL`, linha ~30):

Antes:
```python
    DEFAULT_SUMMARIZATION_MODEL = 'gpt-5-mini'
```
Depois:
```python
    DEFAULT_SUMMARIZATION_MODEL = 'gpt-5-mini'
    DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'
```

Antes:
```python
    DEFAULT_LANGUAGE = "pt"
```
Depois:
```python
    DEFAULT_LANGUAGE = None
```

Antes:
```python
    NUM_SPEAKERS = 2
```
Depois:
```python
    NUM_SPEAKERS = None
```

- [ ] **Step 4: Ajustar `src/core/summarizer.py`**

Adicionar o helper e usá-lo em `save_results` (substituindo o bloco das linhas ~28-39):

Antes:
```python
def save_results(generated_text: str, result: SummarizationResult, audio_name: str, model_name: str) -> None:
    """Salva os resultados em arquivos."""
    output_text = Config.OUTPUT_DIR / f"atas/ata_{audio_name}_{model_name}.txt"
    save_summarization_output(generated_text, output_text)
    
    output_json = Config.STATS_DIR  / f"summary_stats_{audio_name}_{model_name}.json"
    save_json_result(result, output_json)
    
    log_path = Config.LOGS_DIR / "processing_log.jsonl"
    append_to_results_log(result, log_path)
    
    print(f"Resumo gerado e salvo em {Config.OUTPUT_DIR}/atas")
```
Depois:
```python
def summary_output_path(audio_name: str, model_name: str) -> Path:
    """Caminho neutro de saída da sumarização."""
    return Config.OUTPUT_DIR / "summaries" / f"summary_{audio_name}_{model_name}.txt"


def save_results(generated_text: str, result: SummarizationResult, audio_name: str, model_name: str) -> None:
    """Salva os resultados em arquivos."""
    output_text = summary_output_path(audio_name, model_name)
    save_summarization_output(generated_text, output_text)
    
    output_json = Config.STATS_DIR  / f"summary_stats_{audio_name}_{model_name}.json"
    save_json_result(result, output_json)
    
    log_path = Config.LOGS_DIR / "processing_log.jsonl"
    append_to_results_log(result, log_path)
    
    print(f"Resumo gerado e salvo em {Config.OUTPUT_DIR}/summaries")
```

Trocar os dois `summary_path="outputs/atas/"` (linhas ~68 e ~128) por `summary_path="outputs/summaries/"`:

Antes (nos dois modelos):
```python
            summary_path="outputs/atas/",
```
Depois:
```python
            summary_path="outputs/summaries/",
```

Corrigir o nome fixo do Gemini (linha ~136):

Antes:
```python
        save_results(response.text, result, "gemini", self.model_name)
```
Depois:
```python
        save_results(response.text, result, diarized_text.name, self.model_name)
```

- [ ] **Step 5: Rodar os testes e confirmar que passam**

Run: `python tests/test_summary_output.py -v`
Expected: PASS (7 testes OK).

- [ ] **Step 6: Commit**

```bash
git add config/settings.py src/core/summarizer.py tests/test_summary_output.py
git commit -m "feat: saída neutra em outputs/summaries + defaults auto no Config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Reescrever a CLI (`process_single.py`)

**Files:**
- Modify: `scripts/process_single.py` (arquivo inteiro)

**Interfaces:**
- Consumes:
  - `src.utils.cli_args.build_parser`, `src.utils.cli_args.parse_extra`
  - `AudioProcessor(num_speakers, model_name, use_diarization, hf_token)` e `.process(file_path, output_name, language)`
  - `OpenAIModel(api_key, model_name)`, `GeminiModel(api_key, model_name)`, ambos com `.generate(prompt, diarized_text, extra_files)`
  - `Config.OPENAI_API_KEY`, `Config.GEMINI_API_KEY`, `Config.DEFAULT_SUMMARIZATION_MODEL`, `Config.DEFAULT_GEMINI_MODEL`, `Config.DEFAULT_TRANSCRIPTION_MODEL`, `Config.HF_TOKEN`
- Produces: script executável por CLI; `main(argv=None)` para permitir teste de smoke.

- [ ] **Step 1: Reescrever o arquivo**

Replace todo o conteúdo de `scripts/process_single.py` por:

```python
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processor import AudioProcessor
from src.core.summarizer import OpenAIModel, GeminiModel
from src.utils.cli_args import build_parser, parse_extra
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

from config.settings import Config

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")


def build_summarizer(provider: str):
    """Instancia o modelo de sumarização conforme o provider."""
    if provider == "openai":
        return OpenAIModel(Config.OPENAI_API_KEY, Config.DEFAULT_SUMMARIZATION_MODEL)
    if provider == "gemini":
        return GeminiModel(Config.GEMINI_API_KEY, Config.DEFAULT_GEMINI_MODEL)
    raise ValueError(f"Provider desconhecido: {provider}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Arquivo não encontrado: {audio_path}")
        sys.exit(1)

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        logger.error(f"Prompt não encontrado: {prompt_path}")
        sys.exit(1)

    try:
        extra_files = parse_extra(args.extra)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    output_name = args.output or audio_path.stem

    try:
        # Transcrição e diarização
        processor = AudioProcessor(
            num_speakers=args.num_speakers,
            model_name=Config.DEFAULT_TRANSCRIPTION_MODEL,
            use_diarization=True,
            hf_token=Config.HF_TOKEN,
        )
        _, diarized_path = processor.process(
            file_path=audio_path,
            output_name=output_name,
            language=args.language,
        )
        processor.cleanup()

        # Sumarização
        summarizer = build_summarizer(args.provider)
        summarizer.generate(prompt_path, diarized_path, extra_files)

    except Exception as e:
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — `--help` lista as flags**

Run: `python scripts/process_single.py --help`
Expected: uso impresso com `--audio`, `--prompt`, `--output`, `--extra`, `--provider`, `--num-speakers`, `--language`; sai com código 0.

- [ ] **Step 3: Smoke test — áudio inexistente sai com erro (sem carregar modelos)**

Run: `python scripts/process_single.py --audio nao_existe.mp3 --prompt prompts/generic_summary.txt`
Expected: log `Arquivo não encontrado: nao_existe.mp3` e código de saída 1. (O prompt ainda não existe neste ponto, mas a checagem do áudio vem primeiro, então o erro é o do áudio.)

Verificar o exit code:
Run: `python scripts/process_single.py --audio nao_existe.mp3 --prompt x.txt; echo "exit=$?"`
Expected: `exit=1`.

- [ ] **Step 4: Rodar toda a suíte de testes (garantir que nada regrediu)**

Run: `python tests/test_cli_args.py -v && python tests/test_core_plumbing.py -v && python tests/test_summary_output.py -v`
Expected: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/process_single.py
git commit -m "feat: CLI parametrizada com flags explícitas e seleção de provider

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Prompt genérico versionado + doc de uso + verificação e2e

**Files:**
- Create: `prompts/generic_summary.txt`
- Modify: `README.md` (bloco de uso)

**Interfaces:**
- Consumes: a CLI da Task 4.
- Produces: prompt genérico versionado e instruções de uso.

- [ ] **Step 1: Criar o prompt genérico**

Create `prompts/generic_summary.txt`:

```
Você é um assistente que resume conteúdo de áudio ou vídeo transcrito e
diarizado (falantes identificados como SPEAKER_00, SPEAKER_01, etc.).

Analise a transcrição abaixo e produza um resumo claro e bem estruturado,
em português, com as seções:

1. Temas abordados — os assuntos principais tratados.
2. Pontos principais — as ideias, argumentos e informações mais relevantes.
3. Conclusões ou decisões — a que se chegou, ou o que ficou definido, se houver.
4. Trechos marcantes — falas ou momentos que se destacam.

Escreva de forma objetiva e fiel ao conteúdo. Não invente informações que não
estejam na transcrição.

Transcrição:
<arquivo_input>
```

- [ ] **Step 2: Confirmar que o prompt NÃO está gitignored**

Run: `git check-ignore prompts/generic_summary.txt; echo "ignored=$?"`
Expected: `ignored=1` (exit 1 = NÃO ignorado; nenhuma linha impressa). O `.gitignore` ignora `data` e `outputs/`, não `prompts/`.

- [ ] **Step 3: Adicionar bloco de uso no README**

Em `README.md`, acrescentar (no fim do arquivo) uma seção de uso da CLI:

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add prompts/generic_summary.txt README.md
git commit -m "feat: prompt genérico versionado e doc de uso da CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Verificação e2e manual (perfil genérico)**

Pré-requisitos: dependências instaladas e `.env` com `HF_TOKEN`, `OPENAI_API_KEY` (e `GEMINI_API_KEY` se usar `--provider gemini`).

Run (substituir pelo caminho de um áudio/vídeo público de tema qualquer):
```bash
python scripts/process_single.py --audio <arquivo_publico> --prompt prompts/generic_summary.txt
```
Expected:
- `outputs/diarizations/<nome>_diarized.txt` gerado (transcrição com falantes).
- `outputs/summaries/summary_<nome>_<modelo>.txt` gerado (resumo coerente com as 4 seções).
- Sem informar `--num-speakers` nem `--language`: nº de falantes e idioma detectados automaticamente (conferir no log `idioma=...` e `Número de falantes`).

- [ ] **Step 6: Verificação e2e manual (perfil ata — não regressão)**

Pré-requisito: arquivos locais `data/prompts/main.txt`, `data/prompts/context_ativos.txt` e um áudio de reunião em `data/input/`.

Run:
```bash
python scripts/process_single.py \
  --audio data/input/<audio_reuniao> \
  --prompt data/prompts/main.txt \
  --extra context_ativos=data/prompts/context_ativos.txt \
  --num-speakers 2 --language pt --provider openai
```
Expected: ata gerada em `outputs/summaries/summary_<nome>_<modelo>.txt`, com o `context_ativos` injetado, 2 falantes e idioma pt — conteúdo equivalente ao comportamento anterior.

---

## Self-Review

**Spec coverage:**
- CLI com flags explícitas → Task 1 (parser) + Task 4 (wiring). ✅
- `--provider {openai,gemini}` + fix nome Gemini → Task 3 (fix) + Task 4 (seleção). ✅
- Saída neutra `outputs/summaries/` → Task 3. ✅
- `num_speakers` None fluindo (remover coerção) → Task 2. ✅
- `language` None (default) + plumbing → Task 2. ✅
- Config defaults `None` + `DEFAULT_GEMINI_MODEL` + mkdir summaries → Task 3. ✅
- Prompt genérico versionado fora de `data/` → Task 5. ✅
- `HUBSPOT_TOKEN`/`.env`/`data/` seguem gitignored e intocados → nenhuma task mexe neles. ✅
- Testes de parsing (`auto`→None, `--extra`) e caminho de saída → Tasks 1 e 3. ✅
- Verificação e2e dos dois perfis → Task 5. ✅

**Placeholder scan:** sem "TBD"/"TODO"/"handle edge cases" genéricos; todo passo de código traz o código completo. ✅

**Type consistency:** `parse_num_speakers`/`parse_language`/`parse_extra`/`build_parser` (Task 1) usados idênticos na Task 4; `summary_output_path(audio_name, model_name)` (Task 3) idêntico no teste e no uso; `process(..., language=...)` (Task 2) consumido pela Task 4 com o mesmo nome. ✅
