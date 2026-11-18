from config.settings import Config

from pathlib import Path
from time import perf_counter
from datetime import datetime

from google.genai.types import GenerateContentConfig
from google.genai import types
from google import genai
from openai import OpenAI

from src.models.schemas import SummarizationResult
from src.utils.files import save_json_result, append_to_results_log, load_file,save_summarization_output
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

def prepare_prompt(prompt: str | Path, diarized_text: str | Path, extra_files: dict[str, str]) -> str:
    """Prepara o prompt substituindo os placeholders."""
    prompt_text = load_file(prompt).replace("<arquivo_input>", load_file(diarized_text))
    
    for placeholder, file_path in extra_files.items():
        prompt_text = prompt_text.replace(f"<{placeholder}>", load_file(file_path))
    
    return prompt_text


def summary_output_path(audio_name: str, model_name: str) -> Path:
    """Caminho neutro de saída da sumarização."""
    return Config.OUTPUT_DIR / "summaries" / f"summary_{audio_name}_{model_name}.txt"


def clean_output_name(output_name: str | None, diarized_text: str | Path) -> str:
    """Resolve o nome limpo de saída: usa output_name se fornecido, senão deriva
    do caminho do texto diarizado removendo o sufixo '_diarized' e a extensão."""
    return output_name or Path(diarized_text).stem.removesuffix("_diarized")


def save_results(generated_text: str, result: SummarizationResult, audio_name: str, model_name: str) -> None:
    """Salva os resultados em arquivos."""
    output_text = summary_output_path(audio_name, model_name)
    save_summarization_output(generated_text, output_text)

    output_json = Config.STATS_DIR  / f"summary_stats_{audio_name}_{model_name}.json"
    save_json_result(result, output_json)

    log_path = Config.LOGS_DIR / "processing_log.jsonl"
    append_to_results_log(result, log_path)

    print(f"Resumo gerado e salvo em {Config.OUTPUT_DIR}/summaries")


class OpenAIModel:
    """Model para OpenAI."""
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str | Path, diarized_text: str | Path, extra_files: dict[str, str] = None, output_name: str | None = None) -> dict:
        """Gera conteúdo usando OpenAI."""
        logger.info(f"========== Iniciando geração com OpenAI: {self.model_name} ==========")
        start_time = perf_counter()
        
        prompt_text = prepare_prompt(prompt, diarized_text, extra_files or {})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}]
            )
        except Exception as e:
            logger.error(f"========== Erro ao gerar conteúdo com OpenAI: {e} ==========")
            raise
        
        end_time = perf_counter()
        generated_text = response.choices[0].message.content
        
        result = SummarizationResult(
            summary_path="outputs/summaries/",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=self.model_name,
            summary_time_sec=end_time - start_time,
            timestamp=datetime.now().isoformat()
        )
        
        name = clean_output_name(output_name, diarized_text)
        save_results(generated_text, result, name, self.model_name)
        
        logger.info("========== Geração concluída ==========")
        return result.to_dict()

class GeminiModel:
    """Model para Gemini."""
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        candidate_count: int = 1
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_count
        )
    
    def generate(self, prompt: str | Path, diarized_text: str | Path, extra_files: dict[str, str] = None, output_name: str | None = None) -> dict:
        """Gera conteúdo usando Gemini."""
        logger.info(f"========== Iniciando geração com Gemini: {self.model_name} ==========")
        start_time = perf_counter()
        
        prompt_text = prepare_prompt(prompt, diarized_text, extra_files or {})
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt_text)]
                    )
                ],
                config=self.config
            )
        except Exception as e:
            logger.error(f"========== Erro ao gerar conteúdo com Gemini: {e} ==========")
            raise
        
        end_time = perf_counter()
        
        result = SummarizationResult(
            summary_path="outputs/summaries/",
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
            model=self.model_name,
            summary_time_sec=end_time - start_time,
            timestamp=datetime.now().isoformat()
        )
        
        name = clean_output_name(output_name, diarized_text)
        save_results(response.text, result, name, self.model_name)
        
        logger.info("========== Geração concluída ==========")
        return result.to_dict()