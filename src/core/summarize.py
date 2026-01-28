from importlib.resources import path
from time import perf_counter
from datetime import datetime
from anyio import Path

from google.genai.types import GenerateContentConfig
from google.genai import types
from google import genai
from openai import OpenAI

from src.models.schemas import SummarizationResult
from src.utils.files import save_json_result, append_to_results_log
from config.settings import Config
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class OpenAiModel:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    def _load_file(self, file_path: str | Path) -> str:
        """Carrega o conteúdo de um arquivo de texto."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _save_output(self, content: str, output_path: Path) -> None:
        """Salva o conteúdo gerado em um arquivo."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def generate(self, prompt: str | Path, diarized_text: str | Path, extra_files: dict[str, str]) -> dict:
        logger.info(f"========== Iniciando geração de resumo com modelo OpenAI: {self.model_name} ==========")
        
        st = perf_counter()
        _prompt_text = self._load_file(prompt).replace("<arquivo_input>", self._load_file(diarized_text))
        
        for placeholder, file_path in extra_files.items():
            _prompt_text = _prompt_text.replace(f"<{placeholder}>", self._load_file(file_path))

        client = OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": _prompt_text
                    }
                ]
            )

        except Exception as e:
            logger.error(f"========== Erro ao gerar conteúdo com OpenAI: {e} ==========")
            raise e
        
        logger.info("========== Geração de ata concluída ==========")

        et = perf_counter()
        
        # Extrair o texto da resposta
        generated_text = response.choices[0].message.content
        
        output_name = Config.OUTPUT_DIR / f"atas/ata_openai_{self.model_name}.txt"
        self._save_output(generated_text, output_name)

        result = SummarizationResult(
            summary_path=r"outputs/atas/",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=self.model_name,
            summary_time_sec=et - st,
            timestamp=datetime.now().isoformat()
        )

        output_json = Config.OUTPUT_DIR / "atas" / f"summary_result_openai_{self.model_name}.json"
        save_json_result(result, output_json)
        
        log_path = Config.OUTPUT_DIR / "processing_log.jsonl"
        append_to_results_log(result, log_path)

        print("Resumo gerado e salvo em output/atas")
    
        return result.to_dict()

class GeminiModel:
    def __init__(self, api_key:str, model_name:str):
        self.api_key = api_key
        self.model_name = model_name

    def _load_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _save_output(self, output: str, output_path: Path) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)

    def configure(self,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        candidate_count: int = 1
    ) -> GenerateContentConfig:

        config = GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k, 
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_count
        )

        return config

    def generate(self, prompt: str | Path, diarized_text: str | Path, extra_files: dict[str, str], config: GenerateContentConfig | None) -> str:
        logger.info(f"========== Iniciando geração de resumo com modelo Gemini: {self.model_name}==========")
        
        st = perf_counter()
        _prompt_text = self._load_file(prompt).replace("<arquivo_input>", self._load_file(diarized_text))
        
        for placeholder, file_path in extra_files.items():
            _prompt_text = _prompt_text.replace(f"<{placeholder}>", self._load_file(file_path))

        client = genai.Client(api_key=self.api_key)

        if config is None:
            config = GenerateContentConfig()

        try:
            response = client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=_prompt_text)]
                        )
                    ],
                    config=config
                )
        except Exception as e:
            logger.error(f"========== Erro ao gerar conteúdo com Gemini: {e} ==========")
            raise e
        
        logger.info("========== Geração de ata concluída ==========")

        et = perf_counter()
        
        output_name = Config.OUTPUT_DIR / "atas/ata_gemini.txt"
        self._save_output(response.text, output_name)

        result = SummarizationResult(
            summary_path=r"outputs/atas/",
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
            model=self.model_name,
            summary_time_sec= et - st
        )

        output_json = Config.OUTPUT_DIR / "atas" / f"summary_result_gemini.json"
        save_json_result(result, output_json)
        
        log_path = Config.OUTPUT_DIR / "processing_log.jsonl"
        append_to_results_log(result, log_path)

        print("Resumo gerado e salvo em output/atas")
    
        return result.to_dict()
    