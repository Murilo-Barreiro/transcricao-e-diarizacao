from importlib.resources import path
from anyio import Path
from google.genai.types import GenerateContentConfig
from google.genai import types
from google import genai


class OpenAIModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

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

    def configure(
        self,
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
        _prompt_text = self._load_file(prompt).replace("<arquivo_input>", self._load_file(diarized_text))
        

        
        for placeholder, file_path in extra_files.items():
            _prompt_text = _prompt_text.replace(f"<{placeholder}>", self._load_file(file_path))

        client = genai.Client(api_key=self.api_key)

        if config is None:
            config = GenerateContentConfig()

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
        
        self._save_output(response.text, Path(r"outputs/atas/ata_gemini.txt"))

        print("Resumo gerado e salvo em output/atas")