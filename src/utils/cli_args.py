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
