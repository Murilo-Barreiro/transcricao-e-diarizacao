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
