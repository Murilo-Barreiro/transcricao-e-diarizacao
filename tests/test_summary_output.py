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
