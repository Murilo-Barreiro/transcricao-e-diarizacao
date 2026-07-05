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
