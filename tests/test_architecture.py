import os
import unittest
from pathlib import Path
from unittest.mock import patch

from server.runtime import Readiness, VoiceRuntime
from server.settings import Settings
from server.llm.diagrams import normalize, should_generate


class ArchitectureTests(unittest.TestCase):
    def test_readiness_requires_tts_and_provider(self):
        readiness = Readiness(tts=True, provider=False)
        self.assertFalse(readiness.ready)
        readiness.provider = True
        self.assertTrue(readiness.ready)

    def test_runtime_owns_provider_clients(self):
        runtime = VoiceRuntime(llm_timeout=10, vision_timeout=10)
        self.assertIsNotNone(runtime.provider_clients)
        self.assertFalse(runtime.readiness.ready)

    def test_settings_resolve_reference_audio_relative_to_root(self):
        root = Path("/workspace/Voice_Cloning")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOICE_REF_AUDIO", None)
            settings = Settings.from_environment(root)
        self.assertEqual(settings.reference_audio, root / "my_voice.wav")

    def test_diagram_intent_and_normalization_are_bounded(self):
        self.assertTrue(should_generate("draw a flowchart", [], True))
        diagram = normalize({
            "elements": [
                {"id": "node", "type": "rectangle", "x": 0, "y": 0, "text": "x"},
                {"id": "edge", "type": "arrow", "x": 0, "y": 0, "startNodeId": "node", "endNodeId": "missing"},
            ]
        })
        self.assertEqual([item["id"] for item in diagram["elements"]], ["node"])


if __name__ == "__main__":
    unittest.main()
