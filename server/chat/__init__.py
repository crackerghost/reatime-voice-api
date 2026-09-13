"""Streaming chat text pipeline: speakable sentences and TTS window units.

Pure functions only — no app state, no GPU, no network. Importable and
unit-testable in isolation (see tests/test_phrases.py). This is the first
extraction out of the app.py god file; the LLM-stream and worker threads
(which need live config) follow the same pattern next.
"""
