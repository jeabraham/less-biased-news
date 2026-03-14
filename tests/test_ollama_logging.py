#!/usr/bin/env python3
"""
Unit tests for _call_ollama_api() in AIUtils.

Covers:
- stream=False is passed to ollama_client.generate()
- model name and prompt length are included in the INFO log line
- empty response triggers a WARNING that includes model, prompt_length, stream, and raw_body
- non-empty response produces no WARNING
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Minimal stubs so ai_utils can be imported without heavy ML dependencies
# ---------------------------------------------------------------------------
for _mod in ("tiktoken", "psutil", "torch", "transformers", "openai", "ollama"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide just enough torch stubs
torch_stub = sys.modules["torch"]
torch_stub.cuda.is_available.return_value = False
torch_stub.backends.mps.is_available.return_value = False

# Provide tiktoken stub that returns an encoder
tiktoken_stub = sys.modules["tiktoken"]
tiktoken_stub.encoding_for_model.return_value = MagicMock()

import ai_utils  # noqa: E402


def _make_ai_utils(response_text: str = "yes") -> tuple:
    """
    Return (AIUtils instance, mock ollama_client) wired for Ollama-only use.
    """
    cfg = {
        "ollama": {
            "enabled": True,
            "base_url": "http://localhost:11434",
            "model": "test-model",
            "temperature": 0.1,
        }
    }

    mock_client = MagicMock()

    # Build a mock response whose .get() behaves like a dict and supports dict() conversion
    response_dict = {"response": response_text, "done": True}
    mock_response = MagicMock()
    mock_response.get.side_effect = response_dict.get
    mock_response.keys.return_value = response_dict.keys()
    mock_response.__getitem__ = MagicMock(side_effect=response_dict.__getitem__)
    mock_response.__iter__ = MagicMock(return_value=iter(response_dict.keys()))
    mock_client.generate.return_value = mock_response

    # Patch ollama.Client in ai_utils so __init__ wires our mock
    with patch.object(sys.modules["ollama"], "Client", return_value=mock_client):
        ai_utils.OLLAMA_AVAILABLE = True
        instance = ai_utils.AIUtils(cfg)
        instance.ollama_client = mock_client

    return instance, mock_client


class TestCallOllamaApiStreamFalse(unittest.TestCase):
    """stream=False must be forwarded to ollama_client.generate()."""

    def test_stream_false_passed(self):
        utils, mock_client = _make_ai_utils("hello")
        utils._call_ollama_api("test prompt")
        _, kwargs = mock_client.generate.call_args
        self.assertIn("stream", kwargs, "stream keyword must be explicitly passed")
        self.assertEqual(kwargs["stream"], False, "stream must be False")


class TestCallOllamaApiInfoLog(unittest.TestCase):
    """INFO log must contain model name, prompt_length, and stream: False."""

    def test_info_log_contains_expected_fields(self):
        utils, _ = _make_ai_utils("some result")
        prompt = "This is a test prompt of known length."
        with self.assertLogs("ai_utils", level="INFO") as cm:
            utils._call_ollama_api(prompt, task=None)

        info_lines = [l for l in cm.output if "INFO" in l and "Calling Ollama API" in l]
        self.assertTrue(info_lines, "Expected at least one INFO log about Calling Ollama API")
        log_line = info_lines[0]
        self.assertIn("test-model", log_line)
        self.assertIn(str(len(prompt)), log_line)
        self.assertIn("stream: False", log_line)


class TestCallOllamaApiEmptyResponseWarning(unittest.TestCase):
    """Empty response must emit a WARNING containing model, prompt_length, stream, raw_body."""

    def test_warning_on_empty_response(self):
        utils, _ = _make_ai_utils("")

        prompt = "classify this article"
        with self.assertLogs("ai_utils", level="WARNING") as cm:
            result = utils._call_ollama_api(prompt)

        self.assertEqual(result, "")
        warning_lines = [l for l in cm.output if "WARNING" in l and "empty response" in l]
        self.assertTrue(warning_lines, "Expected a WARNING about empty response")
        warn = warning_lines[0]
        self.assertIn("test-model", warn)
        self.assertIn(str(len(prompt)), warn)
        self.assertIn("stream: False", warn)
        self.assertIn("raw_body", warn)

    def test_no_warning_on_non_empty_response(self):
        utils, _ = _make_ai_utils("non-empty answer")
        prompt = "classify this"
        with self.assertLogs("ai_utils", level="DEBUG") as cm:
            result = utils._call_ollama_api(prompt)

        self.assertEqual(result, "non-empty answer")
        warning_lines = [l for l in cm.output if "WARNING" in l and "empty response" in l]
        self.assertFalse(warning_lines, "No empty-response WARNING expected for non-empty result")


if __name__ == "__main__":
    unittest.main()
