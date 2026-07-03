"""Unit tests for Improvement B: judge_backends (no network / no models needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval.judge_backends import (   # noqa: E402
    build_judge, NoneJudge, BaseJudge, JudgeVerdict, extract_tag,
)


def test_extract_tag():
    txt = "<State>suspicious</State>\n<Reasoning>has a url</Reasoning>"
    assert extract_tag(txt, "State") == "suspicious"
    assert extract_tag(txt, "Reasoning") == "has a url"
    assert extract_tag("no tags here", "State") is None


def test_none_backend_accepts_everything():
    j = build_judge("none")
    assert isinstance(j, NoneJudge)
    assert j.judge("Click <malicious_url>").is_suspicious is True
    assert j.judge("totally benign text").is_suspicious is True


def test_unknown_backend_raises():
    try:
        build_judge("banana")
        assert False, "should have raised"
    except ValueError:
        pass


class FakeJudge(BaseJudge):
    """Backend that returns a canned raw completion, to test parsing + retry."""
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def _raw_complete(self, prompt):
        self.calls += 1
        return self.raw


def test_parsing_suspicious():
    j = FakeJudge("<State>suspicious</State><Reasoning>destructive command</Reasoning>")
    v = j.judge("rm -rf /db")
    assert v.is_suspicious and "destructive" in v.reasoning


def test_parsing_safe():
    j = FakeJudge("<State>safe</State><Reasoning>ordinary summary</Reasoning>")
    v = j.judge("Here is the news summary")
    assert v.is_suspicious is False


def test_malformed_then_retry_exhausts():
    j = FakeJudge("garbage with no tags")
    v = j.judge("x", max_retries=2, retry_delay=0.0)
    assert v.is_suspicious is False          # safe fallback
    assert j.calls == 2                       # retried


def test_local_judge_mocked():
    from unittest.mock import patch, MagicMock
    with patch("transformers.pipelines.pipeline") as mock_pipeline:
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = [{"generated_text": "<State>suspicious</State><Reasoning>contains toxic content</Reasoning>"}]
        mock_pipeline.return_value = mock_pipe_instance
        
        j = build_judge("local", model="test-model")
        
        mock_pipeline.assert_called_once_with(
            "text-generation", model="test-model",
            device_map="auto", max_new_tokens=256
        )
        
        verdict = j.judge("harmful text", max_retries=1)
        assert verdict.is_suspicious is True
        assert verdict.reasoning == "contains toxic content"
        mock_pipe_instance.assert_called_once()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
