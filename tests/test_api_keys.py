"""Tests for the startup API-key check (no network, no keys needed)."""

import subprocess
import sys
from pathlib import Path

import pytest

import agent

REPO_ROOT = Path(agent.__file__).parent


@pytest.mark.parametrize("anthropic_key, freesound_key, expected", [
    (None, "fs-key", "ANTHROPIC_API_KEY"),
    ("sk-key", None, "FREESOUND_API_KEY"),
    ("", "", "ANTHROPIC_API_KEY, FREESOUND_API_KEY"),
])
def test_missing_keys_exit_with_clear_message(monkeypatch, anthropic_key, freesound_key, expected):
    monkeypatch.setattr(agent, "ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setattr(agent, "FREESOUND_API_KEY", freesound_key)
    with pytest.raises(SystemExit) as exc:
        agent.check_api_keys()
    assert str(exc.value).startswith(f"Error: missing {expected}.")


def test_present_keys_pass(monkeypatch):
    monkeypatch.setattr(agent, "ANTHROPIC_API_KEY", "sk-key")
    monkeypatch.setattr(agent, "FREESOUND_API_KEY", "fs-key")
    agent.check_api_keys()  # does not exit


def test_cli_without_keys_prints_message_not_traceback():
    # Empty values count as set, so load_dotenv() will not fill them from .env.
    env = {"PATH": "", "ANTHROPIC_API_KEY": "", "FREESOUND_API_KEY": ""}
    proc = subprocess.run(
        [sys.executable, "-c", "import agent; agent.main()"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert "Error: missing ANTHROPIC_API_KEY, FREESOUND_API_KEY." in proc.stderr
    assert "Traceback" not in proc.stderr
