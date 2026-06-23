"""Tests for prompt assembly."""

from __future__ import annotations

from soc_summarizer.models import Offense, RetrievedChunk
from soc_summarizer.prompts import SYSTEM_PROMPT, build_user_prompt


def test_system_prompt_encodes_decision_rules():
    text = SYSTEM_PROMPT.lower()
    assert "successful authentication" in text  # the decisive brute-force rule
    assert "false positive" in text and "true positive" in text
    assert "knowledge base" in text


def test_user_prompt_includes_offense_and_evidence(sample_offense):
    offense = Offense.model_validate(sample_offense)
    chunks = [
        RetrievedChunk(
            source="brute_force_playbook.md",
            heading="Escalate as INCIDENT if",
            text="There is at least one successful authentication from the brute-force source IP.",
            score=0.81,
        ),
        RetrievedChunk(
            source="tor_exit_nodes.md",
            heading=None,
            text="Successful auth from a Tor exit node must be escalated immediately.",
            score=0.74,
        ),
    ]
    prompt = build_user_prompt(offense, chunks)

    # Offense context is present.
    assert "auth-server-prod-01" in prompt
    # Retrieved evidence is present, with source labelling for citation.
    assert "brute_force_playbook.md" in prompt
    assert "successful authentication from the brute-force source IP" in prompt
    assert "source:" in prompt
    # Task framing is present.
    assert "## TASK" in prompt
    assert "classification" in prompt


def test_user_prompt_handles_no_chunks(sample_offense):
    offense = Offense.model_validate(sample_offense)
    prompt = build_user_prompt(offense, [])
    assert "No knowledge-base passages" in prompt
