"""Tests for offense parsing, retrieval-query building, and context rendering."""

from __future__ import annotations

from soc_summarizer.models import Offense


def test_offense_parses_permissively_and_keeps_unknown_fields(sample_offense):
    offense = Offense.model_validate(sample_offense)
    assert offense.id == 48291
    assert offense.targetNetwork == "PROD_SERVERS"
    assert len(offense.top_events) == 3
    # PROTOCOLNAME alias is honoured.
    assert offense.top_events[0].protocol == "TCP"


def test_retrieval_query_surfaces_decisive_signals(sample_offense):
    query = Offense.model_validate(sample_offense).to_retrieval_query()
    lowered = query.lower()
    # The decisive facts for triage must appear in the query.
    assert "success" in lowered  # the one successful auth
    assert "brute" in lowered  # rule/description
    assert "root" in lowered and "admin" in lowered  # targeted usernames
    assert "22" in query  # SSH destination port


def test_context_block_is_human_readable(sample_offense):
    block = Offense.model_validate(sample_offense).to_context_block()
    assert "auth-server-prod-01" in block
    assert "Top events:" in block
    assert "deploy-svc" in block  # the successful-login username is shown


def test_minimal_offense_does_not_crash():
    offense = Offense.model_validate({"id": 1})
    assert offense.to_retrieval_query()  # non-empty fallback
    assert "Offense ID: 1" in offense.to_context_block()
