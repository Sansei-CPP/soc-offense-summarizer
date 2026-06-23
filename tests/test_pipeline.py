"""End-to-end pipeline tests using fake (Ollama-free) backends."""

from __future__ import annotations

import pytest

from soc_summarizer.models import AnalysisResult, OffenseAnalysis
from soc_summarizer.pipeline import AnalysisError, OffenseSummarizer


def test_index_built_at_startup(settings, lexical_embedder, fake_llm):
    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=fake_llm)
    assert summarizer.document_count == 4
    assert summarizer.chunk_count >= 4


def test_retrieval_surfaces_relevant_playbook(settings, lexical_embedder, fake_llm, sample_offense):
    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=fake_llm)
    query, chunks = summarizer.retrieve(sample_offense)

    assert len(chunks) == settings.top_k
    sources = {c.source for c in chunks}
    # The brute-force offense should pull in the brute-force playbook.
    assert "brute_force_playbook.md" in sources
    # Scores are sorted descending.
    assert [c.score for c in chunks] == sorted((c.score for c in chunks), reverse=True)


def test_analyze_returns_validated_structured_result(settings, lexical_embedder, fake_llm, sample_offense):
    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=fake_llm)
    result = summarizer.analyze(sample_offense)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.analysis, OffenseAnalysis)
    assert result.analysis.classification in {"TP", "FP"}
    assert 0 <= result.analysis.confidence_pct <= 100
    assert result.retrieved_chunks  # evidence carried through
    assert result.llm_model == settings.llm_model


def test_llm_receives_system_user_and_schema(settings, lexical_embedder, fake_llm, sample_offense):
    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=fake_llm)
    summarizer.analyze(sample_offense)

    # The offense and retrieved evidence reach the model.
    assert "auth-server-prod-01" in fake_llm.last_user
    assert fake_llm.last_system and "SOC" in fake_llm.last_system
    # The schema handed to the model constrains the classification field.
    props = fake_llm.last_schema["properties"]
    assert set(props["classification"]["enum"]) == {"TP", "FP"}
    assert "summary" in props and "recommended_action" in props


def test_invalid_llm_output_raises_analysis_error(settings, lexical_embedder, sample_offense):
    class BrokenLLM:
        def generate_json(self, system, user, schema):
            return "not json at all"

    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=BrokenLLM())
    with pytest.raises(AnalysisError):
        summarizer.analyze(sample_offense)


def test_code_fenced_json_is_tolerated(settings, lexical_embedder, sample_offense):
    import json

    from tests.conftest import _FAKE_ANALYSIS

    class FencedLLM:
        def generate_json(self, system, user, schema):
            return "```json\n" + json.dumps(_FAKE_ANALYSIS) + "\n```"

    summarizer = OffenseSummarizer(settings, embedder=lexical_embedder, llm=FencedLLM())
    result = summarizer.analyze(sample_offense)
    assert result.analysis.classification == "TP"
