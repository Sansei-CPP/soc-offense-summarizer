"""SOC offense summarizer.

A small Retrieval-Augmented-Generation service that turns a raw SIEM offense
(alert) into a structured, analyst-ready AI summary. Knowledge base, embeddings,
and the LLM all run locally through Ollama.
"""

from .config import Settings
from .models import Offense, OffenseAnalysis, AnalysisResult
from .pipeline import OffenseSummarizer

__all__ = [
    "Settings",
    "Offense",
    "OffenseAnalysis",
    "AnalysisResult",
    "OffenseSummarizer",
]

__version__ = "1.0.0"
