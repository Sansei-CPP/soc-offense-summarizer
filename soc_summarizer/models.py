"""Pydantic models for offense input and the structured AI summary output.

The input ``Offense`` model is intentionally permissive: real SIEM offenses vary
in shape, so every field is optional and unknown fields are preserved. The
output ``OffenseAnalysis`` model is the contract the LLM must satisfy — its JSON
schema is handed to Ollama for constrained decoding, so it is kept deliberately
flat (``Literal`` enums rather than nested ``$ref`` definitions) for maximum
compatibility with the grammar-constrained generation backend.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- Output vocabulary -------------------------------------------------------

Classification = Literal["TP", "FP"]
Confidence = Literal["High", "Medium", "Low"]


# --- Offense input -----------------------------------------------------------


class TopEvent(BaseModel):
    """A single representative security event attached to an offense."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    qid: Optional[int] = None
    low_level_category: Optional[str] = None
    high_level_category: Optional[str] = None
    sourceip: Optional[str] = None
    sourceport: Optional[int] = None
    destinationip: Optional[str] = None
    destinationport: Optional[int] = None
    username: Optional[str] = None
    event_outcome: Optional[str] = None
    protocol: Optional[str] = Field(default=None, alias="PROTOCOLNAME")
    count: Optional[int] = None

    def one_line(self) -> str:
        outcome = (self.event_outcome or "").upper()
        parts = [
            f"[{outcome or '?'}]",
            self.low_level_category or self.high_level_category or "event",
        ]
        if self.username:
            parts.append(f"user={self.username}")
        if self.destinationport is not None:
            parts.append(f"dport={self.destinationport}")
        if self.protocol:
            parts.append(self.protocol)
        if self.count is not None:
            parts.append(f"x{self.count}")
        return " ".join(parts)


class Offense(BaseModel):
    """A SIEM offense (alert). All fields optional; unknown fields preserved."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    description: Optional[str] = None
    magnitude: Optional[int] = None
    severity: Optional[int] = None
    credibility: Optional[int] = None
    relevance: Optional[int] = None

    offenseSource: Optional[str] = None
    formattedOffenseType: Optional[str] = None
    attacker: Optional[str] = None
    attackerDescription: Optional[str] = None
    target: Optional[str] = None
    targetDescription: Optional[str] = None
    targetNetwork: Optional[str] = None
    domainName: Optional[str] = None

    eventCount: Optional[int] = None
    eventDescription: Optional[str] = None
    categoryCount: Optional[int] = None
    attackerCount: Optional[int] = None
    targetCount: Optional[int] = None

    startTime: Optional[str] = None
    endTime: Optional[str] = None
    formattedDuration: Optional[str] = None

    top_events: list[TopEvent] = Field(default_factory=list)

    # -- helpers --------------------------------------------------------------

    def short_title(self) -> str:
        return self.description or f"Offense {self.id}" if self.id else "Unidentified offense"

    def to_retrieval_query(self) -> str:
        """Build a natural-language query that captures the offense's security
        signal, used to search the knowledge base semantically.

        We surface the things an analyst would search a playbook for: the rule
        description, offense type, attacker/target context, and — critically —
        the event categories, target usernames and outcomes (a single SUCCESS
        among many failures completely changes the verdict).
        """
        parts: list[str] = []
        if self.description:
            parts.append(self.description.replace("_", " "))
        if self.formattedOffenseType:
            parts.append(self.formattedOffenseType)
        if self.attackerDescription:
            parts.append(f"attacker {self.attackerDescription}")
        if self.targetDescription:
            parts.append(f"target {self.targetDescription}")
        if self.targetNetwork:
            parts.append(f"network {self.targetNetwork}")

        categories = sorted(
            {e.low_level_category for e in self.top_events if e.low_level_category}
            | {e.high_level_category for e in self.top_events if e.high_level_category}
        )
        if categories:
            parts.append("event categories: " + ", ".join(categories))

        outcomes = sorted({(e.event_outcome or "").lower() for e in self.top_events if e.event_outcome})
        if outcomes:
            parts.append("authentication outcomes: " + ", ".join(outcomes))
        if any((e.event_outcome or "").lower() == "success" for e in self.top_events):
            parts.append("successful authentication occurred")

        usernames = sorted({e.username for e in self.top_events if e.username})
        if usernames:
            parts.append("target usernames: " + ", ".join(usernames))

        ports = sorted({e.destinationport for e in self.top_events if e.destinationport is not None})
        if ports:
            parts.append("destination ports: " + ", ".join(str(p) for p in ports))

        return ". ".join(parts) if parts else "security offense triage"

    def to_context_block(self) -> str:
        """Render the offense as readable text for the LLM prompt."""
        lines: list[str] = []

        def add(label: str, value: Any) -> None:
            if value not in (None, ""):
                lines.append(f"- {label}: {value}")

        add("Offense ID", self.id)
        add("Description / rule", self.description)
        add("Offense type", self.formattedOffenseType)
        add("Magnitude", self.magnitude)
        add("Severity", self.severity)
        add("Credibility", self.credibility)
        add("Relevance", self.relevance)
        add("Attacker", self.attackerDescription or self.attacker)
        add("Target", self.targetDescription or self.target)
        add("Target network", self.targetNetwork)
        add("Domain", self.domainName)
        add("Event summary", self.eventDescription)
        add("Event count", self.eventCount)
        add("Time window", f"{self.startTime} -> {self.endTime}" if self.startTime else None)
        add("Duration", self.formattedDuration)

        if self.top_events:
            lines.append("- Top events:")
            for event in self.top_events:
                lines.append(f"    * {event.one_line()}")

        return "\n".join(lines)


# --- Structured output -------------------------------------------------------


class OffenseAnalysis(BaseModel):
    """The structured AI summary returned to the analyst.

    This schema is the LLM's output contract (enforced via Ollama structured
    outputs) and maps 1:1 to the deliverable requirements.
    """

    summary: str = Field(
        description="Plain-language summary of what happened in this offense, "
        "written for a SOC analyst."
    )
    classification: Classification = Field(
        description="Most likely classification: 'TP' (true positive / real "
        "incident) or 'FP' (false positive / benign)."
    )
    classification_rationale: str = Field(
        description="Short justification for the classification, grounded in the "
        "offense data and the knowledge base."
    )
    recommended_action: str = Field(
        description="The concrete next action the analyst should take, "
        "referencing the relevant playbook by name where applicable."
    )
    confidence: Confidence = Field(
        description="Confidence in the classification: High, Medium, or Low."
    )
    confidence_pct: int = Field(
        ge=0,
        le=100,
        description="Confidence as a percentage from 0 to 100.",
    )
    key_indicators: list[str] = Field(
        default_factory=list,
        description="The specific signals that drove the verdict (e.g. "
        "'successful auth from Tor exit node', 'target is a production asset').",
    )
    referenced_sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base document names that informed this analysis.",
    )


# --- Result wrapper ----------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A knowledge-base chunk returned by semantic search, with its score."""

    source: str
    heading: Optional[str] = None
    text: str
    score: float


class AnalysisResult(BaseModel):
    """Everything the pipeline produced for one offense.

    Bundles the LLM's structured analysis together with the retrieval evidence
    and the query used, so callers can display *why* the verdict was reached.
    """

    analysis: OffenseAnalysis
    retrieval_query: str
    retrieved_chunks: list[RetrievedChunk]
    llm_model: str
    embedding_model: str
