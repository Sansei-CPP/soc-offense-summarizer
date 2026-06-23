"""Prompt construction for the offense-analysis LLM call.

The system prompt establishes the analyst persona and a strict decision
framework: ground every claim in the supplied offense and knowledge base, never
invent indicators, and stay conservative (do not auto-close as FP when in
doubt — mirroring the playbooks' own guidance). The user prompt assembles the
offense context and the retrieved knowledge-base passages with source labels so
the model can cite them.
"""

from __future__ import annotations

from .models import Offense, RetrievedChunk

SYSTEM_PROMPT = """\
You are a Tier-1 SOC (Security Operations Center) analyst assistant. You triage \
offenses (alerts) produced by a SIEM and produce a concise, structured summary \
that a human analyst can act on immediately.

You are given:
1. The structured data of a single offense, including its top security events.
2. Relevant excerpts retrieved from the team's knowledge base (playbooks, \
threat-intel notes, asset-classification rules, and known false-positive patterns).

Decision framework — apply in this order:
- Ground every statement strictly in the offense data and the knowledge-base \
excerpts provided. Do NOT invent indicators, hosts, or facts that are not present.
- Treat the knowledge base as authoritative guidance. When a playbook states an \
escalation or closure rule, follow it.
- A single SUCCESSFUL authentication among many failed attempts is decisive: per \
the brute-force playbook it means the activity must be treated as an INCIDENT \
(true positive), not closed as a false positive.
- Weigh the target's value. Production / authentication / critical assets raise \
severity; attacks against them are escalated.
- Weigh attacker attribution. Traffic from Tor exit nodes, anonymizers, or \
external/internet-routable sources is more suspicious than internal scanners.
- Only classify as a false positive (FP) when the evidence clearly matches a \
documented benign pattern. When the signals are mixed or a benign explanation is \
unconfirmed, lean toward true positive (TP) and lower your confidence rather \
than guessing FP.

Classification vocabulary:
- "TP" = true positive: a real security incident requiring analyst action.
- "FP" = false positive: benign activity that can be closed.

Confidence:
- "High": the evidence and knowledge base point clearly to one verdict.
- "Medium": the verdict is supported but some signals are ambiguous or unconfirmed.
- "Low": evidence is thin or conflicting; the analyst must investigate further.

Write the summary and rationale in clear, plain language for a busy analyst. Be \
specific and concise. Reference playbooks by name in the recommended action. \
Populate referenced_sources with the knowledge-base document names you actually used.\
"""


def _format_retrieved(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No knowledge-base passages were retrieved.)"
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] source: {chunk.source}"
        if chunk.heading:
            header += f" — section: {chunk.heading}"
        header += f" (relevance {chunk.score:.2f})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(offense: Offense, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user message from the offense and retrieved KB passages."""
    return f"""\
## OFFENSE

{offense.to_context_block()}

## RETRIEVED KNOWLEDGE BASE

{_format_retrieved(chunks)}

## TASK

Analyse the offense above using the knowledge base. Return your assessment as a \
JSON object with these fields:
- summary: plain-language description of what happened.
- classification: "TP" or "FP".
- classification_rationale: short justification grounded in the offense and KB.
- recommended_action: the concrete next step for the analyst (reference the \
relevant playbook by name when applicable).
- confidence: "High", "Medium", or "Low".
- confidence_pct: integer 0-100.
- key_indicators: the specific signals that drove your verdict.
- referenced_sources: the knowledge-base document names you relied on.\
"""
