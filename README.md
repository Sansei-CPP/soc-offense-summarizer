# SOC Offense Summarizer

An AI service that automates the **first step** of SOC alert triage. Given a raw
SIEM **offense** (alert) as JSON, it runs a Retrieval-Augmented-Generation (RAG)
pipeline over a security knowledge base and returns a structured, analyst-ready
summary: what happened, whether it is a **true** or **false positive** (with a
short justification), the recommended action / playbook, and a confidence level.

Everything runs **locally through [Ollama](https://ollama.com)** — both the
embedding model and the LLM — so there are no API keys and nothing leaves the
machine. This matters for a SOC tool: offenses contain sensitive infrastructure
detail.

---

## What it does (maps 1:1 to the assignment)

| Requirement | Where |
|---|---|
| 1. Accept an offense JSON object | [`cli.py`](soc_summarizer/cli.py), [`models.Offense`](soc_summarizer/models.py) |
| 2. Load KB docs → **chunk** → **embed** → store in a vector store **at startup** | [`knowledge_base.py`](soc_summarizer/knowledge_base.py), [`embeddings.py`](soc_summarizer/embeddings.py), [`vector_store.py`](soc_summarizer/vector_store.py) |
| 3. **Semantic search** using the offense as the query | [`retriever.py`](soc_summarizer/retriever.py), [`models.Offense.to_retrieval_query`](soc_summarizer/models.py) |
| 4. Assemble a prompt from the offense + retrieved chunks | [`prompts.py`](soc_summarizer/prompts.py) |
| 5. Call the LLM, return **structured** output (summary / TP-FP + rationale / action / confidence) | [`llm.py`](soc_summarizer/llm.py), [`models.OffenseAnalysis`](soc_summarizer/models.py) |

The vector store is **populated at startup from the documents on disk** — never
hardcoded. Point `SOC_KB_DIR` at any folder of `.md`/`.txt` files and it re-indexes.

---

## Architecture

```
                          offense.json
                               │
                               ▼
                     ┌───────────────────┐
   data/             │  Offense (parsed) │
   knowledge_base/   └─────────┬─────────┘
       │                       │ to_retrieval_query()
       │ load + chunk          ▼
       │ (heading-aware)  "brute force … successful auth … prod auth server …"
       ▼                       │
  ┌──────────┐   embed    ┌────┴─────┐  embed query   ┌──────────────┐
  │  chunks  │──────────▶ │  vector  │ ◀───────────── │   Ollama     │
  └──────────┘  (Ollama)  │  store   │  cosine top-k  │ nomic-embed  │
                          └────┬─────┘                └──────────────┘
                               │ retrieved passages
                               ▼
                     ┌───────────────────┐   system + user prompt
                     │   prompt builder  │ ───────────────────────┐
                     └───────────────────┘                        ▼
                                                          ┌────────────────┐
                          OffenseAnalysis  ◀──────────────│  Ollama LLM    │
                          (validated, structured)         │  qwen2.5:14b   │
                          JSON-schema-constrained         └────────────────┘
```

The whole pipeline is built around small, swappable pieces — `Embedder` and
`LLMClient` are protocols, so the Ollama backends can be replaced (or faked in
tests) without touching the pipeline.

---

## Requirements

- **Python 3.10+** (developed and tested on 3.11)
- **[Ollama](https://ollama.com)** running locally, with two models pulled:
  - an embedding model — **`nomic-embed-text`** (multilingual; the KB here is in Russian, queries in English — retrieval still works)
  - a chat model — **`qwen2.5:14b`** (any instruction-following Ollama model that supports structured outputs works; override with `--model`)

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:14b
ollama serve            # if not already running as a service
```

---

## Setup

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate soc-summarizer
pip install -e .          # installs the `soc-summarize` command
```

### Option B — venv / pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Usage

```bash
# Analyse the bundled sample offense (pretty text output)
soc-summarize --offense examples/offense_sample.json

# Read an offense from stdin, emit machine-readable JSON
cat examples/offense_sample.json | soc-summarize --format json

# Inspect the RAG step only — query + retrieved passages, no LLM call
soc-summarize --offense examples/offense_sample.json --dry-run

# Override models / retrieval depth
soc-summarize -f offense.json --model qwen2:7b --top-k 6
```

You can also run it as a module: `python -m soc_summarizer --offense …`.

### Example output (bundled sample offense)

The sample is a brute-force offense from a **Tor exit node** against
`auth-server-prod-01` with **one successful login** among hundreds of failures —
a textbook escalation:

```
SUMMARY
  The offense involves a brute-force attack against an authentication server
  (auth-server-prod-01.internal) from a Tor exit node IP (185.220.101.47).
  The attacker attempted generic usernames such as 'root' and 'admin', with one
  successful login by the user 'deploy-svc'.

CLASSIFICATION : TRUE POSITIVE (incident)
CONFIDENCE     : High (95%) ●●●

RATIONALE
  A single successful authentication among many failures. Per
  brute_force_playbook.md this indicates an incident and must not be closed as
  a false positive.

RECOMMENDED ACTION
  Escalate as an INCIDENT per the Brute Force Playbook; review the deploy-svc
  account for unauthorized access.

KEY INDICATORS
  - Source IP is a Tor exit node (185.220.101.47)
  - Target host is a production authentication server
  - Multiple failed SSH logins for generic admin accounts
  - Single successful SSH login by 'deploy-svc'

REFERENCED PLAYBOOKS / SOURCES
  - brute_force_playbook.md
  - tor_exit_nodes.md
```

---

## Design notes

- **Heading-aware chunking.** Markdown is split on `#`/`##` headings so each
  playbook section (e.g. *"Escalate as INCIDENT if…"*) stays intact; the heading
  is folded into the embedded text so the section topic is encoded. Oversized
  sections are split by size with character overlap.
- **Asymmetric embedding prefixes.** `nomic-embed-text` was trained with
  `search_document:` / `search_query:` task prefixes; the embedder applies them
  automatically, which measurably improves retrieval.
- **Offense → query.** Rather than dumping raw JSON into the embedder, the
  offense is distilled into a natural-language query that foregrounds the
  triage-decisive signals (event categories, **the presence of a successful
  auth**, target usernames, ports, attacker/target descriptions).
- **In-memory vector store.** A NumPy cosine-similarity store, rebuilt at
  startup. For a KB this size an external vector DB would be over-engineering
  (the assignment explicitly says not to over-complicate).
- **Constrained structured output.** The LLM is given the `OffenseAnalysis`
  JSON schema via Ollama's `format` parameter, so it is grammar-constrained to
  emit valid JSON, which is then validated with Pydantic.
- **Conservative, grounded prompting.** The system prompt encodes the playbooks'
  own decision rules (a successful auth ⇒ incident; critical/production assets
  escalate; Tor/external sources are suspicious) and instructs the model to lean
  TP with lower confidence when signals are mixed — never to guess "false
  positive". It must cite the KB sources it used.

---

## Configuration

All settings are environment variables with sane defaults (see
[`.env.example`](.env.example) and [`config.py`](soc_summarizer/config.py)):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `SOC_LLM_MODEL` | `qwen2.5:14b` | chat model |
| `SOC_EMBEDDING_MODEL` | `nomic-embed-text` | embedding model |
| `SOC_KB_DIR` | `data/knowledge_base` | knowledge-base folder |
| `SOC_CHUNK_SIZE` / `SOC_CHUNK_OVERLAP` | `800` / `120` | chunking (chars) |
| `SOC_TOP_K` | `4` | passages retrieved |
| `SOC_TEMPERATURE` | `0.1` | low → stable verdicts |
| `SOC_NUM_CTX` | `8192` | model context window |

---

## Tests

The suite uses a deterministic lexical embedder and a fake LLM, so it runs
**without Ollama** in well under a second:

```bash
pytest
```

It covers chunking, the vector store's cosine ranking, offense→query building,
prompt assembly, and the full pipeline (including schema validation and
malformed-output handling).

---

## Project structure

```
soc-offense-summarizer/
├── soc_summarizer/
│   ├── config.py          # env-driven settings
│   ├── models.py          # Offense (input) + OffenseAnalysis (output) schemas
│   ├── knowledge_base.py  # load + heading-aware chunking
│   ├── embeddings.py      # Embedder protocol + Ollama embedder
│   ├── vector_store.py    # in-memory cosine similarity store
│   ├── retriever.py       # build index + semantic search
│   ├── prompts.py         # system + user prompt construction
│   ├── llm.py             # LLMClient protocol + Ollama structured-output client
│   ├── pipeline.py        # orchestration (build at startup → analyze)
│   └── cli.py             # command-line interface
├── data/knowledge_base/   # the 4 provided KB documents
├── examples/offense_sample.json
└── tests/
```

---

## Notes & possible extensions

- **Scope.** Per the brief, there is intentionally no UI, database, auth, or
  deployment pipeline — the focus is the core RAG + structured-summary functionality.
- **Swapping backends.** Because `Embedder` and `LLMClient` are protocols, adding
  an OpenAI/Anthropic backend (or a persistent vector DB like Chroma/FAISS) is a
  drop-in change to one module.
- **Next steps** for production: a reranking stage, an IOC-enrichment tool the LLM
  can call (live Tor/threat-intel lookups), and feedback capture so analyst
  overrides tune retrieval/prompting over time.
```
