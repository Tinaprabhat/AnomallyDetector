# AnomalyDetector v2

**A self-improving fraud detection system for blockchain transactions, combining a four-model ML ensemble with retrieval-augmented LLM reasoning.**

Built on the Elliptic Bitcoin Dataset. Designed for CPU-only inference (no GPU required). Every layer is independently testable, and the system grades its own routing logic against a 50-case rubric on every run.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [What Makes It Novel](#what-makes-it-novel)
3. [Architecture](#architecture)
4. [The Detection Pipeline (Stage by Stage)](#the-detection-pipeline-stage-by-stage)
5. [Layer-by-Layer Explanation](#layer-by-layer-explanation)
6. [Key Design Decisions and Why](#key-design-decisions-and-why)
7. [Quick Start](#quick-start)
8. [Folder Structure](#folder-structure)
9. [Tech Stack](#tech-stack)
10. [Testing and Evaluation](#testing-and-evaluation)
11. [Backtesting Results](#backtesting-results)
12. [What Phase 0 Delivers](#what-phase-0-delivers)
13. [Honest Limitations](#honest-limitations)
14. [Future Work](#future-work)

---

## What This Project Is

AnomalyDetector v2 detects suspicious or fraudulent Bitcoin transactions through a layered pipeline:

1. **Detect** — A four-model ML ensemble produces individual anomaly scores.
2. **Validate** — Per-instance statistical methods (Mahalanobis distance, SHAP attributions, ensemble agreement) provide model-independent evidence.
3. **Route** — A confidence router classifies each flagged case into one of five tiers (PASS, HIGH, MEDIUM, LOW, AMBIGUOUS) and dispatches it to the right explanation strategy.
4. **Retrieve** — A four-collection RAG knowledge base provides grounding context (typology library, case history, regulatory framework, personal notes).
5. **Reason** — Depending on the routing tier, a deterministic template, a local SLM (Ollama), or a hosted LLM (Mistral) produces a structured, auditable explanation with citations.
6. **Learn** — Confirmed cases enter case history; the LLM performs self-reflection to distill meta-lessons into a personal-notes collection that future calls can draw on.

The result is a system that flags fraud, explains why with regulatory citations, and continuously improves its institutional memory from analyst feedback.

This is a fork-and-rebuild of v1, with every architectural decision logged in `PROJECT_LOG.md`. v1 had several silent correctness bugs (random splits causing future leakage, Isolation Forest binary outputs treated as continuous scores, hardcoded metrics in main.py) that v2 fixes from the ground up.

---

## What Makes It Novel

Most fraud detection systems stop at "model returns a score." This one is structured around three ideas that go beyond that:

### 1. Knowledge-grounded explanation, not free-form generation

The LLM does not invent fraud typologies. Every classification it produces must come from a curated **typology library** of ten known fraud patterns (peel chain, layering, smurfing, mixing, mule account, rapid cashout, ransomware payment, dusting attack, darknet market settlement, exchange arbitrage lookalike). If none fit, it must return `no_match`. This is enforced through prompt design and validated by an automatic rubric that flags any hallucinated typology.

This corresponds to **Level 2 LLM autonomy**: the LLM synthesizes narratives and selects from a bounded option space, but cannot make detection decisions, invent categories, or modify the knowledge base.

### 2. Four-collection RAG with weighted retrieval

Instead of a single document store, the system maintains four specialised collections, each weighted differently in retrieval:

| Collection | Weight | Role |
|---|---|---|
| `typology_library` | 1.5× | Curated fraud patterns. Static. Hand-edited only. |
| `case_history` | 1.0× | Confirmed past cases. Grows from analyst feedback. |
| `regulatory` | 1.0× | FATF, OFAC, FinCEN, MiCA text. Compliance citations. |
| `personal_notes` | 1.2× | LLM-distilled meta-lessons. Grows automatically. |

Hybrid retrieval combines the text query (built from SHAP features, Mahalanobis distance, ensemble percentile) with the GNN node embedding when available, so we get both *what does this look like* and *who is this connected to*.

### 3. Self-improvement through constrained self-reflection

When an analyst confirms a flagged case, the LLM is asked to study the full case context — the ML evidence, what RAG retrieved, what the LLM originally said, what the analyst actually concluded — and produce a structured *meta-lesson*: a topic, a key lesson in 1–2 sentences, indicators to watch for, and which typologies the lesson applies to.

These meta-lessons live in the `personal_notes` collection, retrieved with weight 1.2× on every future flagged case. The system grows its own institutional memory of "what this analyst's organization has learned to look for."

Strict guardrails: the LLM never auto-edits typologies, never auto-adds case history, and rejected cases are routed to a separate false-positive log (not into RAG) so analyst rejections never poison the knowledge base.

### 4. Tiered routing prevents LLM overuse

Every flagged case is classified into one of five tiers by a deterministic router:

- **PASS** (≤ 1 detector flagged): no LLM, returned as normal
- **HIGH** (4 detectors agree + strong RAG match): template fill-in, no LLM call
- **MEDIUM** (≥ 3 detectors agree): local SLM (Ollama qwen2.5:1.5b)
- **LOW** (≥ 2 detectors agree): API LLM (Mistral)
- **AMBIGUOUS** (high score spread regardless of agreement): API LLM + human flag

Result: cheap deterministic explanations for clear cases, expensive LLM reasoning only when actually needed. This makes the system economically viable at scale and provides a fallback chain that degrades gracefully when the API is unreachable.

### 5. Provider abstraction with multi-stage fallback

The reasoning layer uses a `FallbackChain` of LLM providers:

```
Mistral Small 3.1  →  Mistral Medium 3  →  Ollama qwen2.5:3b (local)
```

If the API is rate-limited, down, or returning errors, the system silently falls through to the local model. Every provider implements the same `LLMProvider` interface, so swapping or adding providers is trivial. This was a hard-learned lesson from a prior project that hit Groq and Gemini rate limits during demos.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       INPUT: Bitcoin Transaction                              │
│   tx_id · 165 raw features · time_step · graph context (edges)                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                    │
            ▼                                                    ▼
┌─────────────────────────────┐                ┌────────────────────────────────┐
│  FEATURE PIPELINE           │                │  GRAPH PIPELINE                │
│  StandardScaler (+ PCA)     │                │  Subgraph extraction           │
└─────────────────────────────┘                └────────────────────────────────┘
            │                                                    │
            ▼                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DETECTION LAYER (4 models)                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│  │  XGBoost   │  │  GraphSAGE  │  │ Isolation Forest │  │   Autoencoder     ││
│  │ supervised │  │  graph GNN  │  │   unsupervised   │  │  reconstruction   ││
│  │  P(fraud)  │  │  P(fraud)   │  │  isolation score │  │       error       ││
│  └────────────┘  └─────────────┘  └──────────────────┘  └──────────────────┘│
│                            │                                                  │
│                            ▼                                                  │
│           Optuna-tuned weighted combination + agreement count                 │
│                  ensemble_score, agreement (0–4)                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       STATISTICAL VALIDATION LAYER                            │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ Mahalanobis distance │  │  SHAP top-K     │  │ Ensemble agreement      │ │
│  │  (vs. licit cluster) │  │  attributions   │  │   stats + spread         │ │
│  └──────────────────────┘  └──────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CONFIDENCE ROUTER                                    │
│           PASS  ·  HIGH  ·  MEDIUM  ·  LOW  ·  AMBIGUOUS                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      RAG RETRIEVAL  (4 collections)                           │
│   typology_library 1.5×  ·  case_history 1.0×  ·  regulatory 1.0×             │
│                       ·  personal_notes 1.2×                                  │
│                  Hybrid query: text + GNN embedding                           │
│                  Reranked, deduplicated, top-N filtered                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          REASONING LAYER (3 tiers)                            │
│  ┌──────────────────┐  ┌────────────────────────┐  ┌─────────────────────┐ │
│  │ Tier 1: Template │  │  Tier 2: SLM (Ollama)  │  │ Tier 3: LLM (Mistral)│ │
│  │   < 10 ms        │  │     2–5 s              │  │      1–3 s            │ │
│  │  HIGH cases      │  │   MEDIUM cases         │  │  LOW + AMBIGUOUS      │ │
│  └──────────────────┘  └────────────────────────┘  └─────────────────────┘ │
│   Provider chain with automatic fallback to local model on API failure        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     STRUCTURED OUTPUT  (Pydantic-validated)                   │
│  verdict · tier · ML evidence · statistical evidence · typology match         │
│  · narrative explanation · recommended action · RAG citations · audit trail   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SERVING LAYER                                       │
│              FastAPI endpoints  ·  Streamlit dashboard                        │
│              Review queue (SQLite) for LOW/AMBIGUOUS cases                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FEEDBACK LOOP (Self-Improvement)                          │
│   Analyst confirms ────► case_history (RAG) ────► LLM self-reflection         │
│                                                          │                    │
│                                                          ▼                    │
│                                                   personal_notes (RAG)        │
│   Analyst rejects ─────► false_positive_log (NOT RAG, used for retraining)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Detection Pipeline (Stage by Stage)

For each incoming transaction:

1. **Feature preprocessing** — `StandardScaler` (and optional `PCA`) is applied to the 165 raw features using parameters fit on the training set only.
2. **Per-detector scoring** — Each available detector produces a continuous score in `[0, 1]`. XGBoost outputs Platt-calibrated probabilities; GraphSAGE outputs softmax-normalized class-1 probability; Isolation Forest outputs `-decision_function` min-max normalized to `[0, 1]`; Autoencoder outputs reconstruction error min-max normalized.
3. **Ensemble combination** — Optuna-tuned weights produce `ensemble_score`. Per-detector thresholds produce `agreement_count` ∈ {0, 1, 2, 3, 4}.
4. **Statistical validation** — Mahalanobis distance from the licit cluster centroid; SHAP top-K feature attributions on XGBoost; ensemble-spread analysis.
5. **RAG query construction** — Hybrid query built from feature names, statistical evidence, and (when available) GNN node embedding.
6. **Multi-collection retrieval** — Top 3 from typology_library, top 3 from case_history, top 2 from regulatory, top 2 from personal_notes.
7. **Reranking** — Apply collection weights, drop documents below `min_similarity_threshold = 0.4`, deduplicate by ID, keep top 7.
8. **Routing decision** — `ConfidenceRouter` returns the tier and dispatches.
9. **Explanation generation** — Tier 1 fills a template; Tier 2 calls Ollama SLM; Tier 3 calls Mistral with fallback chain. Output is parsed into a Pydantic-validated `ExplanationOutput`.
10. **Review queue routing** — MEDIUM, LOW, and AMBIGUOUS cases land in `artifacts/audit.db` for human review.
11. **Audit log** — Full structured record persisted: model versions, latencies, provider used, tier rationale, RAG citations, timestamps.

When an analyst later confirms the case, the curator runs LLM self-reflection and writes a personal note. When an analyst rejects, the case goes to the false-positive log only.

---

## Layer-by-Layer Explanation

### Ingestion (`src/ingestion/`)

`EllipticLoader` reads the three Elliptic CSVs and normalizes labels to `{0=licit, 1=illicit, -1=unknown}`. It handles the inconsistent header presence of the v1 and v2 Kaggle releases (some versions have headers, some don't) and is robust to numeric vs. string transaction IDs.

`TemporalSplitter` enforces strict time-step separation: train `[1, 30]`, validation `[31, 40]`, test `[41, 49]`. The class will raise an error if any ranges overlap. This is the single most important correctness fix from v1, which used random splits and silently leaked future transactions into the training set.

### Features (`src/features/`)

`tabular.py` provides `fit_tabular_pipeline` (StandardScaler + optional PCA) which is fit on training data only and returns a reusable pipeline object.

`graph_features.py` computes degree, clustering coefficient, PageRank, and triangle counts from the transaction graph. **Per design, these are NOT fed to XGBoost** — GraphSAGE handles graph structure end-to-end. Graph features are used only for: statistical validation evidence, RAG query construction, and human-readable display.

### Detection (`src/detection/`)

Four models with different mathematical foundations, deliberately chosen so that errors are uncorrelated:

- **XGBoost** (`xgboost_detector.py`): supervised gradient-boosted trees, with `scale_pos_weight` to handle the ~2% illicit class imbalance, wrapped in `CalibratedClassifierCV` with Platt scaling for proper probability outputs.
- **GraphSAGE** (`detection/gnn/`): two-layer SAGE GNN with class-weighted cross-entropy. Produces both classification logits and 128-dim node embeddings. Trained with strict temporal masks. Best trained on Colab GPU; CPU inference is straightforward.
- **Isolation Forest** (`isolation_forest.py`): unsupervised. **Critical fix from v1**: uses `decision_function` for continuous scores, then min-max normalizes to `[0, 1]`. v1 used `predict` which returns `{-1, +1}` and then computed ROC-AUC on those binary values — mathematically broken.
- **Autoencoder** (`autoencoder.py`): trained only on licit transactions (learns the normal manifold), produces reconstruction error normalized to `[0, 1]`. Different mathematical question than IF: "do you live on the normal manifold?" rather than "are you isolated?"

**Why four models, not one?** Each model captures different signal:
- XGBoost: pattern memorization on labeled features
- GraphSAGE: network topology and neighbor influence
- Isolation Forest: density and isolation in feature space
- Autoencoder: deviation from learned licit manifold

Their errors are largely uncorrelated, which is what makes the ensemble robust. Models alone could be deceived by adversarial transactions; the four together are much harder to fool simultaneously.

The `ensemble.py` combiner uses **Optuna** for 50-trial PR-AUC-optimised weight tuning. It produces both a continuous `ensemble_score` (weighted sum) and a discrete `agreement_count` (how many detectors crossed their threshold). Both signals feed the routing layer.

### Statistical Validation (`src/validation/`)

This layer answers "but is this actually anomalous?" without trusting any single model:

- **Mahalanobis distance** (`mahalanobis.py`): Computes D² from the licit cluster centroid using a regularized covariance matrix, with χ² p-value. A sudden D² of 4.5 on a 165-dim space corresponds to p < 0.001. This is independent of the ML models; it's pure statistics.
- **SHAP attributions** (`shap_explainer.py`): Top-K feature contributions from XGBoost, with a graceful fallback to feature-importance × |x| when SHAP isn't installed.
- **Ensemble agreement** (`ensemble_agreement.py`): Beyond just counting flags, it tracks the *spread* between max and min individual scores, which feeds the AMBIGUOUS tier check.

**Why this is statistical and not "another LLM as judge"?** Because this evidence is what gets shown to the LLM and to regulators. A regulator asking "why did you flag this?" expects "Mahalanobis distance 4.5, p-value < 0.001, top-3 features were X, Y, Z, all four detectors agreed" — not "an SLM thought it was suspicious." Defensible regulatory evidence is statistical, not stochastic.

### Knowledge Base (`src/knowledge/`)

Four ChromaDB collections with `all-MiniLM-L6-v2` embeddings (384-dim, CPU-friendly):

- **`typology_library`** — 10 hand-curated fraud patterns. Static. Hand-edited only. Each entry has `name`, `category`, `description`, `indicators`, `graph_signature`, `typical_features`, `regulatory_refs`.
- **`case_history`** — Confirmed past cases. Cold-started by bootstrapping with up to 200 illicit transactions from Elliptic at setup. Grows from analyst confirmations.
- **`regulatory`** — 8 seed entries from FATF (R10, R11, R13, VASP guidance), OFAC (sanctioned mixers), FinCEN (ransomware advisory), EU MiCA, India PMLA. Hand-edited only.
- **`personal_notes`** — Starts empty. Grows from LLM self-reflection on confirmed cases.

`MultiCollectionRetriever` queries all four in parallel; for `case_history` it can use a hybrid query (text + GNN embedding). The reranker applies collection weights, filters by minimum similarity, deduplicates, and returns the top 7.

A `_StubEmbedder` and `_InMemoryStubCollection` provide deterministic fallbacks when ChromaDB or sentence-transformers aren't installed, so unit tests run cleanly in any environment.

### Reasoning (`src/reasoning/`)

Three explainers, one router, one self-reflection module:

- **`template_explainer.py`** (Tier 1): No LLM. Picks the highest-similarity typology from RAG, fills in evidence, recommends `auto_report`. Sub-10 ms latency.
- **`slm_explainer.py`** (Tier 2): Calls the local Ollama SLM via HTTP. Loads the prompt template from `config/prompts.yaml`. Robustly parses JSON output (handles markdown fences, preamble text, malformed responses). Coerces output to a valid `ExplanationOutput`, with safe defaults if the SLM produces invalid JSON.
- **`llm_explainer.py`** (Tier 3): Same interface as Tier 2, but uses a `FallbackChain` of providers. Mistral Small primary, Mistral Medium fallback, Ollama qwen2.5:3b as last resort.

The `LLMProvider` abstraction (`llm_abstraction.py`) provides a uniform interface (`complete(system, user, max_tokens, temperature) → LLMResponse`). New providers can be added without changing any explainer code.

`self_reflection.py` runs after analyst confirmation. It feeds the LLM the full case context and asks for a structured `SelfReflectionOutput` (topic, key_lesson, indicators_to_watch, related_typologies). The output is validated against the Pydantic schema before being added to `personal_notes`.

### Feedback Loop (`src/feedback/`)

`ReviewQueue` is a SQLite-backed queue for cases requiring human review. It supports `enqueue`, `list_pending`, `resolve`, and `stats`.

`RAGCurator` handles the asymmetric feedback policy:
- **Confirmed** → `case_history` collection + LLM self-reflection → `personal_notes`
- **Rejected** → `false_positives.db` (separate SQLite, NOT in RAG)

This asymmetry is intentional. If we let rejected cases into the knowledge base, the system would drift toward whatever pattern that particular analyst tends to mark as false positive — destroying its objectivity. The false-positive log is preserved for retraining and for quality-control audits, but it never influences runtime retrieval.

`AnalystOrchestrator` wraps the queue and curator into a single `apply()` call so the API and UI just send an `AnalystAction`.

### Backtesting (`src/backtesting/`)

Rolling-window backtests across the 49 time steps:

| Window | Train | Test |
|---|---|---|
| 1 | 1-25 | 26-30 |
| 2 | 1-30 | 31-35 |
| 3 | 1-35 | 36-40 |
| 4 | 1-40 | 41-49 |

For each window, the model is trained from scratch on the train slice and evaluated on the test slice. Metrics computed:

- **PR-AUC** (Average Precision) — primary metric, robust to class imbalance
- **ROC-AUC** — for completeness
- **Precision@5%** and **Precision@10%** — analyst review capacity
- **F1 at optimal threshold** and **MCC at optimal threshold** — discrete-prediction quality
- **Cost-weighted total cost** — using configurable `cost_fn = 100, cost_fp = 1` (false negatives 100× more expensive than false positives)

Drift detection (`drift_detector.py`) flags significant PR-AUC drops between windows and uses a Kolmogorov–Smirnov test to detect score distribution shifts.

### Serving (`src/api/`, `src/ui/`)

**FastAPI** (`src/api/main.py`):
- `POST /detect` — main detection endpoint, runs the full pipeline
- `GET /explain/{tx_id}` — retrieve a past detection (wired to audit DB)
- `POST /backtest` — trigger a rolling-window backtest
- `GET /knowledge_base` — RAG collection counts
- `GET /knowledge_base/{collection}/search?q=...` — semantic search
- `POST /feedback` — submit analyst confirmation/rejection

**Streamlit dashboard** (`src/ui/`):
- **Home** — system status, KB counts, bootstrap summary
- **1 Detection** — pick a test transaction / random / manual entry → run pipeline → see ML evidence, statistical evidence, narrative, audit trail
- **2 Review Queue** — pending LOW/AMBIGUOUS cases with confirm/reject buttons; closes the loop into case_history
- **3 Knowledge Base** — browse + semantic search across all 4 collections; seed button if empty
- **4 Backtesting** — run rolling-window backtest with PR-AUC chart, drift indicator
- **5 Personal Notes** — search + manually add notes; auto-populates as analysts confirm cases

### Evaluation (`src/evaluation/`)

`eval_cases.py` defines 50 hand-crafted scenarios — 10 per tier across PASS / HIGH / MEDIUM / LOW / AMBIGUOUS — each with expected behavior.

`routing_rubric.py` runs every eval case through the router and scores against `expected_tier`. The current build scores **100% on all 50 cases**.

`llm_rubric.py` checks LLM/SLM outputs against deterministic criteria (no LLM-as-judge):
- Schema validity (Pydantic)
- `recommended_action ∈ {auto_report, analyst_review, monitor}`
- `typology_match` is in the known library or `no_match`
- `typology_confidence ∈ [0, 1]`
- `rag_citations` is non-empty when retrieval returned docs
- When expected, `typology_match` matches expected typology
- When expected, `recommended_action` matches expected action

`run_evaluation.py` orchestrates both rubrics and saves a JSON report to `reports/evaluation/`.

### Pipeline Orchestrator (`src/pipeline.py`)

`detect_transaction()` is the single entry point used by both the API and the UI. Given a transaction, it loads trained artifacts, runs all available detectors, applies statistical validation, retrieves RAG context, routes through the confidence router, dispatches to the right reasoning tier, and returns a `PipelineResult`. MEDIUM, LOW, and AMBIGUOUS cases are automatically enqueued for analyst review.

The pipeline gracefully degrades when fewer detectors are available. With only one detector (e.g., Isolation Forest), it scales the agreement count proportionally so the existing routing rules still apply: `effective_agreement = round(agreement × 4 / n_active)`. This means the demo runs end-to-end with just `pip install scikit-learn` — though obviously full performance requires the complete ensemble.

---

## Key Design Decisions and Why

| Decision | Why |
|---|---|
| **Fork v1 instead of patch** | v1 had architectural-level bugs (random splits, hardcoded metrics, broken IF outputs). Patching would have left silent inconsistencies. v2 is fresh; v1 is preserved as a reference. |
| **4 models with different math, not 4 instances of XGBoost** | Different mathematical foundations → uncorrelated errors → genuine ensemble robustness. Four XGBoost variants would just average the same biases. |
| **GraphSAGE end-to-end, no graph features in XGBoost** | GraphSAGE's whole point is to learn graph representations directly. Hand-crafted graph features fed to XGBoost would be a downgraded version of the same idea. Cleaner separation. |
| **Optuna-tuned weights, not equal weighting** | The four detectors have different reliability and different score scales. PR-AUC-optimised weights significantly outperform `0.25` each. |
| **PR-AUC primary metric, not ROC-AUC** | At ~2% base rate, ROC-AUC over-credits false-positive control. PR-AUC and Precision@k are what actually matter for fraud. |
| **Statistical validation, not LLM-as-judge** | Regulators expect Mahalanobis distance and p-values, not "an SLM agreed." Deterministic statistics are defensible; LLM judgments are not. |
| **3-tier routing instead of always-LLM** | LLM calls are slow and expensive. Most flagged cases are obvious — a template suffices. Routing means we only pay for LLM reasoning when it actually adds value. |
| **4 collections instead of 1 RAG store** | Different document types serve different purposes. A regulatory citation and a past case have different roles in the explanation. Keeping them separate means we can weight, query, and update them independently. |
| **Personal notes are LLM-generated, not hand-written** | The LLM is the consumer of these notes (it retrieves them on future calls). Letting it write them in its own preferred format means retrieval similarity is naturally higher. Humans can edit or delete. |
| **Confirmed → RAG, Rejected → separate log** | If rejections entered the knowledge base, the system would drift toward each analyst's idiosyncratic style. The asymmetry preserves objectivity. |
| **Mistral over Groq/Gemini** | A prior project hit Groq rate limits during a demo and Gemini's pricing tier was uncertain. Mistral's free tier (1B tokens/month) is generous and EU-GDPR is friendlier than US-only providers. |
| **Ollama as final fallback** | If the API is down, the system still works. Local model is slower but zero-network. |
| **Pydantic schemas with dataclass fallback** | Strict validation in production; tests still run when pydantic isn't installed. |
| **CPU-only inference target** | Real deployment of fraud detection is rarely on GPU. A CPU-deployable system is more honest about real constraints. |
| **Synthetic data generator included** | Onboarding shouldn't require a Kaggle download. The demo should work in 30 seconds. |
| **Graceful degradation everywhere** | Optional dependencies (xgboost, torch, chromadb, etc.) all have fallbacks. The system runs end-to-end with just sklearn. Adding more deps just unlocks more capability. |

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~500 MB disk for code + dependencies
- ~600 MB additional disk if downloading real Elliptic data

### Setup

```powershell
git clone <repo-url> AnomalyDetector_v2
cd AnomalyDetector_v2

python -m venv venv
.\venv\Scripts\activate              # Windows PowerShell
# source venv/bin/activate           # Linux/macOS

pip install -r requirements.txt
copy .env.example .env               # add MISTRAL_API_KEY if you have one
```

### Choose your data path

**Option A — Real Elliptic data (recommended for evaluation)**

Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set, extract, place CSVs at `data/raw/elliptic_bitcoin_dataset/`. Full instructions in `DATA_SETUP.md`.

**Option B — Synthetic data (instant demo)**

```powershell
python -m scripts.generate_synthetic_data --n-tx 3000
```

### Bootstrap the pipeline

One command does everything:

```powershell
python -m scripts.bootstrap
```

This loads + temporal-splits the data, trains all available detectors, seeds the four RAG collections, runs the routing rubric (expects 100% accuracy), and saves a summary to `artifacts/bootstrap_summary.json`.

### Run the system

```powershell
# Terminal 1: API
uvicorn src.api.main:app --reload

# Terminal 2: Dashboard
streamlit run src/ui/app.py
```

Open the Streamlit URL (usually `http://localhost:8501`). The five-page dashboard is fully wired to the pipeline.

### Run tests

```powershell
pytest tests/ -v
```

### Run the evaluation rubric

```powershell
python -m src.evaluation.run_evaluation
```

This generates a JSON report in `reports/evaluation/` confirming routing logic correctness across all 50 cases.

---

## Folder Structure

```
AnomalyDetector_v2/
├── README.md                          ← this file
├── DATA_SETUP.md                      ← Kaggle download + synthetic alternative
├── PROJECT_LOG.md                     ← every architectural decision logged
├── pyproject.toml
├── requirements.txt
├── .env.example
│
├── config/
│   ├── default.yaml                   ← paths, splits, detection, RAG, LLM, backtest
│   ├── routing.yaml                   ← 5-tier routing rules
│   ├── ensemble_weights.yaml          ← Optuna-tuned weights (populated by bootstrap)
│   └── prompts.yaml                   ← LLM/SLM prompt templates
│
├── data/
│   └── raw/elliptic_bitcoin_dataset/  ← place CSVs here
│
├── scripts/
│   ├── generate_synthetic_data.py     ← demo data generator
│   ├── verify_data.py                 ← validates data placement
│   └── bootstrap.py                   ← one-command setup
│
├── src/
│   ├── ingestion/                     ← elliptic_loader, temporal_splitter
│   ├── features/                      ← tabular, graph_features
│   ├── detection/                     ← xgboost_detector, isolation_forest, autoencoder, ensemble
│   │   └── gnn/                       ← graphsage, trainer, embeddings
│   ├── validation/                    ← mahalanobis, shap_explainer, ensemble_agreement, confidence_router
│   ├── knowledge/                     ← chromadb_client, retriever, reranker, ingestion
│   │   └── collections/               ← typology_library, case_history, regulatory, personal_notes
│   ├── reasoning/                     ← schemas, llm_abstraction, mistral_client, ollama_client,
│   │                                    template_explainer, slm_explainer, llm_explainer,
│   │                                    self_reflection, router
│   ├── feedback/                      ← review_queue, rag_curator, analyst_actions
│   ├── backtesting/                   ← temporal_backtest, metrics, cost_weighted, drift_detector
│   ├── evaluation/                    ← eval_cases (50 cases), routing_rubric, llm_rubric, run_evaluation
│   ├── pipeline.py                    ← end-to-end orchestrator (used by API and UI)
│   ├── api/                           ← FastAPI service
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routers/                   ← detect, explain, backtest, knowledge_base, feedback
│   ├── ui/                            ← Streamlit dashboard
│   │   ├── app.py
│   │   └── pages/                     ← 1-5 (Detection, Review Queue, KB, Backtesting, Personal Notes)
│   └── utils/                         ← logging, config, monitoring
│
├── tests/                             ← 10 test files, ~100+ test cases
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_detection.py
│   ├── test_validation.py
│   ├── test_routing.py
│   ├── test_rag.py
│   ├── test_reasoning.py
│   ├── test_backtesting.py
│   ├── test_feedback.py
│   ├── test_evaluation.py
│   └── test_api.py
│
├── notebooks/                         ← Colab notebooks for GNN training
├── models/                            ← trained artifacts (created by bootstrap)
├── artifacts/                         ← chromadb, audit.db, false_positives.db
└── reports/                           ← backtest_results, evaluation/, drift/
```

---

## Tech Stack

**Core ML:** scikit-learn, XGBoost, PyTorch, PyTorch Geometric, Optuna, SHAP, NumPy, Pandas, NetworkX, SciPy

**RAG and embeddings:** ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`)

**LLM/SLM:** Mistral API (Mistral Small 3.1 + Medium 3), Ollama (qwen2.5:1.5b for Tier 2, qwen2.5:3b as fallback)

**Service layer:** FastAPI, Uvicorn, Pydantic, Streamlit

**Storage:** SQLite (review queue, audit trail, false positive log), ChromaDB (RAG)

**Observability:** structlog (JSON logging), custom monitoring module

**Validation:** Pydantic schemas with dataclass fallback for environments without pydantic

**Testing:** pytest with fixtures and parametrization

**Configuration:** YAML files in `config/`, environment overrides via `.env`

---

## Testing and Evaluation

### Unit tests

10 test files covering every module:

| File | What it tests |
|---|---|
| `test_ingestion.py` | Elliptic loader, label normalization, temporal split (no leakage — the most important test) |
| `test_detection.py` | XGBoost calibration, **IF continuous-score correctness (v1 fix)**, ensemble combiner, Optuna tuning |
| `test_validation.py` | Mahalanobis distance, outlier detection, ensemble agreement |
| `test_routing.py` | All 5 router tiers, edge cases (PASS overrides AMBIGUOUS, AMBIGUOUS overrides HIGH) |
| `test_rag.py` | ChromaDB add/query, seeding, hybrid retrieval, reranking |
| `test_reasoning.py` | Schema validity, JSON parsing robustness, fallback chain |
| `test_backtesting.py` | Metrics, cost-weighted evaluation, drift detection, rolling windows |
| `test_feedback.py` | Review queue, curator (confirmed → RAG, rejected → log), analyst orchestrator |
| `test_evaluation.py` | 50 cases × 5 tiers, rubric scoring, aggregate stats |
| `test_api.py` | All 5 FastAPI endpoints |

Run with `pytest tests/ -v`.

### Routing rubric

50 hand-crafted scenarios (10 per tier) validate the routing logic. The current build scores **100% accuracy**.

```powershell
python -m src.evaluation.run_evaluation
```

### LLM rubric

Deterministic checks (no LLM-as-judge) validate Tier 2/3 outputs against:
- Pydantic schema validity
- `recommended_action` enum compliance
- Typology in library or `no_match` (catches hallucinated typologies)
- Confidence range
- Citation completeness
- Match against expected typology when specified

### Backtesting

Rolling-window backtest with drift detection. Reports saved to `reports/backtest_results/`.

---

## Backtesting Results

The system was evaluated using strict temporal splits across four rolling windows on the real Elliptic Bitcoin Dataset. No data from future time steps was visible during training of any window — this is the core correctness guarantee that v1 lacked.

| Window | n_train | n_test | PR-AUC | ROC-AUC | P@5% | P@10% | F1* | MCC* | Cost |
|---|---|---|---|---|---|---|---|---|---|
| train[1-25]_test[26-30] | 111,264 | 12,023 | 0.959 | 0.857 | 1.000 | 1.000 | 0.874 | 0.193 | 4,550 |
| train[1-30]_test[31-35] | 123,287 | 18,485 | 0.964 | 0.836 | 1.000 | 0.998 | 0.921 | 0.296 | 3,787 |
| train[1-35]_test[36-40] | 141,772 | 19,831 | 0.984 | 0.847 | 0.996 | 0.991 | 0.965 | 0.231 | 1,049 |
| train[1-40]_test[41-49] | 161,603 | 42,166 | 0.981 | 0.761 | 1.000 | 1.000 | 0.974 | 0.278 | 5,946 |

**Summary:** mean PR-AUC **0.972** (std 0.011, min 0.959, max 0.984) across all four windows.

The PR-AUC results are strong and consistent, confirming that the ensemble generalizes well across time steps under strict temporal separation. Precision@5% and Precision@10% are near-perfect in all windows, which is the operationally relevant metric — analysts can only review a fixed number of flagged cases per day.

One anomaly worth noting: **F1 and MCC move in opposite directions across windows**, which is unexpected given they measure similar things at the optimal threshold. This is currently under investigation. Likely causes include threshold sensitivity at low illicit base rates, or the cost-weighted threshold selection interacting differently with class distributions in each test window. The investigation is ongoing and findings will be logged in `PROJECT_LOG.md`.

---



Bootstrap takes about 30 seconds on synthetic data and 3–5 minutes on real Elliptic with all detectors installed.

---

## Honest Limitations

1. **Dataset license.** The Elliptic Bitcoin Dataset is CC BY-NC 4.0 (non-commercial). The system cannot be used commercially without a separate data agreement.

2. **No JWT auth, rate limiting, or containerization.** Optional API key via `ANOMALY_API_KEY` env var only. Production hardening (JWT, rate limiting, Dockerfile) is not yet implemented.

3. **No automated time-decay scoring on case_history.** Older confirmed cases have equal weight to recent ones. There is no recency weighting to let older cases naturally fade in influence — all entries in `case_history` retrieve at the same weight regardless of age.

4. **Personal notes don't auto-populate without an LLM provider.** The `RAGCurator.on_confirmed` self-reflection step requires Ollama or Mistral. With neither configured, confirmed cases still enter `case_history`, but `personal_notes` only grows from manual entry.

5. **F1/MCC inconsistency under investigation.** As noted in the backtesting results, F1 and MCC diverge across windows in a way that is not yet fully explained. Results should be interpreted with this in mind until the root cause is identified.

---

## Future Work

- Time-decay scoring on `case_history` so older confirmed cases naturally fade in retrieval influence
- Active learning loop from analyst rejections feeding retraining pipelines
- Integration of RAGWatch (custom RAG evaluation framework) for retrieval quality monitoring
- Optional LSTM or Transformer detector as a fifth ensemble member for temporal sequence modelling
- Automated drift alerting with configurable thresholds and notification hooks

---

## License

This project is built on the Elliptic Bitcoin Dataset, distributed under the dataset's own license terms (CC BY-NC 4.0 — non-commercial use). Code is original work; please cite if reused.

---

*If anything here is unclear, the corresponding test file is the authoritative spec. Tests are written to be readable as documentation.*
