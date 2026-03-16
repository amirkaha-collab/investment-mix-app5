# Architecture

## Overview

The Portfolio Analyzer follows a clean layered architecture where each layer
has a single responsibility and depends only on layers below it.

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit UI                          │
│           app.py  ·  src/ui/state.py  ·  components.py   │
└───────────────────────────┬──────────────────────────────┘
                            │ calls
┌───────────────────────────▼──────────────────────────────┐
│               Pipeline / Service Layer                    │
│          src/services/pipeline.py                        │
│          src/services/export_service.py                  │
└──────┬────────────┬──────────────┬───────────────────────┘
       │            │              │
 ┌─────▼────┐ ┌────▼──────┐ ┌────▼────────┐ ┌────────────┐
 │ Parsers  │ │ Research  │ │  Analysis   │ │Presentation│
 │          │ │           │ │   Engine    │ │  Builder   │
 │ PDF      │ │ Mock /    │ │ 10 modules  │ │ PPTX       │
 │ XLSX     │ │ Web /     │ │ QA Engine   │ │            │
 │ CSV      │ │ Custom    │ │             │ │            │
 └─────┬────┘ └────┬──────┘ └────┬────────┘ └────────────┘
       │            │              │
┌──────▼────────────▼──────────────▼───────────────────────┐
│                  Domain Models                            │
│           src/domain/models.py                           │
│  HoldingNormalized · UserAnalysisPreferences             │
│  AnalysisOutputs · AssumptionRow · DataQualityNote …     │
└──────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│               Config / Constants                          │
│  src/config/settings.py  ·  src/config/constants.py      │
└──────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
File Upload (bytes)
       │
       ▼
  BaseParser.parse()
       │ ParseResult (raw DataFrame)
       ▼
  HoldingsNormalizer.normalize()
       │ list[HoldingNormalized]  +  warnings
       ▼
  [User Review & Edit – Streamlit]
       │ confirmed list[HoldingNormalized]
       ▼
  [Clarification Q&A – Streamlit]
       │ UserAnalysisPreferences
       ▼
  EnrichmentService.enrich()   ← ResearchProvider
       │ mutates holdings in-place
       ▼
  run_analysis(holdings, prefs)
       │ AnalysisOutputs
       ▼
  QAEngine.run()               ← validates consistency
       │ qa_warnings / qa_errors added to AnalysisOutputs
       ▼
  [Streamlit Report Dashboard]
       │
       ├──▶ PPTXBuilder.build()   → bytes
       ├──▶ ExportService.to_json() → bytes
       └──▶ ExportService.holdings_to_csv() → bytes
```

---

## Key Design Decisions

### 1. No silent estimation

Every field that cannot be obtained from an official source is:
- Added to `HoldingNormalized.estimated_fields`
- Added to `AnalysisOutputs.assumptions`
- Rendered with `[E]` marker in all outputs

### 2. Provider pattern for research

```
ResearchProvider  (abstract)
├── MockResearchProvider     → deterministic, no network
├── WebResearchProvider      → HTTP + cache + retry
└── (future) APIResearchProvider  → licensed data API
```

Switching providers requires only changing `RESEARCH_PROVIDER` in `.env`.

### 3. Two-denominator cost model

`weighted_fund_cost_on_funds`          → denominator = fund AUM only  
`effective_fund_cost_on_total_portfolio` → denominator = total portfolio  
`total_cost_percent`                   → only computed when PM fee is provided

This prevents over-reporting costs on non-fund holdings.

### 4. Two-methodology US exposure

`conservative_us_exposure` → only direct US holdings / US-domiciled ETFs  
`broad_us_exposure`        → adds global USD bond funds (user opt-in)

### 5. QA Engine blocks PPTX on errors

`qa_errors` (not warnings) block `PPTXBuilder.build()`.  
`qa_warnings` are non-blocking but shown in the report.

---

## Module Responsibilities

| Module | Responsible for | NOT responsible for |
|--------|----------------|---------------------|
| `parsers/` | Raw bytes → DataFrame | Validation, business rules |
| `parsers/normalizer.py` | DataFrame → HoldingNormalized | Research enrichment |
| `research/` | Filling metadata gaps | Computation |
| `analysis/` | Computing all metrics | Rendering, serialisation |
| `presentation/` | PPTX layout + injection | Computing metrics |
| `services/pipeline.py` | Orchestration | Business logic |
| `ui/` | Rendering + input | Business logic |

---

## Adding a New Analysis Module

1. Create `src/analysis/my_module.py`
2. Implement a pure function: `def compute_X(holdings, prefs) -> list[MyRow]`
3. Add `MyRow` to `src/domain/models.py`
4. Add `outputs.my_x = compute_X(...)` in `src/analysis/engine.py`
5. Add QA check in `src/analysis/qa_engine.py`
6. Add rendering in `src/ui/components.py`
7. Add a slide builder in `src/presentation/pptx_builder.py`

---

## Storage Layer Hooks

Local temp storage is used by default.  To add S3/GCS/Supabase:

1. Implement `StorageBackend` interface in `src/services/storage/`
2. Factory-select based on `settings.storage_backend`
3. Replace `tempfile` usage in `pipeline.py` with storage calls

No other code changes are needed.

---

## Future Authentication Hook

```python
# Future: src/services/auth.py
class AuthService:
    def require_authenticated(self, request) -> User: ...
    def charge_report(self, user_id: str, amount: float) -> None: ...
```

The pipeline accepts an optional `user_id` parameter (currently unused)
to be wired to auth/billing when needed.
