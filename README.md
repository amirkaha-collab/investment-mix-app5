# 📊 Portfolio Analyzer

> מנתח תיק השקעות אוטומטי | Automated Investment Portfolio Analyzer

A production-grade Streamlit application that ingests investment portfolio statements
(PDF / XLSX / CSV), extracts and normalises holdings, runs multi-layer analysis,
and generates a complete PowerPoint presentation and data exports.

**Every computed number is traceable to its source row.**  
**Every estimated field is explicitly labelled.**

---

## Features

| Module | Description |
|--------|-------------|
| **File Parsing** | PDF (pdfplumber → camelot → pymupdf), XLSX, CSV |
| **Holdings Review** | Editable table before analysis begins |
| **Clarification Q&A** | Collects PM fees, methodology preferences |
| **Research Enrichment** | Fetches expense ratios, duration, sector weights |
| **Analysis Engine** | 10 analysis modules (allocation, geography, sectors, bonds, duration, FX, costs, concentration) |
| **QA Engine** | Internal consistency checks before report generation |
| **PPTX Builder** | Injects charts + tables into branded template |
| **Exports** | JSON (full), CSV (holdings), CSV (analysis summary) |
| **RTL Support** | Full Hebrew right-to-left rendering in Streamlit + PPTX |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ghostscript](https://www.ghostscript.com/) (required by camelot for PDF table extraction)
- Optional: Java (for tabula-py fallback)

### Installation

```bash
git clone https://github.com/yourorg/portfolio-analyzer.git
cd portfolio-analyzer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env as needed
```

### Running Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All configuration lives in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_PROVIDER` | `mock` | `mock` (no network) or `web` |
| `USD_TO_ILS_RATE` | `3.75` | Fallback exchange rate |
| `APP_LOG_LEVEL` | `INFO` | Logging verbosity |
| `CACHE_DIR` | `.cache/research` | Research result cache |

### Research Providers

| Provider | Description |
|----------|-------------|
| `mock` | In-memory database. Fast, no network calls. Best for development. |
| `web` | Fetches from Yahoo Finance + ETF.com. Cached for 24h. |

To add a custom provider:

```python
# src/research/my_provider.py
from src.research.base import ResearchProvider, ResearchResult

class MyProvider(ResearchProvider):
    def lookup(self, *, ticker="", isin="", name="", asset_class="") -> ResearchResult:
        ...
```

Register in `src/research/__init__.py`:
```python
if settings.research_provider == "my_provider":
    from .my_provider import MyProvider
    return MyProvider()
```

---

## Adding a PPTX Template

1. Place your template at `templates/Strategic_Portfolio_Template_AI_Fillable.pptx`
2. The builder uses slide layout index 6 (blank) by default
3. If the template is missing, a blank presentation is created automatically

**Template requirements:**
- Minimum 7 slide layouts
- Layout index 6 should be a clean blank layout
- No required placeholders (all content is injected programmatically)

---

## Project Structure

```
portfolio-analyzer/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env.example
├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── PROMPTING_NOTES.md
├── src/
│   ├── config/
│   │   ├── settings.py           # Pydantic settings (env vars)
│   │   └── constants.py          # All magic numbers, labels, reference tables
│   ├── domain/
│   │   └── models.py             # All Pydantic data models
│   ├── parsers/
│   │   ├── base.py               # Abstract BaseParser + ParseResult
│   │   ├── pdf_parser.py         # pdfplumber → camelot → pymupdf
│   │   ├── excel_csv_parser.py   # XLSX + CSV parsers
│   │   ├── normalizer.py         # Raw DataFrame → HoldingNormalized
│   │   └── __init__.py           # Parser factory
│   ├── research/
│   │   ├── base.py               # Abstract ResearchProvider
│   │   ├── mock_provider.py      # In-memory mock database
│   │   ├── web_provider.py       # HTTP-based provider (cached)
│   │   └── __init__.py           # Factory + EnrichmentService
│   ├── analysis/
│   │   ├── engine.py             # Orchestrator
│   │   ├── asset_allocation.py   # Rule 1
│   │   ├── equity_geography.py   # Rule 2
│   │   ├── us_exposure.py        # Rule 3
│   │   ├── sector_allocation.py  # Rule 4
│   │   ├── bond_analysis.py      # Rules 5 + 6
│   │   ├── fund_costs.py         # Rules 7 + 8
│   │   ├── fx_exposure_concentration.py  # Rules 9 + 10
│   │   └── qa_engine.py          # Internal QA checks
│   ├── presentation/
│   │   └── pptx_builder.py       # PPTX generation (12 slide types)
│   ├── services/
│   │   ├── pipeline.py           # Orchestration pipeline
│   │   └── export_service.py     # JSON + CSV export
│   ├── ui/
│   │   ├── state.py              # Session state management
│   │   └── components.py         # Reusable Streamlit components
│   └── utils/
│       ├── chart_utils.py        # matplotlib chart → PNG bytes
│       ├── formatters.py         # Number/table formatting
│       └── rtl_utils.py          # Hebrew RTL text handling
├── templates/
│   └── Strategic_Portfolio_Template_AI_Fillable.pptx
├── tests/
│   ├── conftest.py
│   ├── test_normalizer.py
│   ├── test_analysis_engine.py
│   ├── test_parsers.py
│   ├── test_research.py
│   └── test_export.py
└── sample_data/
    ├── sample_holdings.csv
    └── sample_analysis_output.json
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run only fast (non-integration) tests:
```bash
pytest tests/ -v -m "not integration"
```

---

## Deploying to Streamlit Community Cloud

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/portfolio-analyzer.git
   git push -u origin main
   ```

2. **Connect on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app** → select your repo
   - Main file path: `app.py`
   - Python version: `3.11`

3. **Set secrets** (App Settings → Secrets):
   ```toml
   RESEARCH_PROVIDER = "mock"
   USD_TO_ILS_RATE   = "3.75"
   APP_LOG_LEVEL     = "INFO"
   APP_ENV           = "production"
   ```

4. **Deploy** – Streamlit installs `requirements.txt` and `packages.txt` automatically.

### Files that enable Cloud deployment

| File | Purpose |
|------|---------|
| `.streamlit/config.toml` | UI theme + 50 MB upload limit |
| `.streamlit/secrets.toml.example` | Secrets template (never committed) |
| `packages.txt` | System apt packages (libgl1 for PDF rendering) |
| `requirements.txt` | camelot disabled (needs Ghostscript, not on Cloud) |
| `src/config/settings.py` | Auto-bridges `st.secrets` → `os.environ` |

### Limitations on the free tier
- No outbound HTTP → keep `RESEARCH_PROVIDER=mock`
- 1 GB RAM → avoid PDFs larger than ~20 MB
- No persistent disk → all exports stay in-memory (already the default)

### For full PDF table extraction (self-hosted Docker)
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ghostscript libgl1 && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt && pip install "camelot-py[cv]"
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## User Flow

```
Upload File
    ↓
Parse (PDF/XLSX/CSV)
    ↓
Normalize Holdings
    ↓
[User Review & Edit Holdings]
    ↓
[Clarification Q&A]
    ↓
Research Enrichment
    ↓
Analysis Engine (10 modules)
    ↓
QA Checks
    ↓
Report Dashboard (Streamlit)
    ↓
Export (PPTX + JSON + CSV)
```

---

## Analysis Modules

1. **Asset Allocation** – equity / bond / cash / other
2. **Equity Geography** – Israel / USA / Europe / EM / Global
3. **US Exposure** – conservative (direct) + broad (optional USD bond funds)
4. **Sector Allocation** – GICS sectors, ETF composition applied proportionally
5. **Bond Breakdown** – CPI-linked / nominal ILS / USD / global
6. **Weighted Duration** – conservative (official sources only) + extended (with estimates)
7. **Fund Costs** – per-fund fee, weighted cost on funds, effective cost on total
8. **Total Cost** – fund cost + PM fee (only when PM fee is provided)
9. **FX Exposure** – currency bucketing, hedging status
10. **Concentration** – top-5, top-10, single-holding max weight

---

## Data Traceability Guarantee

The system enforces the following invariant:

> **Every number that appears in the report, PPTX, or exports must be traceable to source holdings rows.**

This is enforced by:
- The QA engine verifying weight sums after every calculation
- All estimated fields being tagged in `estimated_fields` and the `assumptions` table
- JSON export including the full `AnalysisOutputs` model (every row-level detail)

---

## Limitations

- OCR for scanned PDFs is not implemented (planned via vision API fallback)
- Live market prices / NAVs are not fetched (planned via market data provider)
- Multi-currency FX rates: currently only USD→ILS is applied; other currencies use the same rate as a proxy
- The PPTX template bundled is a plain placeholder; replace with your own branded template
- LLM-generated insights are not yet implemented (see `PROMPTING_NOTES.md`)

---

## License

MIT License – see `LICENSE` file.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run `pytest` before submitting a PR
4. Follow the clean architecture patterns in `ARCHITECTURE.md`

---

*This system provides informational analysis only and does not constitute investment advice.*
