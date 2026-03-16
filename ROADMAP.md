# Roadmap

## Phase 1 – Core (Current)
- [x] PDF / XLSX / CSV parsing with fallback chain
- [x] Holdings normalisation + user review step
- [x] Clarification Q&A
- [x] Mock research provider
- [x] 10 analysis modules
- [x] QA engine
- [x] Streamlit report dashboard (RTL, Hebrew)
- [x] PPTX generation from template
- [x] JSON + CSV exports

---

## Phase 2 – Production Data

- [ ] **Live FX rates** – fetch USD/ILS from Bank of Israel API (free, official)
- [ ] **Real research provider** – integrate with Morningstar API or Bloomberg Open Symbology
- [ ] **TASE integration** – Israeli bond data via TASE public API
- [ ] **OCR fallback** – for scanned PDFs, route to Google Vision / AWS Textract

---

## Phase 3 – Multi-User + Auth

- [ ] **Supabase auth** – email/password + magic link
- [ ] **Row-level security** – each user sees only their own reports
- [ ] **Report history** – stored in Supabase; browseable in sidebar
- [ ] **User preferences persistence** – default PM fee, language, etc.

---

## Phase 4 – Payments

- [ ] **Stripe integration** – pay-per-report or subscription
- [ ] **Usage gating** – free tier: 3 reports/month; paid: unlimited
- [ ] **Invoice generation** – PDF invoice per report
- [ ] **Webhook handling** – subscription events

---

## Phase 5 – Job Queue + Scale

- [ ] **Background jobs** – Celery or Inngest for long-running analysis
- [ ] **Progress streaming** – real-time updates via Streamlit `st.status`
- [ ] **Report caching** – same file hash → return cached result instantly
- [ ] **Horizontal scaling** – stateless pipeline, Redis for job queue

---

## Phase 6 – LLM Insights (Wording Only)

See `PROMPTING_NOTES.md` for the strict contract.

- [ ] **Slide summaries** – 2-sentence executive summary per slide
- [ ] **Concentration commentary** – automated note on concentration risk
- [ ] **Cost commentary** – contextualise fund costs vs benchmarks
- [ ] **Clarification question phrasing** – LLM rewrites static Q&A in natural language

---

## Phase 7 – Additional Exports

- [ ] **Google Slides export** – via Google Slides API
- [ ] **Excel report** – multi-tab workbook with all analysis tables
- [ ] **Interactive HTML report** – standalone HTML with Chart.js (no Streamlit needed)
- [ ] **Audit log CSV** – timestamp + user + every assumption made

---

## Phase 8 – Advanced Analysis

- [ ] **Performance attribution** – if historical NAV data is available
- [ ] **Benchmark comparison** – portfolio vs 60/40, vs TA-125, vs user-defined
- [ ] **Stress testing** – scenario analysis (interest rate +1%, equity -20%)
- [ ] **Compliance flags** – flag holdings that violate user-defined mandates

---

## Technical Debt / Quality

- [ ] Type-check with `mypy --strict`
- [ ] Property-based testing with Hypothesis for analysis engine
- [ ] Integration test with real sample PDFs
- [ ] GitHub Actions CI/CD pipeline
- [ ] Docker image + Compose file
- [ ] Streamlit secrets integration for cloud deployment
