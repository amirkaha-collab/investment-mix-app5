# LLM Prompting Notes

## Core Contract

> **The LLM generates wording. The code generates numbers.**

This is non-negotiable.  No metric, weight, value, or calculation should ever
be delegated to an LLM.  LLMs are used only for natural language generation
based on pre-computed, validated data.

---

## What the LLM Can Generate

| Use Case | Input to LLM | Expected Output |
|----------|-------------|-----------------|
| Slide executive summary | `asset_allocation` table, total value | 2–3 sentence summary |
| Concentration insight | `top10_concentration`, `max_single_holding_weight` | 1 sentence risk commentary |
| Cost commentary | `total_cost_percent`, benchmark fee | 1 sentence relative context |
| Duration insight | `conservative_weighted_duration` | 1 sentence rate sensitivity note |
| Clarification Q&A phrasing | Question ID + context | Natural language question |
| Assumptions footnote | `AssumptionRow` list | Bullet list of caveats |

---

## What the LLM Must NEVER Do

- Compute or infer a numeric value
- Decide how to classify an asset class
- Determine whether a bond is CPI-linked
- Fill in a missing fee or duration
- Calculate weights, sums, or percentages
- Decide if a holding is FX-hedged
- Override a QA error

---

## Prompt Template (Slide Summary)

```
You are a professional investment analyst writing a summary for a Family Office report.
The data below has been computed by a rules-based system. Do NOT change any numbers.
Write a 2-sentence executive summary in {language}. Be factual, concise, and professional.

ASSET ALLOCATION:
{asset_allocation_json}

TOTAL PORTFOLIO VALUE: {total_ils} ILS
CLIENT: {client_name}

Rules:
- Do not add numbers not present in the data above
- Do not use hedging language like "may" or "might" for facts
- Do not recommend any action
- End with: "ניתוח זה מיועד למטרות מידע בלבד."
```

---

## Prompt Template (Clarification Question Phrasing)

```
You are a Hebrew-speaking investment analyst assistant.
Rephrase the following technical question in natural, polite Hebrew
suitable for a non-technical client. Do not change the meaning.
Keep it to 1–2 sentences. Do not add examples.

Question ID: {question_id}
Technical phrasing: {technical_text}
Context: {context}
```

---

## Implementation Notes

When implementing LLM integration (Phase 6):

1. Always pass data as structured JSON, not prose
2. Set `temperature=0` for factual summaries
3. Set `max_tokens` to a small value (≤200) to prevent hallucination sprawl
4. Always post-process: strip any numbers that appear in LLM output but not in input data
5. Mark all LLM-generated text in the report with an `[AI summary]` tag
6. Store the prompt + response in the audit log

---

## Model Recommendation

For wording tasks: `claude-3-haiku` or `gpt-4o-mini` (fast, cheap, deterministic)  
For complex multi-document insights: `claude-3-5-sonnet` or `gpt-4o`  
For RTL/Hebrew quality: test both; Claude tends to handle Hebrew better

---

## Anti-patterns to Avoid

```python
# ❌ WRONG – asking the LLM to compute
prompt = f"What is the weighted average duration of these bonds? {bonds_json}"

# ✅ CORRECT – pre-compute, then ask for wording
wad = compute_wad(bonds)
prompt = f"Write a 1-sentence comment on a portfolio with weighted duration of {wad:.1f} years."
```

```python
# ❌ WRONG – asking the LLM to classify
prompt = f"Is this bond CPI-linked? Name: {name}, ISIN: {isin}"

# ✅ CORRECT – use the rules-based classifier
linkage = _infer_bond_linkage(name, currency)
```
