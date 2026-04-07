# Travel Search Query — Exploratory Analysis

> **Testing Context:** This is a personal experiment to evaluate how well [Claude Code](https://claude.ai/code) (Anthropic's AI coding assistant) can independently plan, execute, and communicate a full exploratory data analysis — given only a raw dataset and a high-level brief from a non-engineer. The dataset used is a **mock/synthetic dataset** and does not reflect real production traffic. All findings are illustrative, not operational.

---

## What This Is

A complete exploratory analysis of 5,000 mock travel search queries, covering:

- Query distribution across 6 product categories
- Multi-category intent overlap (queries that span more than one category)
- Destination coverage and frequency
- Vague / low-intent query detection
- Temporal signals (dates, seasons, urgency)
- Price-tier signals (budget vs. luxury)
- Traveller persona themes (solo female, family, all-inclusive, etc.)

The final output is a self-contained **executive HTML report** with embedded charts and tables, structured as a narrative from most to least critical finding, with recommendations and data gap callouts.

---

## Why This Experiment?

The goal was to answer: *Can Claude independently produce the quality of analysis a Senior Product Data Scientist would deliver — from raw data to executive-ready output — with minimal hand-holding?*

The human's role was limited to:
- Providing the dataset
- Asking the questions in plain English
- Reviewing and iterating on outputs

Claude handled all code, methodology decisions, chart/table format choices, and written interpretation.

---

## Repo Structure

```
travel-query-eda/
│
├── data/
│   └── travel_search_queries.csv       # Mock dataset — 5,000 travel search queries with category labels
│
├── analysis/
│   ├── eda.py                          # Part 1: Basic EDA — category counts, query length, overlap detection, vague queries
│   ├── eda2.py                         # Part 2: Deep-dive — destinations, temporal signals, price tiers, themes
│   └── generate_report.py             # Generates the self-contained HTML executive report
│
└── output/
    └── executive_report.html           # Final executive report (open in any browser — no dependencies)
```

---

## How to Run

**Requirements:** Python 3.8+, with `pandas`, `matplotlib`, and `numpy`.

```bash
# Install dependencies if needed
pip install pandas matplotlib numpy

# Run Part 1 EDA
python analysis/eda.py

# Run Part 2 deep-dive
python analysis/eda2.py

# Generate the executive HTML report
python analysis/generate_report.py
# → Opens output/executive_report.html in your browser
```

---

## Key Findings (Summary)

| Finding | Headline | Impact |
|---|---|---|
| 1 | ~20% of queries span multiple product categories | High — taxonomy & bundling |
| 2 | 77% of queries name a destination; distribution looks synthetic | High — data quality flag |
| 3 | 14.5% of queries are too vague to route accurately | High — UX & search quality |
| 4 | Only 7.7% include a time signal | Medium — pricing & availability |
| 5 | Only 10% signal a price tier; budget outpaces luxury 2:1 | Medium — merchandising |
| 6 | Female solo travel dominates the solo segment (~85%) | Strategic — underserved segment |

See `output/executive_report.html` for the full narrative, charts, and recommendations.

---

## Data & Methodology Notes

- **Dataset is synthetic/mock.** Destination frequencies (e.g., Maldives, Ulaanbaatar) do not reflect real-world search distribution.
- **Overlap detection is heuristic** (keyword-based per category). Estimated false-positive rate ~5–8%, primarily in the *Book a Car* category.
- **Destination matching** covers ~200 major cities and countries. Smaller or regional destinations are undercounted.
- **No multilingual queries** present — dataset is entirely in ASCII English.

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python / pandas | Data loading and feature engineering |
| re (regex) | Pattern-based signal detection |
| matplotlib | Chart generation |
| HTML + CSS | Self-contained executive report |
| Claude Code (Sonnet 4.6) | End-to-end analysis, code, and interpretation |

---

*Generated with Claude Code — April 2026*
