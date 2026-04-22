# Lessons Learned

## Session: 2026-04-21

### L001 — Straits Times XLSX has no article text
**Pattern:** The XLSX dataset contains metadata and URLs only — no embedded article text.
**Fix:** Must scrape article text from live URLs using BeautifulSoup before LLM extraction can run.
**Rule:** Always inspect data files for actual content before assuming completeness.

### L002 — Groundsource data does not cover Singapore
**Pattern:** User initially asked about Groundsource relevance; upon investigation it confirmed the parquet does not cover Singapore.
**Fix:** Completely excluded from pipeline. Only local sources (Straits Times + PUB Telegram) are used as flood labels.
**Rule:** Verify geographic coverage of any dataset before designing architecture around it.

### L003 — OpenAI API is the primary LLM provider
**Pattern:** User requested OpenAI as primary, Anthropic as optional fallback.
**Fix:** Provider is configurable in config.yaml (one line change). Both clients are loaded via src/utils.py only.
**Rule:** Never hardcode API provider or key. Use config + .env pattern.

### L004 — API keys must never be committed to Git
**Pattern:** User plans to publish source code to GitHub.
**Fix:** All keys in .env (gitignored). .env.example committed with placeholder values. Pre-commit hook scans for sk- patterns.
**Rule:** Every new script that uses an API must import from src/utils.py — never os.environ directly inline.

### L005 — Notebooks should be .ipynb (VS Code), not Google Colab
**Pattern:** User explicitly does not want Google Colab.
**Fix:** All notebooks are standard .ipynb format, VS Code Jupyter extension compatible.
**Rule:** Never link to or reference Google Colab in documentation for this project.
