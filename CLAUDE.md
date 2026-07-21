# macOS Troubleshooting RAG System — Project Context

This file orients Claude Code to the project so it doesn't need re-explaining
across sessions. Read this fully before touching scrapers, the knowledge base,
or retrieval code.

## What this project is

An agentic RAG system that diagnoses macOS problems better than a frontier
model would, using a smaller/open-source LLM backed by a deep, narrow,
well-structured knowledge base. The thesis: a 7-8B model with precise
retrieval beats a 70B+ model with generic knowledge, on this domain
specifically. This is a resume project — code quality and architecture
decisions should be defensible in an interview, not just functional.

## Core objectives (do not violate these when making changes)

1. **Beat frontier models on this narrow domain** via KB depth, not model size.
   Never "fix" a weak answer by reaching for a bigger model — fix the KB or
   the retrieval first.
2. **Resolve in 1-2 turns.** The intake prompt must extract macOS version,
   Mac chip (Intel/M1/M2/M3), and what the user has already tried — all from
   one message. Do not design flows that require multiple clarifying
   round-trips by default.
3. **Open-source / low-cost models are a feature, not a compromise.** Default
   target stack: Ollama + Mistral-7B-Instruct or Llama-3.3-70B via Groq for
   speed. Don't add a second specialized model (e.g. a coder model) for
   command generation — grounding via RAG matters more than model choice for
   shell command accuracy. Keep the architecture single-model unless there's
   measured evidence otherwise.
4. **Hybrid retrieval with confidence-based fallback.** RAG is the default
   path. Web search (`web_search_apple` tool, Tavily, scoped to
   apple.com/support and discussions.apple.com) is the fallback when:
   - vector search top score is below the confidence threshold, or
   - the query references a macOS version not covered in the KB.
   Don't call web search speculatively "just in case" — that defeats the
   purpose of having a KB at all.
5. **Data quality is the actual product.** Every other component (agent
   logic, prompts, UI) is replaceable. The KB is not. Default to spending
   more effort on collection/cleaning/tagging than on agent orchestration.
6. **Solution depth triage — the most distinctive feature.** Never let the
   system lead with a solution the user has obviously already tried (restart,
   toggle off/on, basic settings checks). Every KB entry carries a
   `difficulty_tier` and an `implied_already_tried` list. The agent must infer
   what a user has likely already done from how they phrase the problem (e.g.
   someone citing a log path has already tried the GUI toggle) and rank
   solutions accordingly. This is the system's main qualitative edge — protect
   it in every prompt change.

## Architecture (high level)

```
User (web app)
  -> Orchestrator agent (LangChain / LangGraph)
    -> search_knowledge_base   (vector search, metadata pre-filtered)
    -> web_search_apple        (Tavily, fallback only)
    -> suggest_commands        (grounded in man-page KB entries)
    -> classify_issue          (lightweight tagging call)
  -> LLM synthesizes final answer, respecting depth triage
```

Vector store: ChromaDB. Embeddings: `nomic-embed-text` (local) or
`text-embedding-3-small`. LLM: Groq-hosted Llama 3.3 70B for dev, Ollama
local model for the no-cost deployment story.

## Document contract (every KB entry must satisfy this)

```python
{
  "id": str,                 # stable, source-prefixed (e.g. "manpage_diskutil_8")
  "title": str,
  "url": str | None,         # None only for synthetic/custom docs
  "source": str,              # "apple_support" | "apple_user_guide" |
                               # "apple_technotes" | "apple_developer_docs" |
                               # "macos_man_pages" | "ask_different" |
                               # "apple_developer_forums" |
                               # "stackoverflow" | "reddit" |
                               # "github_issues" | "wwdc_transcripts"
  "category": str,            # controlled vocabulary -- see below, singular field
  "difficulty_tier": int,     # 1, 2, or 3 -- see below
  "macos_versions": list[str],  # e.g. ["14.0","14.1"] or ["all"]
  "embed_text": str,          # 100-2000 chars, clean prose, what gets embedded
}
```

`category` and `source` MUST propagate all the way into `chunks.jsonl` (the
ChromaDB payload), not just the article-level JSON. This was a real bug found
in a schema audit -- `base_meta` in `knowledge_base.py` silently dropped both
fields, which breaks metadata-filtered retrieval. Treat any future scraper
addition with the same suspicion: verify fields survive to the chunk level,
don't just check the article JSON.

### Controlled category vocabulary (do not extend ad hoc)

```
bluetooth, wifi, disk, battery, performance,
permissions, system, diagnostics, general
```

Every scraper's category-mapping logic must resolve into exactly this set.
If a scraper's natural categories don't fit (e.g. man pages producing
`network`, `security`, `preferences`, `spotlight`, `processes`), they get
remapped through a shared `normalize_category()` helper -- never hardcoded
per-scraper with off-vocabulary values. If you add a new scraper, write its
category mapping against this helper from day one.

### Difficulty tiers (drives the depth triage system)

- **Tier 1 -- obvious.** Anything a non-technical blog post or a support rep's
  first response would say: restart, toggle off/on, check cable, update
  software, log out/in.
- **Tier 2 -- intermediate.** Requires Terminal comfort: `sudo` commands,
  `tccutil`, `defaults`, reading `log show` output, resetting NVRAM via
  keyboard shortcut.
- **Tier 3 -- deep.** Requires reading system/kernel logs with predicates,
  kernel extensions, SMC resets, recovery mode, code-signing/security
  internals, anything from Tech Notes.

Tag conservatively -- when unsure between two tiers, default to the higher
one. False "this is obvious" tagging is worse than the reverse, because it
risks suppressing genuinely useful advice.

## Data source tiers

### Tier 1 -- authoritative (implemented)
- Apple Support HT articles (`support.apple.com`, sitemap + category crawl)
  → `SiteMap.py`, Step 1
- macOS User Guide (`support.apple.com/guide/mac-help`, tree crawl)
  → `SiteMap.py`, Step 1
- Man pages + synthetic docs for `log`, `defaults`, `tccutil`
  → `man_pages_scraper.py`, Step 3
- Apple Developer Tech Notes TN2xxx (`developer.apple.com/library/archive`)
  and selected framework docs (CoreBluetooth, OSLog, IOKit, etc.)
  → `dev_docs_scraper.py`, Step 2 (`TechNoteScraper`, `FrameworkDocScraper`)
- Apple Developer Tech Notes TN3xxx (modern, `developer.apple.com/documentation/technotes`)
  → `dev_docs_scraper.py`, Step 2 (`ModernTechNoteScraper`, Apple JSON API)

### Tier 2 -- expert community (implemented)
- **Ask Different (Stack Exchange) data dump** -- the single best source.
  Download from the Internet Archive (`apple.stackexchange.com.7z`), extract,
  pass `--dump-dir` to `main.py --step 4`. Filters: `PostTypeId == 1`,
  tagged `macos`/`osx`, `Score >= 3`, `AcceptedAnswerId IS NOT NULL`.
  Produces ~30-50k pre-verified problem→solution pairs.
  → `ask_different_scraper.py`, Step 4
- **Apple Developer Forums** -- Playwright scraper, quality-filtered to threads
  with a marked "Correct" answer, minimum 2 replies, macOS-relevant tags.
  Source value: `"apple_developer_forums"`.
  → `apple_devforums_scraper.py`, Step 5

### Tier 3 -- real user problems, heavily filtered (implemented)
- **r/MacOS + r/applehelp** — symptom vocabulary only.  Self-posts, score ≥ 5,
  num_comments ≥ 3.  Only the problem description (title + selftext) is
  ingested — no Reddit answers (unreliable).  These docs widen the embedding
  space so user-typed casual language retrieves the right Tier 1/2 docs.
  Auth: `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` env vars (optional, raises
  rate limit 30 → 100 req/min).  Source: `"reddit"`.
  → `reddit_scraper.py`, Step 6
- **Stack Overflow (`macos` tag)** — Q&A pairs via SE API v2.3.  Filters:
  score ≥ 2, has accepted answer (answer score ≥ 1).  Two-phase: fetch
  questions (`/search?accepted=True`), batch-fetch answer bodies (`/answers`).
  Auth: `SO_API_KEY` env var (optional, raises daily quota 300 → 10,000).
  Source: `"stackoverflow"`.
  → `stackoverflow_scraper.py`, Step 7

### Tier 4 -- supplementary, architectural context (implemented)
- **GitHub Issues (Homebrew/brew, mas-cli/mas)** — closed issues with ≥ 3
  comments, body ≥ 200 chars, macOS-relevant by keyword or label.  No
  accepted-answer mechanism on GitHub; solution is the highest-reacted comment,
  falling back to resolution-keyword comments, then last substantive comment.
  Auth: `GITHUB_TOKEN` env var (optional, raises 60 → 5,000 req/hour).
  Source: `"github_issues"`.
  → `github_scraper.py`, Step 8
- **WWDC Session Content (2019–2024)** — macOS-relevant sessions explaining
  the architectural *why* behind privacy, security, networking, and system
  changes (TCC design, Network.framework, OSLog, Endpoint Security, etc.).
  Captures Catalina through Sequoia.  Strategy 1: Apple's JSON API (same
  endpoint pattern as TN3xxx).  Strategy 2: HTML scraping of session pages.
  Always difficulty tier 3 (developer/architectural content).
  Source: `"wwdc_transcripts"`.
  → `wwdc_scraper.py`, Step 9

Tier 4 is the lowest retrieval weight.  These sources fill KB gaps that
Tiers 1–3 leave: real tool-breakage patterns (GitHub) and the design intent
behind macOS behaviors (WWDC).  Do not add new Tier 4 sources (e.g. WWDC
transcripts for iOS-only sessions, arbitrary GitHub repos) without first
confirming a concrete retrieval gap.

## Chunking strategy -- non-negotiable rule

**One KB document = one problem-solution unit.** Never chunk by fixed token
count or sliding window. If a source article covers five distinct issues,
split it into five documents at ingestion time, each with its own
`embed_text`, rather than one large chunk or five token-window slices that
each lose context. This is the most common RAG mistake and the one most
likely to silently degrade this project's only real differentiator (KB
quality). When reviewing or extending any scraper, check the chunking output
against this rule first.

## Data cleaning checklist (apply to every source before ingestion)

- `embed_text` must be plain prose: no HTML tags, no nav/cookie-banner text,
  no "Was this article helpful?" footers, no troff formatting artifacts
  (form-feed `\x0c`, doubled-letter headers like `MMNNAAMMEE` from man page
  exports).
- Enforce both bounds on `embed_text`: floor at 100 chars (below this, the
  page had too little content to retrieve well -- drop or flag it), ceiling
  at 2000 chars (longer dilutes the embedding signal -- truncate, don't
  summarize, unless a smarter long-form chunking strategy is in place).
- `macos_versions` must be a list, never a comma-joined string (breaks
  Chroma's `$contains` metadata filter). Version-agnostic content (man pages,
  general guides) gets `["all"]` explicitly -- never an empty list, since an
  empty list silently excludes the doc from every version-filtered query.
- Deduplicate Apple Support articles by article ID across locales, preferring
  `en-us`, before scraping bodies -- don't discover-then-scrape-then-dedupe,
  dedupe URLs first to avoid wasted requests.
- Any new source needs explicit `source` and `category` fields populated
  through `normalize_category()` before it's allowed into `knowledge_base.py`'s
  ingestion path -- no exceptions, even for "just testing" scrapers.

## Retrieval logic (target behavior)

```python
def retrieve(query, macos_version, mac_chip, n=5):
    where = {"$and": [
        {"macos_versions": {"$contains": macos_version}},
        # mac_chip filter only when the KB actually has chip-specific docs;
        # don't over-filter and starve retrieval on sparse categories
    ]}
    results = collection.query(query_texts=[query], where=where, n_results=n)
    if results["distances"][0][0] > CONFIDENCE_THRESHOLD:  # tune empirically
        return web_search_fallback(query)
    return results
```

Confidence threshold should be tuned against a real test set (see Testing
below), not guessed once and left alone.

## Testing / verification expectations

When asked to verify the KB or pipeline, do not process the entire corpus.
Sample deliberately: a few documents per source, plus any source/category
combination that's new or recently changed. Check, in order: (1) does
`embed_text` exist, fall within length bounds, and read as clean prose, (2)
do `category`/`source`/`difficulty_tier`/`macos_versions` survive into
`chunks.jsonl` and not just the article JSON, (3) is the category value a
member of the controlled vocabulary, (4) run a handful of realistic
troubleshooting queries and sanity-check which documents would be retrieved
and whether the ranking makes sense given difficulty tiers.

After any schema-affecting change, treat cached `articles/*.json` and
`index.json` as potentially stale -- confirm whether the chunker reads cached
fields or re-derives them, and re-run ingestion from scratch if there's any
doubt rather than assuming the fix applies retroactively.

## What NOT to do

- Don't add a second LLM (e.g. a code-specialized model) for command
  generation. Diagnose command-accuracy problems as retrieval/grounding
  issues first.
- Don't loosen the chunking rule for convenience -- fixed-size chunking will
  pass tests superficially while quietly destroying retrieval quality.
- Don't invent new categories outside the controlled vocabulary without
  updating this file and `normalize_category()` together.
- Don't let the agent lead a response with a Tier 1 solution unless the
  conversation gives no signal the user has tried it -- re-read the depth
  triage objective before changing prompt templates.
- Don't scrape Tier 2/3 sources live in a tight loop without rate limiting
  and a quality filter (score thresholds, accepted-answer checks) -- unfiltered
  community data will dilute the KB faster than it improves it.