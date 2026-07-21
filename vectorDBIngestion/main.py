"""
MacOS Agent — RAG Knowledge Base Pipeline

Tier 1 — authoritative Apple sources
  Step 1  SiteMap.py         Apple Support articles            (support.apple.com)
  Step 2  dev_docs_scraper   Developer docs: TN2xxx + TN3xxx   (developer.apple.com)
  Step 3  man_pages_scraper  macOS man pages                   (local filesystem)

Tier 2 — expert community sources (high quality, pre-filtered)
  Step 4  ask_different_scraper    Ask Different SE data dump (offline XML)
  Step 5  apple_devforums_scraper  Apple Developer Forums (Playwright)

Tier 3 — real user problems, heavily filtered
  Step 6  reddit_scraper       r/MacOS + r/applehelp (symptom vocabulary)
  Step 7  stackoverflow_scraper  Stack Overflow macos tag (SE API v2.3)

Tier 4 — supplementary context, fills architectural gaps
  Step 8  github_scraper   GitHub issues: Homebrew/brew, mas-cli/mas
  Step 9  wwdc_scraper     WWDC session content (2019–2024)

Same-day guard: each step is skipped automatically if it already ran today
(UTC). Override with --force to run regardless.

Run everything:
    python main.py

Run only one step:
    python main.py --step 1
    python main.py --step 4 --dump-dir ./se_dump/

Limit article count during testing:
    python main.py --limit 20
    python main.py --step 2 --limit 10

Force re-run even if already ran today:
    python main.py --force
    python main.py --step 1 --force

Step 4 requires the Ask Different data dump:
    Download: https://archive.org/download/stackexchange/apple.stackexchange.com.7z
    Extract and pass: --dump-dir /path/to/extracted/
"""

import argparse
from datetime import datetime, timezone

from knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# Rebuild — regenerate index + chunks from existing article files
# ---------------------------------------------------------------------------

def run_rebuild(kb: KnowledgeBase):
    print("=" * 60)
    print("REBUILD — regenerating index + chunks from disk articles")
    print("  Patches missing source/category fields inline.")
    print("  No network requests.")
    print("=" * 60)
    print()
    saved = kb.rebuild_from_articles()
    print()
    print_summary(kb)
    return saved


# ---------------------------------------------------------------------------
# Step 1 — Apple Support articles
# ---------------------------------------------------------------------------

def run_step1(kb: KnowledgeBase, limit: int | None = None):
    from scrapers.sitemap import AppleSupportScraper

    scraper = AppleSupportScraper(delay_seconds=1.0)

    print("=" * 60)
    print("STEP 1 — APPLE SUPPORT ARTICLES  (support.apple.com)")
    print("=" * 60)

    # 1a — robots.txt compliance
    print("\n[1a] Checking robots.txt...")
    scraper.check_robots_txt()

    # 1b — discover URLs from category pages (numeric-ID articles)
    print("\n[1b] Crawling category pages...")
    category_urls = scraper.discover_urls_from_categories()
    print(f"  → {len(category_urls)} URLs from category pages")

    # 1c — discover HT article URLs from XML sitemaps
    print("\n[1c] Fetching XML sitemaps (HT articles)...")
    sitemap_urls = scraper.discover_urls_from_xml_sitemaps()
    print(f"  → {len(sitemap_urls)} URLs from XML sitemaps")

    # 1d — merge and deduplicate by article ID (prefer en-us)
    all_urls = list(dict.fromkeys(category_urls + sitemap_urls))
    deduped_urls = scraper.deduplicate_by_article_id(all_urls)
    print(f"\n[1d] After dedup: {len(deduped_urls)} unique articles")

    # 1e — scrape
    urls_to_scrape = deduped_urls[:limit] if limit else deduped_urls
    print(f"\n[1e] Scraping {len(urls_to_scrape)} articles...")

    scraped = skipped = failed = 0
    for url in urls_to_scrape:
        art_id = scraper._article_id_from_url(url)
        if kb.already_scraped(art_id):
            print(f"  — Already in KB: {art_id}")
            skipped += 1
            continue
        article = scraper.scrape_article(url)
        if article:
            kb.save_article(article)
            scraped += 1
        else:
            failed += 1

    kb.save_run_metadata({
        "step":            1,
        "run_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category_urls":   len(category_urls),
        "sitemap_urls":    len(sitemap_urls),
        "unique_articles": len(deduped_urls),
        "scraped":         scraped,
        "skipped":         skipped,
        "failed":          failed,
    })

    print(f"\n  Step 1 done — {scraped} scraped, {skipped} skipped, {failed} failed")
    return scraped


# ---------------------------------------------------------------------------
# Step 2 — Apple Developer Tech Notes + framework guide pages
# ---------------------------------------------------------------------------

def run_step2(kb: KnowledgeBase, limit: int | None = None):
    from scrapers.dev_docs_scraper import TechNoteScraper, FrameworkDocScraper, ModernTechNoteScraper

    print("=" * 60)
    print("STEP 2 — APPLE DEVELOPER DOCS  (developer.apple.com)")
    print("=" * 60)

    # 2a — TN2xxx Tech Notes from the static archive (no extra deps)
    print("\n[2a] Scraping Tech Notes TN2xxx (static archive)...")
    tn_scraper = TechNoteScraper(delay_seconds=1.2)
    tn_saved = tn_scraper.scrape_all(kb, limit=limit)

    # 2b — Framework guide pages via Playwright (install separately)
    print("\n[2b] Scraping framework guide pages (Playwright)...")
    fw_scraper = FrameworkDocScraper(delay_seconds=2.0)
    fw_saved = fw_scraper.scrape_all(kb)

    # 2c — TN3xxx Modern Tech Notes via Apple JSON API (no Playwright needed)
    print("\n[2c] Scraping Tech Notes TN3xxx (Apple JSON API)...")
    tn3_scraper = ModernTechNoteScraper(delay_seconds=1.5)
    tn3_saved = tn3_scraper.scrape_all(kb, limit=limit)

    kb.save_run_metadata({
        "step":               2,
        "run_at":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tn2xxx_saved":       tn_saved,
        "framework_saved":    fw_saved,
        "tn3xxx_saved":       tn3_saved,
    })

    total = tn_saved + fw_saved + tn3_saved
    print(f"\n  Step 2 done — {tn_saved} TN2xxx, {fw_saved} framework pages, {tn3_saved} TN3xxx")
    return total


# ---------------------------------------------------------------------------
# Step 3 — macOS man pages (local filesystem, no HTTP)
# ---------------------------------------------------------------------------

def run_step3(kb: KnowledgeBase):
    from scrapers.man_pages_scraper import ManPageScraper

    print("=" * 60)
    print("STEP 3 — macOS MAN PAGES  (local filesystem)")
    print("=" * 60)
    print()

    scraper = ManPageScraper()
    saved = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":    3,
        "run_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":   saved,
    })

    print(f"\n  Step 3 done — {saved} man page docs added")
    return saved


# ---------------------------------------------------------------------------
# Step 4 — Ask Different (Stack Exchange offline dump)
# ---------------------------------------------------------------------------

def run_step4(kb: KnowledgeBase, dump_dir: str | None = None):
    from scrapers.ask_different_scraper import AskDifferentScraper

    print("=" * 60)
    print("STEP 4 — ASK DIFFERENT  (apple.stackexchange.com SE dump)")
    print("=" * 60)

    if not dump_dir:
        print(
            "\n  ✗ --dump-dir is required for Step 4.\n"
            "  Download the dump from:\n"
            "    https://archive.org/download/stackexchange/apple.stackexchange.com.7z\n"
            "  Extract it, then run:\n"
            "    python main.py --step 4 --dump-dir /path/to/extracted/\n"
        )
        return 0

    scraper = AskDifferentScraper(dump_dir=dump_dir)
    saved   = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":     4,
        "run_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dump_dir": dump_dir,
        "saved":    saved,
    })

    print(f"\n  Step 4 done — {saved:,} Ask Different Q&A pairs added")
    return saved


# ---------------------------------------------------------------------------
# Step 5 — Apple Developer Forums (Playwright)
# ---------------------------------------------------------------------------

def run_step5(kb: KnowledgeBase):
    from scrapers.apple_devforums_scraper import AppleDevForumsScraper

    print("=" * 60)
    print("STEP 5 — APPLE DEVELOPER FORUMS  (developer.apple.com/forums)")
    print("=" * 60)
    print()

    scraper = AppleDevForumsScraper(delay_seconds=2.5)
    saved   = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":   5,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":  saved,
    })

    print(f"\n  Step 5 done — {saved} Apple Dev Forum threads added")
    return saved


# ---------------------------------------------------------------------------
# Step 6 — Reddit (r/MacOS + r/applehelp, symptom vocabulary)
# ---------------------------------------------------------------------------

def run_step6(kb: KnowledgeBase):
    from scrapers.reddit_scraper import RedditScraper

    print("=" * 60)
    print("STEP 6 — REDDIT  (r/MacOS + r/applehelp, symptom vocabulary)")
    print("=" * 60)
    print()
    print("  Auth: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env vars (optional)")
    print("  Ingesting: problem descriptions only (no solutions per Tier 3 spec)")
    print()

    scraper = RedditScraper()
    saved   = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":   6,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":  saved,
    })

    print(f"\n  Step 6 done — {saved:,} Reddit symptom-vocabulary posts added")
    return saved


# ---------------------------------------------------------------------------
# Step 7 — Stack Overflow (macos tag, Q&A pairs)
# ---------------------------------------------------------------------------

def run_step7(kb: KnowledgeBase, limit: int | None = None):
    from scrapers.stackoverflow_scraper import StackOverflowScraper

    print("=" * 60)
    print("STEP 7 — STACK OVERFLOW  (macos tag, SE API v2.3)")
    print("=" * 60)
    print()
    print("  Auth: SO_API_KEY env var (optional, raises quota 300→10,000/day)")
    print()

    max_pages = (limit // 100 + 1) if limit else 50
    scraper   = StackOverflowScraper(max_pages=max_pages)
    saved     = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":   7,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":  saved,
    })

    print(f"\n  Step 7 done — {saved:,} Stack Overflow Q&A pairs added")
    return saved


# ---------------------------------------------------------------------------
# Step 8 — GitHub Issues (Homebrew + macOS-adjacent repos)
# ---------------------------------------------------------------------------

def run_step8(kb: KnowledgeBase, limit: int | None = None):
    from scrapers.github_scraper import GitHubScraper

    print("=" * 60)
    print("STEP 8 — GITHUB ISSUES  (Homebrew/brew, mas-cli/mas)")
    print("=" * 60)
    print()
    print("  Auth: GITHUB_TOKEN env var (optional, raises 60→5,000 req/hour)")
    print("  Filter: closed issues, body≥200 chars, ≥3 comments, macOS-relevant")
    print()

    max_pages = (limit // 100 + 1) if limit else 30
    scraper   = GitHubScraper(max_pages=max_pages)
    saved     = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":   8,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":  saved,
    })

    print(f"\n  Step 8 done — {saved:,} GitHub issue Q&A pairs added")
    return saved


# ---------------------------------------------------------------------------
# Step 9 — WWDC Session Transcripts (developer.apple.com)
# ---------------------------------------------------------------------------

def run_step9(kb: KnowledgeBase):
    from scrapers.wwdc_scraper import WWDCScraper

    print("=" * 60)
    print("STEP 9 — WWDC SESSIONS  (developer.apple.com, 2019–2024)")
    print("=" * 60)
    print()
    print("  Scraping macOS-relevant WWDC sessions (Catalina → Sequoia)")
    print("  Purpose: architectural 'why' context for macOS privacy/security changes")
    print()

    scraper = WWDCScraper()
    saved   = scraper.scrape_all(kb)

    kb.save_run_metadata({
        "step":   9,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved":  saved,
    })

    print(f"\n  Step 9 done — {saved:,} WWDC session documents added")
    return saved


# ---------------------------------------------------------------------------
# Same-day guard
# ---------------------------------------------------------------------------

def _should_skip(kb: KnowledgeBase, step: int, force: bool) -> bool:
    if force:
        return False
    if kb.was_run_today(step=step):
        print(f"\n  ⊘  Step {step} already ran today — skipping.")
        print(     "     Use --force to override.")
        return True
    return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(kb: KnowledgeBase):
    stats = kb.stats()
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE SUMMARY")
    print("=" * 60)
    print(f"  Total articles : {stats['total_articles']:,}")
    print(f"  Total chunks   : {stats['total_chunks']:,}")
    print(f"  KB root        : {stats['kb_root']}/")
    print("    articles/     — full article JSON per file")
    print("    chunks.jsonl  — RAG-ready chunks for embedding")
    print("    index.json    — fast article lookup")
    print("    run_metadata  — scrape history")

    # Per-source breakdown from the index
    from collections import Counter
    index = kb._index
    by_source: Counter = Counter(meta.get("source", "unknown") for meta in index.values())
    if by_source:
        print("\n  Articles by source:")
        for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"    {source:<30} {count:>6,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the MacOS Agent RAG knowledge base"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        default=None,
        help=(
            "Run only one step: "
            "1=Support articles, 2=Developer docs + Tech Notes, 3=Man pages, "
            "4=Ask Different dump, 5=Apple Dev Forums, "
            "6=Reddit symptom vocab, 7=Stack Overflow, "
            "8=GitHub issues, 9=WWDC sessions. "
            "Omit to run all steps."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to scrape per step (useful for test runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if this step already ran today.",
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        help=(
            "Path to extracted apple.stackexchange.com SE dump directory. "
            "Required for --step 4. "
            "Download from: https://archive.org/download/stackexchange/apple.stackexchange.com.7z"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Rebuild chunks.jsonl and index.json from existing article files. "
            "Patches missing source/category fields without re-scraping. "
            "Run this after any schema change."
        ),
    )
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.rebuild:
        run_rebuild(kb)
        return

    if args.step == 1:
        if not _should_skip(kb, 1, args.force):
            run_step1(kb, limit=args.limit)
    elif args.step == 2:
        if not _should_skip(kb, 2, args.force):
            run_step2(kb, limit=args.limit)
    elif args.step == 3:
        if not _should_skip(kb, 3, args.force):
            run_step3(kb)
    elif args.step == 4:
        if not _should_skip(kb, 4, args.force):
            run_step4(kb, dump_dir=args.dump_dir)
    elif args.step == 5:
        if not _should_skip(kb, 5, args.force):
            run_step5(kb)
    elif args.step == 6:
        if not _should_skip(kb, 6, args.force):
            run_step6(kb)
    elif args.step == 7:
        if not _should_skip(kb, 7, args.force):
            run_step7(kb, limit=args.limit)
    elif args.step == 8:
        if not _should_skip(kb, 8, args.force):
            run_step8(kb, limit=args.limit)
    elif args.step == 9:
        if not _should_skip(kb, 9, args.force):
            run_step9(kb)
    else:
        # Full pipeline — each step guards itself independently.
        # Tier 1: steps 1-3 (always run)
        if not _should_skip(kb, 1, args.force):
            run_step1(kb, limit=args.limit)
        print()
        if not _should_skip(kb, 2, args.force):
            run_step2(kb, limit=args.limit)
        print()
        if not _should_skip(kb, 3, args.force):
            run_step3(kb)
        print()
        # Tier 2: steps 4-5
        if args.dump_dir:
            if not _should_skip(kb, 4, args.force):
                run_step4(kb, dump_dir=args.dump_dir)
            print()
        else:
            print("  ⊘  Step 4 skipped — no --dump-dir provided (Ask Different SE dump).")
        if not _should_skip(kb, 5, args.force):
            run_step5(kb)
        print()
        # Tier 3: steps 6-7
        if not _should_skip(kb, 6, args.force):
            run_step6(kb)
        print()
        if not _should_skip(kb, 7, args.force):
            run_step7(kb, limit=args.limit)
        print()
        # Tier 4: steps 8-9
        if not _should_skip(kb, 8, args.force):
            run_step8(kb, limit=args.limit)
        print()
        if not _should_skip(kb, 9, args.force):
            run_step9(kb)

    print_summary(kb)


if __name__ == "__main__":
    main()
