"""
Reddit scraper for the RAG knowledge base.

Sources : r/MacOS, r/applehelp
Purpose : symptom vocabulary bridge.  Real users describe problems in casual
          language.  These documents help retrieval surface the right Tier 1/2
          technical documents for a given user query.

IMPORTANT — no solutions are ingested.
  Reddit answers/comments are NOT included per CLAUDE.md Tier 3 spec.
  Community answers are unreliable; only the problem description (title +
  selftext) is stored.  The RAG agent maps these symptoms to Tier 1/2 KB
  solutions at retrieval time via semantic similarity.

Auth setup (recommended — raises rate limit from ~30 to 100 req/min):
  1. Register an app at https://www.reddit.com/prefs/apps
     Type: script   redirect_uri: http://localhost
  2. Set environment variables:
       export REDDIT_CLIENT_ID=your_client_id
       export REDDIT_CLIENT_SECRET=your_client_secret
  Without these, falls back to the public JSON API (no auth, lower limits).

Quality filters (CLAUDE.md Tier 3 spec):
  - Self (text) posts only — link posts carry no problem description
  - score >= 5       (community-validated that this is a real problem)
  - num_comments >= 3  (engagement signal — people recognised the issue)
  - Not stickied, not NSFW

Document shape : one document per post — title + selftext only.
Source value   : "reddit"
Difficulty     : 1 (user-reported, non-technical vocabulary)

Usage:
  python reddit_scraper.py
  python main.py --step 6
"""

import os
import re
import time
import base64
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_SCORE    = 5
MIN_COMMENTS = 3

# Target subreddits — ordered by relevance
SUBREDDITS = ["MacOS", "applehelp"]

# Fetch up to this many posts per subreddit (across all pages)
MAX_POSTS_PER_SUB = 1000

# Request delay between API calls (seconds)
DELAY = 2.0

USER_AGENT = (
    "MacOS-RAG-KB/1.0 (github.com/researchbot; educational use; "
    "contact via github issues)"
)

# Post flairs to skip (mod posts, off-topic, humor)
SKIP_FLAIRS = {"news", "rumor", "humor", "meme", "meta", "announcement", "weekly"}

# ---------------------------------------------------------------------------
# macOS version extraction
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(
    r"macos\s+(\d+(?:\.\d+)*)"
    r"|(?:sequoia|sonoma|ventura|monterey|big\s*sur|catalina|mojave|high\s*sierra)",
    re.IGNORECASE,
)
_NAME_VERSION = {
    "sequoia": "15", "sonoma": "14", "ventura": "13", "monterey": "12",
    "bigsur": "11", "big-sur": "11",
    "catalina": "10.15", "mojave": "10.14", "highsierra": "10.13",
}


def _extract_versions(text: str) -> List[str]:
    versions = set()
    for m in _VERSION_RE.finditer(text):
        if m.group(1):
            versions.add(m.group(1))
        else:
            name = m.group(0).lower().replace(" ", "").replace("-", "")
            v = _NAME_VERSION.get(name)
            if v:
                versions.add(v)
    return sorted(versions) if versions else ["all"]


# ---------------------------------------------------------------------------
# Category inference (keyword scan on title + selftext)
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORY: Dict[str, str] = {
    "bluetooth":        "bluetooth",
    "airpods":          "bluetooth", "magic keyboard": "bluetooth",
    "wifi":             "wifi",      "wi-fi": "wifi", "wireless": "wifi",
    "internet":         "wifi",      "network": "wifi", "ethernet": "wifi",
    "vpn":              "wifi",
    "disk":             "disk",      "storage": "disk", "apfs": "disk",
    "time machine":     "disk",      "hard drive": "disk", "ssd": "disk",
    "not enough space": "disk",
    "battery":          "battery",   "charging": "battery", "power": "battery",
    "sleep":            "battery",   "wake": "battery",
    "slow":             "performance", "freezing": "performance",
    "spinning":         "performance", "beachball": "performance",
    "cpu":              "performance", "memory": "performance", "ram": "performance",
    "permission":       "permissions", "privacy": "permissions",
    "microphone":       "permissions", "camera": "permissions",
    "location":         "permissions", "screen recording": "permissions",
    "crash":            "diagnostics", "kernel panic": "diagnostics",
    "log":              "diagnostics", "console": "diagnostics",
    "boot":             "system",    "startup": "system", "won't turn on": "system",
    "spotlight":        "system",    "finder": "system", "dock": "system",
    "launchpad":        "system",    "reinstall": "system",
}


def _infer_category(text: str) -> str:
    t = text.lower()
    for kw, cat in _KEYWORD_CATEGORY.items():
        if kw in t:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# Text cleaning — Reddit self-text can be markdown or HTML
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Strip HTML (if any) and collapse whitespace."""
    if raw and "<" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        raw = soup.get_text(separator=" ", strip=True)
    # Remove markdown formatting characters
    raw = re.sub(r"\*\*?|__?|~~|`{1,3}|#{1,6}\s", "", raw)
    # Collapse whitespace
    return " ".join(raw.split())


# ---------------------------------------------------------------------------
# Reddit API client (OAuth app-only or public JSON fallback)
# ---------------------------------------------------------------------------

class RedditClient:
    """
    Thin Reddit API wrapper.

    Uses OAuth app-only auth (client_credentials) when env vars are set;
    falls back to the public JSON API (no auth) otherwise.

    OAuth is recommended:
      export REDDIT_CLIENT_ID=...
      export REDDIT_CLIENT_SECRET=...
    """

    OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_API_BASE  = "https://oauth.reddit.com"
    PUBLIC_API_BASE = "https://www.reddit.com"

    def __init__(
        self,
        client_id:     Optional[str] = None,
        client_secret: Optional[str] = None,
        delay:         float         = DELAY,
    ):
        self.client_id     = client_id     or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.delay         = delay
        self._token:     Optional[str]   = None
        self._token_exp: float           = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    @property
    def _use_oauth(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _ensure_token(self):
        """Refresh the OAuth bearer token if expired."""
        if time.time() < self._token_exp - 60:
            return
        creds = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        r = self.session.post(
            self.OAUTH_TOKEN_URL,
            headers={"Authorization": f"Basic {creds}"},
            data={"grant_type": "client_credentials"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self._token    = data["access_token"]
        self._token_exp = time.time() + data.get("expires_in", 3600)

    def get(self, path: str, **params) -> dict:
        """Make a GET request to the Reddit API."""
        if self._use_oauth:
            self._ensure_token()
            base = self.OAUTH_API_BASE
            headers = {"Authorization": f"Bearer {self._token}"}
        else:
            base    = self.PUBLIC_API_BASE
            headers = {}

        url = f"{base}{path}"
        r   = self.session.get(url, params=params, headers=headers, timeout=15)

        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  [Reddit] Rate limited — waiting {wait}s")
            time.sleep(wait)
            return self.get(path, **params)

        if r.status_code == 403:
            raise RuntimeError(
                "Reddit returned 403 (Forbidden).\n"
                "  Reddit now requires OAuth for API access. Set up credentials:\n"
                "  1. Go to https://www.reddit.com/prefs/apps\n"
                "  2. Create a 'script' app (redirect URI: http://localhost)\n"
                "  3. export REDDIT_CLIENT_ID=<id>\n"
                "     export REDDIT_CLIENT_SECRET=<secret>\n"
                "  Then re-run: python main.py --step 6 --force"
            )

        r.raise_for_status()
        return r.json()

    def top_posts(self, subreddit: str, max_posts: int = MAX_POSTS_PER_SUB) -> Iterator[dict]:
        """
        Yield top self-posts from a subreddit, paginated via 'after'.
        Applies quality filters inline (score, comment count, stickied, etc.).
        """
        after  = None
        count  = 0
        fetched = 0

        while fetched < max_posts:
            params: dict = {"t": "all", "limit": 100}
            if after:
                params["after"] = after

            try:
                data    = self.get(f"/r/{subreddit}/top.json", **params)
                listing = data.get("data", {})
            except Exception as e:
                print(f"  ✗ /r/{subreddit}/top: {e}")
                break

            posts = listing.get("children", [])
            if not posts:
                break

            for post_wrap in posts:
                post = post_wrap.get("data", {})
                fetched += 1

                # Skip non-self (link) posts — no problem description
                if not post.get("is_self"):
                    continue
                # Skip low-score or low-engagement posts
                if post.get("score", 0) < MIN_SCORE:
                    continue
                if post.get("num_comments", 0) < MIN_COMMENTS:
                    continue
                # Skip stickied mod posts and NSFW
                if post.get("stickied") or post.get("over_18"):
                    continue
                # Skip known off-topic flairs
                flair = (post.get("link_flair_text") or "").lower()
                if any(s in flair for s in SKIP_FLAIRS):
                    continue
                # Require at least some body text
                if not (post.get("selftext") or "").strip():
                    continue

                count += 1
                yield post

            after = listing.get("after")
            if not after:
                break

            time.sleep(self.delay)

        print(f"  /r/{subreddit}: yielded {count} qualifying posts from {fetched} fetched")


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_kb_document(post: dict, subreddit: str) -> Optional[Dict]:
    """Convert a Reddit post dict to a KB-ready article."""
    post_id   = post["id"]
    title     = _clean_text(post.get("title", ""))
    selftext  = _clean_text(post.get("selftext") or post.get("selftext_html", ""))

    if not title:
        return None

    # Symptom vocabulary = title + body (no solution)
    full_text  = f"{title}. {selftext}".strip() if selftext else title
    category   = normalize_category(_infer_category(full_text))
    versions   = _extract_versions(full_text)

    embed_text = normalize_embed_text(full_text, title=title, category=category)
    if len(embed_text) < 100:
        return None

    created = post.get("created_utc")
    created_dt = (
        datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if created else None
    )

    return {
        "id":               f"reddit_{post_id}",
        "article_id":       f"REDDIT_{post_id.upper()}",
        "locale":           "en-us",
        "title":            title,
        "url":              f"https://www.reddit.com{post.get('permalink', '')}",
        "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_modified":    created_dt,
        "affected_devices": ["Mac"],
        "macos_versions":   versions,
        # Symptom vocabulary docs: difficulty reflects what the problem likely
        # requires, not user expertise.  Default 1 — these are user-reported
        # issues at the "I tried restarting" level of description.
        "difficulty_tier":  1,
        "category":         category,
        "categories":       [f"r/{subreddit}", "Community"],
        "summary":          full_text[:300],
        "sections": [
            {"heading": "User Problem", "body": full_text},
        ],
        "steps":      [],
        "embed_text": embed_text,
        "source":     "reddit",
    }


# ---------------------------------------------------------------------------
# Scraper class — pipeline integration
# ---------------------------------------------------------------------------

class RedditScraper:
    """
    Fetches qualifying self-posts from r/MacOS and r/applehelp and writes
    symptom-vocabulary documents to the KnowledgeBase.

    Interface: scrape_all(kb) → int (saved count).
    """

    def __init__(
        self,
        subreddits:    List[str]    = SUBREDDITS,
        max_per_sub:   int          = MAX_POSTS_PER_SUB,
        client_id:     Optional[str] = None,
        client_secret: Optional[str] = None,
        delay:         float         = DELAY,
    ):
        self.subreddits  = subreddits
        self.max_per_sub = max_per_sub
        self.client      = RedditClient(
            client_id=client_id,
            client_secret=client_secret,
            delay=delay,
        )

    def scrape_all(self, kb: KnowledgeBase) -> int:
        using_oauth = self.client._use_oauth
        print(
            f"  Auth: {'OAuth app-only (client_credentials)' if using_oauth else 'public JSON API (no auth)'}"
        )
        if not using_oauth:
            print(
                "  Tip: set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET for higher rate limits.\n"
                "  Register at https://www.reddit.com/prefs/apps"
            )

        saved = skipped = failed = 0

        for sub in self.subreddits:
            print(f"\n  Subreddit: r/{sub}")
            for post in self.client.top_posts(sub, max_posts=self.max_per_sub):
                article_id = f"REDDIT_{post['id'].upper()}"
                if kb.already_scraped(article_id):
                    skipped += 1
                    continue

                doc = build_kb_document(post, sub)
                if not doc:
                    failed += 1
                    continue

                kb.save_article(doc)
                saved += 1
                if saved % 200 == 0:
                    print(f"  ... {saved:,} saved")

        print(f"\n  Reddit: {saved:,} saved, {skipped:,} skipped, {failed:,} failed")
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Reddit macOS posts into the RAG knowledge base"
    )
    parser.add_argument(
        "--subreddits", nargs="+", default=SUBREDDITS,
        help="Subreddits to scrape (default: MacOS applehelp)",
    )
    parser.add_argument(
        "--max-per-sub", type=int, default=MAX_POSTS_PER_SUB,
        help=f"Max posts per subreddit (default: {MAX_POSTS_PER_SUB})",
    )
    parser.add_argument(
        "--min-score", type=int, default=MIN_SCORE,
        help=f"Minimum post score (default: {MIN_SCORE})",
    )
    args = parser.parse_args()

    import reddit_scraper as _self
    _self.MIN_SCORE = args.min_score

    kb      = KnowledgeBase()
    scraper = RedditScraper(subreddits=args.subreddits, max_per_sub=args.max_per_sub)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
