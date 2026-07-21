"""
Step 3 — macOS man pages scraper for the RAG knowledge base.

Reads man pages directly from the local filesystem — no HTTP requests needed.
Every Mac has these already. Sections 1 (user commands) and 8 (sysadmin) are
the useful ones for a troubleshooting KB.

Standalone usage (exports raw text + man_pages_kb.jsonl):
    python man_pages_scraper.py

Pipeline usage (via main.py):
    python main.py --step 3
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# ---------------------------------------------------------------------------
# Man page directories (sections 1 and 8 only)
# ---------------------------------------------------------------------------

MAN_DIRS = [
    "/usr/share/man/man1",
    "/usr/share/man/man8",
    "/usr/local/share/man/man1",
    "/usr/local/share/man/man8",
    "/Library/Developer/CommandLineTools/usr/share/man/man1",
    "/Library/Developer/CommandLineTools/usr/share/man/man8",
]

# ---------------------------------------------------------------------------
# Known section headers for troff man page parsing
# ---------------------------------------------------------------------------

SECTION_HEADERS = [
    "NAME", "SYNOPSIS", "DESCRIPTION", "OPTIONS", "VERBS",
    "COMMANDS", "SUBCOMMANDS", "EXAMPLES", "FILES", "NOTES",
    "BUGS", "SEE ALSO", "ENVIRONMENT", "EXIT STATUS", "DIAGNOSTICS",
    "HISTORY", "AUTHORS", "RETURN VALUES", "CONFIGURATION",
]

# ---------------------------------------------------------------------------
# Commands worth keeping for a macOS troubleshooting KB
# ---------------------------------------------------------------------------

MACOS_TROUBLESHOOTING_COMMANDS = {
    # networking / wifi
    "networksetup", "ipconfig", "ifconfig", "ping", "traceroute",
    "nslookup", "dig", "netstat", "lsof", "tcpdump", "curl", "airport",
    # bluetooth (brew)
    "blueutil",
    # disk / storage
    "diskutil", "fsck", "hdiutil", "df", "du", "mount", "fdesetup",
    # system
    "launchctl", "sysctl", "nvram", "pmset", "system_profiler",
    "sw_vers", "uname", "hostinfo",
    # logs / diagnostics
    "log", "syslog", "dmesg", "spindump", "sample",
    # processes
    "ps", "top", "kill", "killall", "nice", "renice",
    # permissions / security
    "chmod", "chown", "xattr", "codesign", "spctl", "security", "tccutil",
    # prefs / metadata
    "defaults", "plutil", "mdls", "mdfind", "dscl", "caffeinate",
}

# Values must stay within the controlled vocabulary in knowledge_base.ALLOWED_CATEGORIES
# {bluetooth, wifi, disk, battery, performance, permissions, system, diagnostics, general}.
CATEGORY_MAP = {
    # networking / connectivity → wifi (the user-facing network bucket)
    "networksetup": "wifi",       "ipconfig": "wifi",     "ifconfig": "wifi",
    "airport": "wifi",            "ping": "wifi",         "traceroute": "wifi",
    "nslookup": "wifi",           "dig": "wifi",
    # network inspection / packet tooling → diagnostics
    "netstat": "diagnostics",     "tcpdump": "diagnostics", "lsof": "diagnostics",
    # bluetooth
    "blueutil": "bluetooth",
    # disk / storage
    "diskutil": "disk",           "fsck": "disk",         "hdiutil": "disk",
    "df": "disk",                 "du": "disk",           "mount": "disk",
    # security → permissions
    "fdesetup": "permissions",    "codesign": "permissions", "spctl": "permissions",
    "security": "permissions",    "tccutil": "permissions",
    "chmod": "permissions",       "chown": "permissions", "xattr": "permissions",
    # system control
    "launchctl": "system",        "sysctl": "system",     "nvram": "system",
    # power / battery
    "pmset": "battery",
    # logs / diagnostics
    "log": "diagnostics",         "syslog": "diagnostics","dmesg": "diagnostics",
    "spindump": "diagnostics",    "sample": "diagnostics",
    "system_profiler": "diagnostics",
    # preferences / metadata → system
    "defaults": "system",         "plutil": "system",
    "mdls": "diagnostics",        "mdfind": "diagnostics",
    # processes / load → performance
    "ps": "performance",          "top": "performance",   "kill": "performance",
    "killall": "performance",     "nice": "performance",  "renice": "performance",
}

# ---------------------------------------------------------------------------
# Custom hand-written docs for commands that have poor or missing man pages
# ---------------------------------------------------------------------------

CUSTOM_DOCS = [
    {
        "id":       "manpage_log_custom",
        "command":  "log",
        "title":    "log — unified logging system",
        "one_liner": "Access and query the Apple Unified Logging system",
        "synopsis": (
            "log show [--predicate <filter>] [--last <time>] [--start <date>]\n"
            "log stream [--predicate <filter>] [--level debug|info|default]"
        ),
        "description": (
            "The log command accesses macOS Unified Logging (OSLog). "
            "Use 'log show' to query past logs, 'log stream' to watch in real time. "
            "Filter with --predicate using subsystem, category, or process name. "
            "Logs are binary on disk — 'log show' decodes them automatically. "
            "Useful for diagnosing Bluetooth, Wi-Fi, app crashes, and permission errors."
        ),
        "flags": ["--predicate", "--last", "--start", "--end", "--level",
                  "--process", "--info", "--debug", "--style"],
        "examples": [
            'log show --predicate \'subsystem=="com.apple.bluetooth"\' --last 1h',
            'log show --predicate \'process=="Finder"\' --last 30m',
            'log stream --predicate \'subsystem=="com.apple.network"\' --level debug',
            'log show --last 1h --info > ~/Desktop/system_log.txt',
            'log show --predicate \'eventMessage contains "error"\' --last 2h',
        ],
        "category":       "diagnostics",
        "source":         "macos_man_pages",
        "difficulty_tier": 2,
    },
    {
        "id":       "manpage_defaults_custom",
        "command":  "defaults",
        "title":    "defaults — read/write macOS preference files",
        "one_liner": "Access and modify macOS .plist preference files",
        "synopsis": "defaults [read|write|delete|find] <domain> [key] [value]",
        "description": (
            "defaults reads, writes, and deletes macOS preferences stored as .plist files. "
            "Each app has a domain (usually its bundle ID, e.g. com.apple.finder). "
            "Use 'defaults delete <domain>' to reset an app's prefs without reinstalling. "
            "Use 'defaults read' to inspect current preference values. "
            "Changes take effect after restarting the affected app."
        ),
        "flags": ["read", "write", "delete", "find", "domains",
                  "-g", "-globalDomain", "-currentHost"],
        "examples": [
            "defaults read com.apple.finder",
            "defaults delete com.apple.dock && killall Dock",
            "defaults write com.apple.screensaver askForPassword -int 1",
            "defaults find bluetooth",
            "defaults read NSGlobalDomain AppleFontSize",
        ],
        "category":       "preferences",
        "source":         "macos_man_pages",
        "difficulty_tier": 2,
    },
    {
        "id":       "manpage_tccutil_custom",
        "command":  "tccutil",
        "title":    "tccutil — manage TCC privacy permissions",
        "one_liner": "Reset app TCC privacy permissions (camera, mic, location, etc.)",
        "synopsis": "tccutil reset <service> [bundleID]",
        "description": (
            "tccutil manages the Transparency Consent and Control (TCC) database. "
            "Use it when an app permission is stuck or incorrectly denied in System Settings. "
            "Resetting a service forces macOS to re-prompt for permission on next use. "
            "Services: Camera, Microphone, Location, Contacts, Calendar, Reminders, "
            "Photos, Accessibility, AddressBook, Bluetooth, ScreenCapture, SystemPolicyAllFiles."
        ),
        "flags": ["reset"],
        "examples": [
            "tccutil reset Camera",
            "tccutil reset Microphone com.zoom.xos",
            "tccutil reset All",
            "tccutil reset Bluetooth",
            "tccutil reset ScreenCapture",
        ],
        "category":       "permissions",
        "source":         "macos_man_pages",
        "difficulty_tier": 2,
    },
    {
        "id":       "manpage_pmset_custom",
        "command":  "pmset",
        "title":    "pmset — configure power management",
        "one_liner": "Configure macOS power management and sleep settings",
        "synopsis": (
            "pmset [-a|-b|-c|-u] <setting> <value>\n"
            "pmset -g [live|log|thermlog|ps|rawlog]"
        ),
        "description": (
            "pmset reads and writes power management settings. "
            "Use 'pmset -g' to show current settings, 'pmset -g log' for sleep/wake history. "
            "Flags: -a (all), -b (battery), -c (charger), -u (UPS). "
            "Common settings: sleep (minutes), displaysleep, disksleep, hibernatemode, "
            "autopoweroff, tcpkeepalive, proximitywake."
        ),
        "flags": ["-a", "-b", "-c", "-g", "sleep", "displaysleep",
                  "hibernatemode", "autopoweroff", "tcpkeepalive"],
        "examples": [
            "pmset -g",
            "pmset -g log | grep -i 'sleep\\|wake'",
            "sudo pmset -a sleep 30",
            "sudo pmset -a hibernatemode 0",
            "pmset -g thermlog",
        ],
        "category":       "battery",
        "source":         "macos_man_pages",
        "difficulty_tier": 2,
    },

    # -----------------------------------------------------------------------
    # DIAGNOSTIC docs — symptom → the EXACT command that reveals the cause.
    # These exist so the agentic troubleshooter proposes the RIGHT grounded
    # diagnostic (wdutil/pmset -g assertions/tmutil listlocalsnapshots) instead
    # of generic advice. embed_text is symptom-rich so casual queries retrieve it.
    # -----------------------------------------------------------------------
    {
        "id": "manpage_diag_wifi", "command": "diag_wifi",
        "title": "Wi-Fi keeps dropping — diagnose with wdutil / log show",
        "one_liner": "Find why Wi-Fi disconnects: wdutil info + wifi log predicate",
        "category": "wifi", "difficulty_tier": 3,
        "embed_text": (
            "Wi-Fi keeps dropping, disconnects every few minutes, random Wi-Fi drops on "
            "MacBook even with a strong signal. Diagnose the real cause: run `sudo wdutil info` "
            "— it shows the current SSID, BSSID, RSSI (signal strength in dBm), the 5GHz "
            "channel, and PHY mode. Then check disconnect reasons with "
            "`log show --predicate 'subsystem == \"com.apple.wifi\"' --last 1h` (look for CSA, "
            "'channel switch announcement', 'radar', or disassociation reason codes). Key cause: "
            "5GHz DFS channels (52-144) require radar detection, and when the router detects "
            "radar it forces a Channel Switch Announcement that disconnects the Mac repeatedly "
            "even though the signal is strong. RSSI guide: -30 to -50 excellent, -60 good, -70 "
            "weak, below -80 unusable. Fix: if the router is on a DFS channel, change the 5GHz "
            "channel to a non-DFS channel (36, 40, 44, or 48). The single command that reveals "
            "this is `sudo wdutil info` — run it first."
        ),
        "examples": ["sudo wdutil info",
                     "log show --predicate 'subsystem == \"com.apple.wifi\"' --last 1h"],
        "source": "macos_man_pages",
    },
    {
        "id": "manpage_diag_battery", "command": "diag_battery",
        "title": "Battery drains in sleep — diagnose with pmset -g assertions",
        "one_liner": "Find what prevents sleep: pmset -g assertions + wake reasons",
        "category": "battery", "difficulty_tier": 3,
        "embed_text": (
            "MacBook battery drains overnight while closed and asleep in a bag, battery draining "
            "during sleep, fast battery drain when not in use. Diagnose: run `pmset -g assertions` "
            "— it lists the power assertions currently PREVENTING sleep and exactly which process "
            "holds each one. A 'PreventUserIdleSystemSleep' assertion held by a process (commonly "
            "'bird', the iCloud sync daemon, or 'sharingd'/'powerd') keeps the Mac awake so it "
            "never enters standby, draining the battery overnight. Also run "
            "`pmset -g log | grep -i wake` to see wake reasons (e.g. 'Wake reason: bird'), and "
            "`pmset -g` for standby/hibernatemode/standbydelay settings. Fix: if iCloud's 'bird' "
            "holds a stuck assertion, sign out of iCloud and back in (or restart the process) to "
            "clear it; disable Power Nap and 'wake for network access' if a network process wakes "
            "the Mac."
        ),
        "examples": ["pmset -g assertions", "pmset -g log | grep -i wake", "pmset -g"],
        "source": "macos_man_pages",
    },
    {
        "id": "manpage_diag_disk", "command": "diag_disk",
        "title": "Disk full but space missing — local APFS snapshots (tmutil)",
        "one_liner": "Reclaim 'missing' disk space from Time Machine local snapshots",
        "category": "disk", "difficulty_tier": 2,
        "embed_text": (
            "Disk says almost full but can't find what's using the space, Finder shows far less "
            "used than the disk reports, missing disk space on Mac, large purgeable space. Cause: "
            "local Time Machine APFS snapshots. macOS keeps hourly local snapshots that consume "
            "real disk space Finder undercounts (reported as 'purgeable'). Diagnose: run "
            "`tmutil listlocalsnapshots /` to list local snapshots (com.apple.TimeMachine.<date>); "
            "`diskutil apfs listSnapshots /` also lists them. Fix: reclaim space by thinning or "
            "deleting local snapshots — `sudo tmutil thinlocalsnapshots / 10000000000 4` frees "
            "space, or `tmutil deletelocalsnapshots <YYYY-MM-DD-HHMMSS>` deletes a specific one. "
            "Turning Time Machine off then on also clears local snapshots."
        ),
        "examples": ["tmutil listlocalsnapshots /",
                     "sudo tmutil thinlocalsnapshots / 10000000000 4",
                     "diskutil apfs listSnapshots /"],
        "source": "macos_man_pages",
    },
    {
        "id": "manpage_diag_tcc", "command": "diag_tcc",
        "title": "Permission won't stick — reset TCC with tccutil (service names)",
        "one_liner": "Reset a stuck privacy permission: tccutil reset <Service> <bundle-id>",
        "category": "permissions", "difficulty_tier": 2,
        "embed_text": (
            "App can't access the microphone or camera even though it looks granted, a permission "
            "won't stick, the mic works in some apps but not one specific app, the app is missing "
            "or greyed out in Privacy settings. macOS TCC (Transparency Consent Control) stores "
            "per-app privacy grants; reset a stuck one with `tccutil reset <Service> <bundle-id>`. "
            "Service names: Microphone, Camera, Accessibility, ScreenCapture, SystemPolicyAllFiles "
            "(Full Disk Access), ListenEvent (input monitoring), AppleEvents, Photos. Example: "
            "`tccutil reset Microphone com.example.app`, then relaunch the app to re-trigger the "
            "permission prompt. Find a bundle id with `osascript -e 'id of app \"AppName\"'`. The "
            "TCC database lives at ~/Library/Application Support/com.apple.TCC/TCC.db (SIP-protected)."
        ),
        "examples": ["tccutil reset Microphone com.example.app",
                     "osascript -e 'id of app \"AppName\"'"],
        "source": "macos_man_pages",
    },
    {
        "id": "manpage_diag_panic", "command": "diag_panic",
        "title": "Kernel panic / random restarts — find the kext (kmutil)",
        "one_liner": "Read the .panic backtrace and identify the third-party kext",
        "category": "diagnostics", "difficulty_tier": 3,
        "embed_text": (
            "Mac randomly restarts with a kernel panic, unexpected restart, a message that the "
            "computer restarted because of a problem, panics a few times a day. Read the panic "
            "report to find the culprit: panic logs are at /Library/Logs/DiagnosticReports/*.panic "
            "(open in Console.app). Look for the 'Kernel Extensions in backtrace' section — a "
            "third-party kext listed there (e.g. com.thirdparty.<name>, often from a VPN, "
            "antivirus, or virtualization app) is the likely cause. List loaded third-party kexts "
            "with `kmutil showloaded` (modern macOS) or `kextstat | grep -v com.apple` (older). "
            "Fix: uninstall or disable the third-party kernel extension named in the backtrace; on "
            "Apple Silicon remove its system extension in Settings > General > Login Items & Extensions."
        ),
        "examples": ["ls /Library/Logs/DiagnosticReports/*.panic",
                     "kmutil showloaded", "kextstat | grep -v com.apple"],
        "source": "macos_man_pages",
    },
    {
        "id": "manpage_diag_spotlight", "command": "diag_spotlight",
        "title": "Hot/sluggish, high mds_stores CPU — rebuild Spotlight (mdutil)",
        "one_liner": "Fix a stuck Spotlight reindex pinning the CPU",
        "category": "performance", "difficulty_tier": 2,
        "embed_text": (
            "Mac is hot with loud fans and feels sluggish but nothing heavy is running, high CPU "
            "from mds or mds_stores or mdworker, fans spinning for no reason. Cause: Spotlight is "
            "stuck reindexing — mds/mds_stores/mdworker pin the CPU when the Spotlight index is "
            "corrupt or a reindex never completes (common after copying a large external drive). "
            "Diagnose: check Activity Monitor for sustained mds_stores CPU; `mdutil -s /` shows "
            "indexing status. Fix: rebuild the Spotlight index with `sudo mdutil -i off /` then "
            "`sudo mdutil -i on /` (or `sudo mdutil -E /` to erase and rebuild). You can also "
            "exclude a volume in Settings > Siri & Spotlight > Spotlight Privacy to stop runaway "
            "indexing."
        ),
        "examples": ["mdutil -s /", "sudo mdutil -i off /", "sudo mdutil -i on /"],
        "source": "macos_man_pages",
    },
]


# ---------------------------------------------------------------------------
# Step 1 — export man pages to plain text
# ---------------------------------------------------------------------------

def export_man_page(gz_path: str) -> Optional[Dict]:
    """
    Convert a man page file to plain text (handles both .gz and plain troff).
    Uses the system `man` command — handles all troff/groff formatting.
    """
    filename = Path(gz_path).name
    # strip .gz suffix if present before splitting on "."
    stem    = filename[:-3] if filename.endswith(".gz") else filename
    parts   = stem.split(".")
    command = parts[0]
    section = parts[1] if len(parts) >= 2 else "1"

    try:
        result = subprocess.run(
            ["man", "-P", "cat", command],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        raw_text = result.stdout.strip()
        if len(raw_text) < 200:
            return None
        return {
            "command":    command,
            "section":    section,
            "gz_path":    gz_path,
            "raw_text":   raw_text,
            "char_count": len(raw_text),
        }
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {command}")
        return None
    except Exception as e:
        print(f"  ✗ {command}: {e}")
        return None


def export_all_man_pages(output_dir: str = "./man_pages_raw") -> List[Dict]:
    """
    Export all man pages from the system to plain text files.
    Deduplicates by command name so the same command from multiple dirs
    is only written once.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict] = []
    seen_commands: set = set()

    for man_dir in MAN_DIRS:
        if not os.path.exists(man_dir):
            continue
        section_num = Path(man_dir).name[-1]  # "1" or "8"
        man_files = sorted(
            list(Path(man_dir).glob("*.gz")) +
            list(Path(man_dir).glob(f"*.{section_num}")) +
            list(Path(man_dir).glob(f"*.{section_num}.gz"))
        )
        print(f"\n{man_dir}: {len(man_files)} pages")

        for gz_path in man_files:
            stem    = gz_path.name[:-3] if gz_path.name.endswith(".gz") else gz_path.name
            command = stem.split(".")[0]
            if command in seen_commands:
                continue
            seen_commands.add(command)

            page = export_man_page(str(gz_path))
            if page:
                out_file = Path(output_dir) / f"{page['command']}.{page['section']}.txt"
                out_file.write_text(page["raw_text"])
                results.append(page)
                print(f"  ✓ {page['command']:<20} ({page['char_count']:,} chars)")

    print(f"\nExported {len(results)} man pages to {output_dir}/")
    return results


# ---------------------------------------------------------------------------
# Step 2 — parse sections from raw text
# ---------------------------------------------------------------------------

def parse_man_sections(raw_text: str) -> Dict[str, str]:
    """
    Split raw man page text into a dict of {SECTION_NAME: content}.
    Section headers are all-caps lines at column 0.
    """
    sections: Dict[str, str] = {}
    current_section = "PREAMBLE"
    current_lines: List[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        is_header = (
            stripped in SECTION_HEADERS
            or (re.match(r'^[A-Z][A-Z ]{2,}$', stripped) and line == line.lstrip())
        )
        if is_header and stripped:
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def extract_flags(synopsis: str) -> List[str]:
    """Extract all flags/options from SYNOPSIS section."""
    return sorted(set(re.findall(r'--?[a-zA-Z][\w-]*', synopsis)))


def extract_examples(raw_text: str) -> List[str]:
    """Extract command examples — lines starting with $ or 4-space indent in EXAMPLES."""
    examples: List[str] = []
    in_examples = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped == "EXAMPLES":
            in_examples = True
            continue
        if in_examples:
            if stripped and line == line.lstrip() and stripped.isupper():
                break
            if stripped.startswith("$") or (
                line.startswith("    ") and stripped and not stripped.startswith("#")
            ):
                examples.append(stripped.lstrip("$ "))
    return [e for e in examples if len(e) > 5]


# ---------------------------------------------------------------------------
# Step 3 — build KB-ready documents
# ---------------------------------------------------------------------------

def build_kb_document(raw_page: Dict) -> Optional[Dict]:
    """
    Convert a raw man page export into a KB-ready document.
    Returns None for commands not in MACOS_TROUBLESHOOTING_COMMANDS.
    """
    command = raw_page["command"]
    if command not in MACOS_TROUBLESHOOTING_COMMANDS:
        return None

    sections    = parse_man_sections(raw_page["raw_text"])
    name_section = sections.get("NAME", "")
    synopsis     = sections.get("SYNOPSIS", "")
    description  = sections.get("DESCRIPTION", "")[:800]
    examples     = extract_examples(raw_page["raw_text"])
    flags        = extract_flags(synopsis)

    one_liner = name_section.split("--")[-1].strip() if "--" in name_section else name_section.strip()

    category = normalize_category(CATEGORY_MAP.get(command, "system"))
    embed_text = normalize_embed_text(
        f"Command: {command}. "
        f"{one_liner}. "
        f"Synopsis: {synopsis[:200]}. "
        f"Description: {description}",
        title=f"{command} {one_liner}",
        category=category,
    )

    return {
        "id":             f"manpage_{command}_{raw_page['section']}",
        "command":        command,
        "section":        raw_page["section"],
        "title":          f"{command}({raw_page['section']}) — {one_liner}",
        "one_liner":      one_liner,
        "synopsis":       synopsis,
        "description":    description,
        "flags":          flags,
        "examples":       examples,
        # man pages are version-agnostic (they document the command, not a macOS
        # release) — ["all"], never [], so the version pre-filter keeps them.
        "macos_versions": ["all"],
        "url":            f"local://manpage/{command}.{raw_page['section']}",
        "category":       category,
        "source":         "macos_man_pages",
        "difficulty_tier": 2,
        "embed_text":     embed_text,
    }


# ---------------------------------------------------------------------------
# Pipeline integration — adapts build_kb_document output to KB article schema
# ---------------------------------------------------------------------------

def _doc_to_kb_article(doc: Dict, section_override: Optional[str] = None) -> Dict:
    """
    Convert a build_kb_document() dict (or CUSTOM_DOCS entry) to the full
    KB article schema expected by KnowledgeBase.save_article().
    """
    command = doc["command"]
    section = section_override or doc.get("section", "1")
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    synopsis    = doc.get("synopsis", "")
    description = doc.get("description", "")
    flags       = doc.get("flags", [])
    examples    = doc.get("examples", [])
    one_liner   = doc.get("one_liner", command)

    kb_sections: List[Dict] = [
        {"heading": "Synopsis",    "body": synopsis},
        {"heading": "Description", "body": description},
    ]
    if flags:
        kb_sections.append({"heading": "Flags", "body": "  ".join(flags)})
    if examples:
        kb_sections.append({"heading": "Examples", "body": "\n".join(f"  {e}" for e in examples)})

    article_id = f"MANPAGE_{command.upper()}_{section}"
    category   = normalize_category(doc.get("category", "system"))
    embed_text = normalize_embed_text(
        doc.get("embed_text")
        or f"Command: {command}. {one_liner}. Synopsis: {synopsis[:200]}. Description: {description}",
        title=f"{command} {one_liner}",
        category=category,
    )

    return {
        "id":               f"manpage_{command}_{section}",
        "article_id":       article_id,
        "locale":           "en-us",
        "title":            doc.get("title", f"{command}({section}) — {one_liner}"),
        "url":              f"local://manpage/{command}.{section}",
        "scraped_at":       now,
        "last_modified":    None,
        "affected_devices": ["Mac"],
        "macos_versions":   ["all"],
        "difficulty_tier":  doc.get("difficulty_tier", 2),
        "category":         category,
        "categories":       ["Man Pages", category.title()],
        "summary":          one_liner,
        "sections":         kb_sections,
        "steps":            examples,
        "embed_text":       embed_text,
        "source":           "macos_man_pages",
    }


# ---------------------------------------------------------------------------
# Pipeline scraper class (used by main.py run_step3)
# ---------------------------------------------------------------------------

class ManPageScraper:
    """
    Collects macOS man pages from the local filesystem and saves them to
    the KnowledgeBase using the same schema as SiteMap.py and dev_docs_scraper.py.
    """

    def scrape_all(self, kb: KnowledgeBase) -> int:
        saved = skipped = failed = 0

        print("  Scanning man page directories...")
        seen_commands: set = set()

        for man_dir in MAN_DIRS:
            path = Path(man_dir)
            if not path.exists():
                continue

            section_num = path.name[-1]  # "1" or "8"
            man_files = sorted(
                list(path.glob("*.gz")) +
                list(path.glob(f"*.{section_num}")) +
                list(path.glob(f"*.{section_num}.gz"))
            )
            print(f"  {man_dir}: {len(man_files)} pages")

            for gz in man_files:
                stem    = gz.name[:-3] if gz.name.endswith(".gz") else gz.name
                command = stem.split(".")[0]
                if command in seen_commands or command not in MACOS_TROUBLESHOOTING_COMMANDS:
                    continue
                seen_commands.add(command)

                raw = export_man_page(str(gz))
                if not raw:
                    failed += 1
                    continue

                doc = build_kb_document(raw)
                if not doc:
                    failed += 1
                    continue

                article    = _doc_to_kb_article(doc)
                article_id = article["article_id"]

                if kb.already_scraped(article_id):
                    skipped += 1
                    continue

                kb.save_article(article)
                print(f"  ✓ {command:<22} {doc['one_liner'][:50]}")
                saved += 1

        print("\n  Adding custom-written docs...")
        for custom_doc in CUSTOM_DOCS:
            command    = custom_doc["command"]
            article    = _doc_to_kb_article(custom_doc, section_override="custom")
            article_id = article["article_id"]

            if kb.already_scraped(article_id):
                print(f"  — Already in KB: {command} (custom)")
                skipped += 1
                continue

            kb.save_article(article)
            print(f"  ✓ {command:<22} {custom_doc['one_liner'][:50]}")
            saved += 1

        print(f"\n  Man pages: {saved} saved, {skipped} skipped, {failed} failed")
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point — exports raw text + man_pages_kb.jsonl
# ---------------------------------------------------------------------------

def main():
    # data/ dir lives one level up from this scrapers/ package
    _data_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")

    # 1. export all man pages from the system
    raw_pages = export_all_man_pages(output_dir=_os.path.join(_data_dir, "man_pages_raw"))

    # 2. convert to KB documents (troubleshooting commands only)
    # CUSTOM_DOCS override the system man page for the same command
    custom_commands = {d["command"] for d in CUSTOM_DOCS}
    kb_docs: List[Dict] = []
    for raw in raw_pages:
        if raw["command"] in custom_commands:
            continue  # custom hand-written version takes priority
        doc = build_kb_document(raw)
        if doc:
            kb_docs.append(doc)

    # 3. add the custom-written docs (always included — richer than man pages)
    kb_docs.extend(CUSTOM_DOCS)

    # 4. save to JSONL
    output_file = _os.path.join(_data_dir, "man_pages_kb.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in kb_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nBuilt {len(kb_docs)} man page KB documents → {output_file}")
    print(f"Categories: {sorted(set(d['category'] for d in kb_docs))}")


if __name__ == "__main__":
    main()
