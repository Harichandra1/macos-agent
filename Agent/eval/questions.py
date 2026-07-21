"""
questions.py — benchmark question set for the head-to-head "beat the frontier
model" evaluation.

Each question is a full, user-style macOS troubleshooting message (not a bare
search query) plus the metadata a grader needs:

  id            stable id
  prompt        what the user actually types
  macos_version numeric version string or None
  chip          "apple_silicon" | "intel" | None
  category      controlled-vocab category (for reference / filtering)
  tier          the difficulty tier a *good* answer should operate at (1-3)
  depth_triage  True when the prompt signals the user has already done the
                obvious Tier-1 steps — the system's distinctive edge is NOT
                leading with those. Frontier models with no context tend to
                fail exactly here, which is why these are over-represented.
  rubric        concrete points a correct, useful answer must contain. The
                LLM judges score answers against this list.

The set intentionally spans every category, all three tiers, version-specific
and version-agnostic problems, and a couple of edge cases (vague + off-domain).
"""

from dataclasses import dataclass, field


@dataclass
class BenchmarkQuestion:
    id: str
    prompt: str
    category: str | None
    tier: int | None
    depth_triage: bool
    rubric: list[str]
    macos_version: str | None = None
    chip: str | None = None
    tags: list[str] = field(default_factory=list)


QUESTIONS: list[BenchmarkQuestion] = [
    # ------------------------------------------------------------------ WiFi
    BenchmarkQuestion(
        id="wifi_drops_tried",
        prompt=(
            "My MacBook Pro drops Wi-Fi every 10-15 minutes. I've already "
            "restarted, toggled Wi-Fi off and on, forgotten the network and "
            "rejoined, and rebooted the router. Still happening. What now?"
        ),
        category="wifi", tier=2, depth_triage=True, macos_version="14",
        chip="apple_silicon",
        rubric=[
            "Does NOT re-suggest restart / toggle Wi-Fi / forget network (already tried)",
            "Suggests removing/regenerating Wi-Fi config: com.apple.wifi/network preference plists "
            "(e.g. /Library/Preferences/SystemConfiguration NetworkInterfaces / preferences.plist)",
            "Mentions creating a new network Location or a fresh Wi-Fi service",
            "Mentions inspecting Wi-Fi logs (wifi disconnect reason / log show or sudo wdutil)",
            "Concrete commands or exact Settings paths, not vague advice",
        ],
    ),
    BenchmarkQuestion(
        id="wifi_5ghz_only",
        prompt=(
            "My Mac connects fine to the 2.4GHz band but refuses to join the "
            "5GHz SSID from the same router. Other devices join 5GHz fine."
        ),
        category="wifi", tier=2, depth_triage=False,
        rubric=[
            "Addresses band/channel/region mismatch (DFS channels, router channel width, country/region)",
            "Mentions checking router 5GHz channel (avoid DFS 52-144) or channel width",
            "Suggests creating a new network location / renewing DHCP as concrete steps",
            "Does not hallucinate a macOS 'prefer 5GHz' toggle that doesn't exist",
        ],
    ),
    BenchmarkQuestion(
        id="wifi_log_predicate",
        prompt=(
            "I want to see exactly why my Wi-Fi disconnects. What log show "
            "predicate or command gives me the disconnect reason codes on macOS?"
        ),
        category="wifi", tier=3, depth_triage=True,
        rubric=[
            "Gives a real command: `log show`/`log stream` with a predicate, or `sudo wdutil log`",
            "Predicate targets the Wi-Fi subsystem (e.g. subsystem == \"com.apple.wifi\") "
            "or process airportd",
            "Explains how to read disconnect reason codes",
            "Command syntax is correct and copy-pasteable",
        ],
    ),
    # ------------------------------------------------------------- Bluetooth
    BenchmarkQuestion(
        id="airpods_switch",
        prompt=(
            "My AirPods keep disconnecting from my Mac and jumping to my iPhone "
            "even when I'm actively using the Mac."
        ),
        category="bluetooth", tier=1, depth_triage=False,
        rubric=[
            "Explains automatic device switching and where to disable it "
            "(AirPods settings > Connect to This Mac > When Last Connected to This Mac)",
            "Correct Settings/menu path for the installed macOS",
            "Optionally mentions disabling Handoff as a contributing factor",
        ],
    ),
    BenchmarkQuestion(
        id="bt_reset_module",
        prompt=(
            "Bluetooth on my Mac is totally flaky — devices won't pair and the "
            "menu shows 'not available'. I've restarted and toggled it already. "
            "I'm comfortable in Terminal."
        ),
        category="bluetooth", tier=2, depth_triage=True,
        rubric=[
            "Does NOT lead with restart / toggle Bluetooth (already tried)",
            "Suggests removing the Bluetooth plist "
            "(com.apple.Bluetooth.plist) and rebooting, or resetting the module",
            "Mentions the hidden Shift-Option Bluetooth menu OR blueutil / pkill bluetoothd where appropriate",
            "Gives exact file path / command",
        ],
    ),
    # ------------------------------------------------------------------ Disk
    BenchmarkQuestion(
        id="ext_drive_no_mount",
        prompt=(
            "My external SSD doesn't show up in Finder. In Disk Utility it "
            "appears greyed out and the Mount button does nothing."
        ),
        category="disk", tier=2, depth_triage=False,
        rubric=[
            "Suggests `diskutil list` to confirm the device is seen",
            "Suggests `diskutil mount /dev/diskN` or First Aid",
            "Mentions filesystem/permission causes (NTFS read-only, corrupt catalog)",
            "Escalation path: First Aid, then `fsck`/`diskutil repairVolume`",
        ],
    ),
    BenchmarkQuestion(
        id="apfs_repair",
        prompt=(
            "diskutil verifyVolume is reporting APFS errors on my data volume. "
            "How do I actually repair it safely?"
        ),
        category="disk", tier=2, depth_triage=True,
        rubric=[
            "Gives `diskutil repairVolume /dev/diskN` (or First Aid) as the repair step",
            "Recommends running from Recovery / unmounted for the container when needed",
            "Warns to back up first",
            "Correct diskutil syntax",
        ],
    ),
    BenchmarkQuestion(
        id="tm_stuck_preparing",
        prompt=(
            "Time Machine has been stuck on 'Preparing backup…' for hours on "
            "the very first backup to a new drive."
        ),
        category="disk", tier=1, depth_triage=False,
        rubric=[
            "Mentions letting the deep-scan / indexing finish or excluding large folders",
            "Suggests removing .inProgress file or resetting the backup / re-adding disk",
            "Mentions Spotlight indexing interaction (mdutil) as a cause",
        ],
    ),
    # --------------------------------------------------------------- Battery
    BenchmarkQuestion(
        id="battery_drain_sonoma",
        prompt=(
            "Since updating to Sonoma my M2 MacBook Air battery drains way "
            "faster, even asleep in my bag. I've already checked Battery "
            "settings and closed heavy apps."
        ),
        category="battery", tier=2, depth_triage=True, macos_version="14",
        chip="apple_silicon",
        rubric=[
            "Does NOT just say 'check Battery settings / close apps' (already done)",
            "Suggests inspecting sleep/wake with `pmset -g log` or Wake reasons",
            "Mentions `pmset` power settings (standby/hibernatemode/powernap) or disabling Power Nap / wake-for-network",
            "Mentions checking 'powerd'/assertions preventing sleep (pmset -g assertions)",
            "Concrete commands",
        ],
    ),
    BenchmarkQuestion(
        id="pmset_hibernate",
        prompt=(
            "How do I check and change hibernation / standby behaviour on my "
            "MacBook from the command line?"
        ),
        category="battery", tier=2, depth_triage=False,
        rubric=[
            "Uses `pmset -g` to read settings and `sudo pmset` to change",
            "References hibernatemode / standby / standbydelay keys correctly",
            "Warns which values do what (0/3/25)",
        ],
    ),
    # ------------------------------------------------------------ Performance
    BenchmarkQuestion(
        id="kernel_task_cpu",
        prompt=(
            "Activity Monitor shows kernel_task eating huge CPU and my whole "
            "Mac beachballs. I've already rebooted and it comes back."
        ),
        category="performance", tier=3, depth_triage=True,
        rubric=[
            "Explains kernel_task often reflects thermal management, not a runaway process",
            "Does NOT just say reboot (already tried)",
            "Suggests checking thermal state / powermetrics, SMC reset (Intel) or checking a specific driver/kext",
            "Mentions identifying a third-party kext or peripheral as the trigger",
        ],
    ),
    BenchmarkQuestion(
        id="memory_pressure",
        prompt=(
            "My Mac gets really slow and Activity Monitor shows memory pressure "
            "in the red with tons of swap used."
        ),
        category="performance", tier=2, depth_triage=False,
        rubric=[
            "Explains memory pressure vs. free RAM and swap",
            "Suggests identifying top memory consumers (Activity Monitor / `top`/`vm_stat`)",
            "Actionable steps: quit offenders, reduce browser tabs, check for a leaking process",
        ],
    ),
    # ------------------------------------------------------------ Permissions
    BenchmarkQuestion(
        id="mic_permission_missing",
        prompt=(
            "An app needs my microphone but it never appears in System Settings "
            "> Privacy & Security > Microphone, so I can't grant it."
        ),
        category="permissions", tier=2, depth_triage=False,
        rubric=[
            "Explains apps only appear after they first request the permission",
            "Suggests resetting TCC for microphone with `tccutil reset Microphone <bundleid>`",
            "Correct tccutil syntax and mention of bundle id",
        ],
    ),
    BenchmarkQuestion(
        id="tccutil_accessibility",
        prompt=(
            "I need to fully reset the Accessibility permission for one specific "
            "app because macOS thinks it's already granted but it isn't working."
        ),
        category="permissions", tier=2, depth_triage=True,
        rubric=[
            "Gives `tccutil reset Accessibility <bundle-id>` with correct syntax",
            "Explains how to find the bundle id (osascript id of / Info.plist)",
            "Notes the app must be re-added / re-prompted afterwards",
        ],
    ),
    BenchmarkQuestion(
        id="gatekeeper_blocked",
        prompt=(
            "macOS refuses to open an app I downloaded, saying it can't verify "
            "the developer / the signature is invalid. It's a tool I trust."
        ),
        category="permissions", tier=3, depth_triage=False,
        rubric=[
            "Explains Gatekeeper / quarantine and the right-click > Open path",
            "Mentions `xattr -d com.apple.quarantine <path>` or `spctl` where appropriate",
            "Mentions Privacy & Security > 'Open Anyway'",
            "Does not recommend globally disabling Gatekeeper as the first resort",
        ],
    ),
    # --------------------------------------------------------------- System
    BenchmarkQuestion(
        id="launchd_boot",
        prompt=(
            "I have a launch agent that's supposed to start a service at login "
            "but it never runs. How do I debug this on modern macOS?"
        ),
        category="system", tier=2, depth_triage=True,
        rubric=[
            "Uses `launchctl` (bootstrap/enable/print) rather than deprecated load/-w only",
            "Suggests `launchctl print` or `launchctl list` to inspect state and last exit code",
            "Mentions checking the plist location and log output / StandardErrorPath",
            "Correct launchctl syntax for current macOS",
        ],
    ),
    BenchmarkQuestion(
        id="defaults_write",
        prompt=(
            "How do I change a hidden macOS preference with defaults write, and "
            "make the change actually take effect?"
        ),
        category="system", tier=2, depth_triage=False,
        rubric=[
            "Correct `defaults write <domain> <key> -<type> <value>` form",
            "Explains needing to restart the affected app / `killall` (e.g. killall Finder/Dock)",
            "Mentions `defaults read` to verify and how to delete a key",
        ],
    ),
    # ------------------------------------------------------------ Diagnostics
    BenchmarkQuestion(
        id="kernel_panic_log",
        prompt=(
            "My Mac restarted with a message saying it panicked. Where is the "
            "panic report and how do I read it to find the cause?"
        ),
        category="diagnostics", tier=2, depth_triage=False,
        rubric=[
            "Points to the panic report location (/Library/Logs/DiagnosticReports, *.panic)",
            "Explains reading the 'panicked task' / backtrace / responsible kext",
            "Mentions Console.app or `log show` as alternatives",
        ],
    ),
    BenchmarkQuestion(
        id="log_show_predicate",
        prompt=(
            "Give me the exact log show command with a predicate to find kernel "
            "panic or crash entries in the unified log for the last day."
        ),
        category="diagnostics", tier=3, depth_triage=True,
        rubric=[
            "Gives `log show --last 1d --predicate '...'` with correct quoting",
            "Predicate uses a real field (eventMessage CONTAINS / processImagePath / subsystem)",
            "Syntax is copy-pasteable and correct",
        ],
    ),
    # ------------------------------------------------------- Version-specific
    BenchmarkQuestion(
        id="sequoia_bt_m3",
        prompt=(
            "After updating to macOS Sequoia 15 my M3 MacBook Air has constant "
            "Bluetooth instability with my mouse and keyboard. Basic toggles "
            "didn't help."
        ),
        category="bluetooth", tier=2, depth_triage=True, macos_version="15",
        chip="apple_silicon",
        rubric=[
            "Acknowledges it's version-specific (Sequoia 15) if the KB has that",
            "Does NOT lead with toggle Bluetooth (already tried)",
            "Suggests removing Bluetooth plist / resetting module, checking interference (USB3/2.4GHz)",
            "Concrete steps appropriate to Apple Silicon (no SMC reset advice for M-series)",
        ],
    ),
    # --------------------------------------------------------------- Edge cases
    BenchmarkQuestion(
        id="vague_slow",
        prompt="My Mac is slow. Help.",
        category="performance", tier=1, depth_triage=False,
        rubric=[
            "Asks for or infers the minimum needed detail without a long interrogation",
            "Gives a small ranked set of first diagnostics (Activity Monitor CPU/memory, storage, login items)",
            "Does not dump an overwhelming generic checklist",
        ],
    ),
    BenchmarkQuestion(
        id="offdomain_windows",
        prompt="How do I install Python on Windows using pip?",
        category=None, tier=None, depth_triage=False,
        rubric=[
            "Recognizes this is outside the macOS troubleshooting domain",
            "Either declines/redirects or answers briefly without fabricating macOS specifics",
            "Does NOT hallucinate macOS KB content for an off-domain question",
        ],
        tags=["off_domain"],
    ),
]


# ---------------------------------------------------------------------------
# HARD SLICE — "frontier-fails" questions.
#
# These target where a raw frontier model is genuinely weak and our KB/man-pages
# + Tavily deep-search are strong: EXACT command syntax (flags/keys GPT-4o tends
# to approximate or invent), and RECENT version-specific behavior (post-cutoff).
# The rubrics reward precise, correct specifics — the axis where grounding wins.
# ---------------------------------------------------------------------------

HARD_QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        id="h_log_wifi_predicate",
        prompt=(
            "Give me the exact `log show` command (with the correct --predicate "
            "syntax and quoting) to pull Wi-Fi association/disconnect events with "
            "their reason codes from the last hour. I keep getting predicate "
            "syntax errors."
        ),
        category="wifi", tier=3, depth_triage=True,
        rubric=[
            "Correct `log show --last 1h --predicate '...'` with valid quoting",
            "Predicate uses a REAL field: subsystem == \"com.apple.wifi\" or "
            "process == \"airportd\"/\"wifid\" (not an invented field)",
            "Flags actually exist (--last, --predicate, --info/--debug, --style)",
            "Mentions `sudo wdutil log` as the modern alternative",
            "No hallucinated predicate keys",
        ],
    ),
    BenchmarkQuestion(
        id="h_defaults_hidden_key",
        prompt=(
            "What's the exact `defaults` command to make the Dock autohide with no "
            "show/hide animation delay, and how do I apply it? Give the precise "
            "domain, keys, and value types."
        ),
        category="system", tier=2, depth_triage=False,
        rubric=[
            "Correct domain `com.apple.dock` and keys `autohide-delay` + `autohide-time-modifier`",
            "Correct `defaults write com.apple.dock <key> -float 0` syntax",
            "Says to `killall Dock` to apply",
            "Correct value types (-float/-int/-bool); no invented keys",
        ],
    ),
    BenchmarkQuestion(
        id="h_pmset_standby",
        prompt=(
            "On an Apple Silicon MacBook, which exact `pmset` keys control the "
            "low/high standby delays and how do I read the current values? I want "
            "the real key names, not a general explanation."
        ),
        category="battery", tier=3, depth_triage=True, chip="apple_silicon",
        rubric=[
            "`pmset -g` (or `pmset -g custom`) to read values",
            "Correct keys: standbydelaylow / standbydelayhigh / highstandbythreshold",
            "`sudo pmset -a <key> <seconds>` to set",
            "Notes Apple Silicon nuances vs Intel; no invented keys",
        ],
    ),
    BenchmarkQuestion(
        id="h_tcc_db_inspect",
        prompt=(
            "I want to directly inspect which apps hold Full Disk Access on macOS "
            "by querying the TCC database, not the GUI. What's the exact path and "
            "a sqlite query, and what are the gotchas?"
        ),
        category="permissions", tier=3, depth_triage=True,
        rubric=[
            "Correct DB path ~/Library/Application Support/com.apple.TCC/TCC.db "
            "(and the system one under /Library/…)",
            "Notes the service string is kTCCServiceSystemPolicyAllFiles for Full Disk Access",
            "Correct sqlite3 query against the `access` table",
            "Gotcha: SIP blocks reading the system TCC.db / needs Full Disk Access for Terminal",
        ],
    ),
    BenchmarkQuestion(
        id="h_spctl_gatekeeper",
        prompt=(
            "An app is blocked by Gatekeeper. Give me the exact commands to (a) see "
            "the assessment verdict, (b) remove the quarantine attribute, and (c) "
            "check the code signature — with correct flags."
        ),
        category="permissions", tier=3, depth_triage=True,
        rubric=[
            "`spctl -a -vvv <path>` (or `spctl --assess`) for the verdict",
            "`xattr -d com.apple.quarantine <path>` (or `xattr -c`)",
            "`codesign -dv --verbose=4 <path>` to inspect the signature",
            "Flags are correct and exist; notes `spctl --master-disable` is not the first resort",
        ],
    ),
    BenchmarkQuestion(
        id="h_launchd_bootstrap",
        prompt=(
            "On current macOS, a LaunchDaemon won't start. Give me the exact modern "
            "`launchctl` subcommands to bootstrap it, enable it, kickstart it, and "
            "read its last exit status — not the deprecated load/-w workflow."
        ),
        category="system", tier=3, depth_triage=True,
        rubric=[
            "Modern `launchctl bootstrap system /Library/LaunchDaemons/<x>.plist`",
            "`launchctl enable system/<label>` and `launchctl kickstart -k system/<label>`",
            "`launchctl print system/<label>` to read state / last exit code",
            "Explicitly avoids deprecated `launchctl load -w`; correct domain target syntax",
        ],
    ),
    BenchmarkQuestion(
        id="h_dscl_hidden_user",
        prompt=(
            "What's the exact `dscl` command sequence to check and change a user's "
            "default login shell from the command line? Give real keys."
        ),
        category="system", tier=3, depth_triage=False,
        rubric=[
            "`dscl . -read /Users/<name> UserShell` to check",
            "`sudo dscl . -change /Users/<name> UserShell <old> <new>` (or -create) to set",
            "Correct `.` datasource and `/Users/<name>` path; real attribute `UserShell`",
            "No invented dscl verbs/attributes",
        ],
    ),
    BenchmarkQuestion(
        id="h_diskutil_apfs_snapshot",
        prompt=(
            "How do I list local APFS snapshots and delete a specific one from the "
            "command line? Give the exact `diskutil`/`tmutil` subcommands."
        ),
        category="disk", tier=3, depth_triage=True,
        rubric=[
            "`diskutil apfs listSnapshots <mountpoint/disk>` (or `tmutil listlocalsnapshots /`)",
            "`tmutil deletelocalsnapshots <date>` or `diskutil apfs deleteSnapshot`",
            "Correct subcommand names and arguments",
            "No invented diskutil verbs",
        ],
    ),
    BenchmarkQuestion(
        id="h_sequoia_recent_bug",
        prompt=(
            "Since updating to macOS Sequoia 15.5 my Mac shows a firewall/local "
            "network prompt loop and some apps can't reach the local network even "
            "after allowing them. What's actually going on and how do I fix it on "
            "this specific version?"
        ),
        category="wifi", tier=3, depth_triage=True, macos_version="15",
        rubric=[
            "Acknowledges this is version-specific (Sequoia 15.x local-network-privacy behavior)",
            "Gives a concrete reset: Settings > Privacy & Security > Local Network toggle, "
            "or `tccutil reset` for the relevant service, and re-approve",
            "Does not give stale/generic firewall advice as if it were older macOS",
            "Ideally cites a current source (web) rather than guessing",
        ],
    ),
    BenchmarkQuestion(
        id="h_kernel_panic_backtrace",
        prompt=(
            "I have a kernel panic report mentioning a specific third-party kext in "
            "the backtrace. Walk me through exactly how to read the panic log to "
            "identify the responsible kext and confirm it, with the exact commands "
            "and file locations."
        ),
        category="diagnostics", tier=3, depth_triage=True,
        rubric=[
            "Panic reports live in /Library/Logs/DiagnosticReports/*.panic",
            "Explains reading the backtrace / 'Kernel Extensions in backtrace' section",
            "`kmutil showloaded` (modern) or `kextstat` to confirm the loaded kext",
            "Correct locations + commands; no invented tools",
        ],
    ),
]

# Depth-triage-heavy quick subset for fast iteration.
QUICK_IDS = [
    "wifi_drops_tried", "bt_reset_module", "battery_drain_sonoma",
    "kernel_task_cpu", "tccutil_accessibility", "log_show_predicate",
    "ext_drive_no_mount", "offdomain_windows",
]

HARD_QUICK_IDS = [
    "h_log_wifi_predicate", "h_pmset_standby", "h_tcc_db_inspect",
    "h_spctl_gatekeeper", "h_launchd_bootstrap", "h_sequoia_recent_bug",
]


def get_questions(quick: bool = False, hard: bool = False) -> list[BenchmarkQuestion]:
    if hard:
        pool = HARD_QUESTIONS
        return [q for q in pool if q.id in HARD_QUICK_IDS] if quick else list(pool)
    if quick:
        return [q for q in QUESTIONS if q.id in QUICK_IDS]
    return list(QUESTIONS)
