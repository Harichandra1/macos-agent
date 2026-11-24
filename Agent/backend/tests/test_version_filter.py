"""
test_version_filter.py — macOS version matching in metadata pre-filtering.

No API keys, no network: `version_match_values` is a pure function and
`_build_filter` never touches `self`, so it is called unbound.

THE BUG THIS PINS DOWN: the KB tags version-specific documents with bare major
releases — 1,238 documents carry "14", only 17 carry any 14.x point release —
while 77.7% of the corpus is tagged ["all"]. Matching the user's version string
verbatim therefore dropped every major-release document the moment the user was
precise. Measured against the live collection before the fix, a query filtered
on "14.1" returned 5 hits that were all ["all"] documents, and a filter of
["14.1"] alone returned zero. Users who stated an exact version were getting
the least specific results.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_version_filter.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WEB_SEARCH_MODE", "off")          # keep tests offline

from app.agent.retrieval import (  # noqa: E402
    MAJOR_VERSION_MATCH_BOOST,
    VERSION_MATCH_BOOST,
    Retriever,
    version_match_values,
)


def test_point_release_widens_to_major():
    assert version_match_values("14.1") == ["14.1", "14", "all"]
    assert version_match_values("15.7.3") == ["15.7.3", "15", "all"]
    assert version_match_values("26.2") == ["26.2", "26", "all"]
    print("  ✓ a point release also matches its major release")


def test_bare_major_does_not_duplicate_itself():
    # "14" is already the major; widening must not emit ["14", "14", "all"],
    # which would be a redundant term in every query's filter.
    assert version_match_values("14") == ["14", "all"]
    print("  ✓ a bare major release widens to itself plus 'all', no duplicate")


def test_legacy_ten_scheme_keeps_two_components():
    # 10.15 is Catalina — a release in its own right. A bare "10" spans
    # 2001-2020 and is meaningless, so it must never be generated.
    assert version_match_values("10.15.7") == ["10.15.7", "10.15", "all"]
    assert version_match_values("10.15") == ["10.15", "all"]
    for v in ("10.15.7", "10.15", "10.14.6"):
        assert "10" not in version_match_values(v), v
    print("  ✓ 10.x widens to two components and never to a bare '10'")


def test_degenerate_input_falls_back_to_all():
    for value in (None, "", "   ", "all", "ALL", "any"):
        assert version_match_values(value) == ["all"], repr(value)
    print("  ✓ missing or version-agnostic input yields just ['all']")


def test_all_is_always_last_and_present():
    # Dropping "all" would starve retrieval: it is 77.7% of the corpus.
    for value in ("14.1", "14", "10.15.7", "26.2", None):
        values = version_match_values(value)
        assert values[-1] == "all", value
        assert len(values) == len(set(values)), value
    print("  ✓ 'all' is always present and always last, with no duplicates")


def test_build_filter_uses_widened_values():
    # _build_filter does not use `self`, so an unbound call keeps this offline.
    built = Retriever._build_filter(None, "14.1", None, None, None)
    conditions = [c for c in built.must if c.key == "macos_versions"]
    assert len(conditions) == 1, built
    assert conditions[0].match.any == ["14.1", "14", "all"]
    print("  ✓ the Qdrant pre-filter carries the widened version list")


def test_build_filter_omits_version_when_agnostic():
    for value in (None, "all", "any", ""):
        built = Retriever._build_filter(None, value, None, None, None)
        keys = [] if built is None else [c.key for c in built.must]
        assert "macos_versions" not in keys, value
    print("  ✓ no version condition is emitted for version-agnostic queries")


def _boost(macos_version, doc_versions):
    """Mirror of the rerank branch, kept in one place so the ordering below
    is asserted rather than described."""
    candidates = version_match_values(macos_version)
    if candidates[0] in doc_versions:
        return VERSION_MATCH_BOOST
    if any(c in doc_versions for c in candidates[1:-1]):
        return MAJOR_VERSION_MATCH_BOOST
    return 0.0


def test_boost_ranks_exact_above_major_above_agnostic():
    exact = _boost("14.1", ["14.1"])
    major = _boost("14.1", ["14"])
    agnostic = _boost("14.1", ["all"])
    assert exact > major > agnostic == 0.0, (exact, major, agnostic)
    # The pre-fix behaviour: a "14" document scored the same as a "15" one.
    assert _boost("14.1", ["14"]) > _boost("14.1", ["15"])
    print("  ✓ exact match outranks major match outranks version-agnostic")


def test_boost_stays_a_tiebreaker():
    # A large boost would surface weak-but-version-matched docs over strong
    # general ones. Both boosts must stay far below a typical cosine gap.
    assert 0 < MAJOR_VERSION_MATCH_BOOST < VERSION_MATCH_BOOST <= 0.05
    print("  ✓ both boosts remain small enough to be tiebreakers")


if __name__ == "__main__":
    test_point_release_widens_to_major()
    test_bare_major_does_not_duplicate_itself()
    test_legacy_ten_scheme_keeps_two_components()
    test_degenerate_input_falls_back_to_all()
    test_all_is_always_last_and_present()
    test_build_filter_uses_widened_values()
    test_build_filter_omits_version_when_agnostic()
    test_boost_ranks_exact_above_major_above_agnostic()
    test_boost_stays_a_tiebreaker()
    print("\nAll version-filter tests passed.")
