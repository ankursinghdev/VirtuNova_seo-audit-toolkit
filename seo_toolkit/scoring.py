"""Declarative page-scoring engine — replaces repeated if/subtract/append blocks."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ScoreRule:
    """A single scoring rule: a condition check, penalty, and message.

    Parameters
    ----------
    check : callable(analysis, fetch_info) -> bool
        Returns True when the penalty should be applied.
    penalty : int
        Points to subtract from a perfect 100.
    reason : str
        Human-readable explanation appended to the reasons list.
    """
    check: Callable
    penalty: int
    reason: str


# Default SEO scoring rules
DEFAULT_RULES = [
    ScoreRule(
        check=lambda a, f: a.get("title", {}).get("length", 0) == 0,
        penalty=20,
        reason="Missing title",
    ),
    ScoreRule(
        check=lambda a, f: a.get("meta_description", {}).get("length", 0) == 0,
        penalty=10,
        reason="Missing meta description",
    ),
    ScoreRule(
        check=lambda a, f: a.get("h1", {}).get("count", 0) == 0,
        penalty=10,
        reason="Missing H1",
    ),
    ScoreRule(
        check=lambda a, f: a.get("word_count", 0) < 100,
        penalty=5,
        reason="Low word count (<100)",
    ),
    ScoreRule(
        check=lambda a, f: (
            f.get("status") is None or
            (f.get("status") and f.get("status") >= 400)
        ),
        penalty=100,
        reason="HTTP error: {status}",
    ),
]


def compute_page_score(analysis, fetch_info, rules=None):
    """Evaluate all scoring rules and return a score dict.

    Parameters
    ----------
    analysis : dict
        Page analysis data (title, meta_description, h1, word_count, ...).
    fetch_info : dict
        Fetch metadata (status code, headers, ...).
    rules : list[ScoreRule] | None
        Custom rules to apply. Defaults to DEFAULT_RULES.

    Returns
    -------
    dict
        {"score": int, "reasons": list[str]}
    """
    if rules is None:
        rules = DEFAULT_RULES

    score = 100
    reasons = []
    for rule in rules:
        if rule.check(analysis, fetch_info):
            score -= rule.penalty
            reason_text = rule.reason.format(status=fetch_info.get("status"))
            reasons.append(reason_text)
    return {"score": max(0, score), "reasons": reasons}
