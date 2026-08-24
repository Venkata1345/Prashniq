"""Evidence retrieval over a candidate's resume claims.

Deliberately not a vector store. A resume is a handful of claims, not a corpus,
so lexical overlap is enough, and being deterministic makes grounded questions
reproducible and testable. Phase 3 replaces the internals of `select_claims`
with embeddings + pgvector; the call sites do not change.
"""

from __future__ import annotations

import re

from app.context.schemas import ResumeClaim, ResumeProfile

DEFAULT_LIMIT = 3
SKILL_MATCH_WEIGHT = 2.0
TERM_MATCH_WEIGHT = 1.0

_TOKEN = re.compile(r"[a-z0-9+#.]+")

# Words that carry no retrieval signal in this domain.
STOPWORDS = frozenset(
    """a an and are as at be by for from how in into is it its of on or that the
    their this to use used using was were what when which why with would your
    you experience work working build built design designed system systems""".split()
)


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    }


def score_claim(query: str, claim: ResumeClaim) -> float:
    """Overlap score. Skill tokens count double: a claim tagged with the skill
    being asked about is stronger evidence than one that merely mentions it."""
    terms = tokenize(query)
    if not terms:
        return 0.0

    skill_terms = tokenize(" ".join(claim.skills))
    text_terms = tokenize(claim.text)

    score = SKILL_MATCH_WEIGHT * len(terms & skill_terms)
    score += TERM_MATCH_WEIGHT * len(terms & (text_terms - skill_terms))
    return score


def select_claims(
    query: str, claims: list[ResumeClaim], *, limit: int = DEFAULT_LIMIT
) -> list[ResumeClaim]:
    """Highest-scoring claims, ties broken by resume order. Claims with no
    overlap are never returned -- irrelevant "evidence" is worse than none."""
    scored = [(score_claim(query, claim), index, claim) for index, claim in enumerate(claims)]
    relevant = [entry for entry in scored if entry[0] > 0]
    relevant.sort(key=lambda entry: (-entry[0], entry[1]))
    return [claim for _, _, claim in relevant[:limit]]


def select_evidence(
    query: str, profile: ResumeProfile | None, *, limit: int = DEFAULT_LIMIT
) -> list[str]:
    if profile is None:
        return []
    return [claim.text for claim in select_claims(query, profile.claims, limit=limit)]
