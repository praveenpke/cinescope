"""Natural-language query parsing behind a single :class:`QueryParser` protocol.

Two implementations:

* :class:`ClaudeParser` — the real integration: Claude Haiku via the official
  ``anthropic`` SDK with structured outputs (``messages.parse`` validates the
  response straight into :class:`~api.schemas.QuerySpec`). Used whenever
  ``ANTHROPIC_API_KEY`` is set.
* :class:`HeuristicParser` — deterministic offline fallback: regexes for
  years/decades, a genre keyword map (with negation handling), and
  ``like <title>`` extraction. Same interface, clearly labeled
  ``heuristic_fallback`` in every response so the UI can tell.

:func:`parse_query` picks the parser per request (so dropping a key into
``.env`` and restarting — or an API blip mid-session — behaves sanely) and
degrades Claude errors to the heuristic instead of failing the request.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Protocol

from api.schemas import QuerySpec, YearRange
from pipeline import config

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

PARSER_CLAUDE = "claude"
PARSER_HEURISTIC = "heuristic_fallback"

# MovieLens canonical genres — both parsers emit these exact names so the SQL
# genre filters (case-insensitive equality on unnest(genres)) always hit.
MOVIELENS_GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

CLAUDE_SYSTEM_PROMPT = f"""You convert one natural-language movie discovery query into a \
structured search spec.

Rules:
- reference_titles: movie titles the user wants results similar to ("like Inception" -> \
["Inception"]). Title only, no year.
- mood_adjustments: tone shifts relative to the references ("funnier", "darker", "more \
romantic"). Keep the user's own words.
- genres_include / genres_exclude: only names from this list: {", ".join(MOVIELENS_GENRES)}. \
Map synonyms ("funny" -> Comedy, "scary" -> Horror, "space" -> Sci-Fi). Negated genres \
("nothing scary", "no musicals") go in genres_exclude.
- year_range: inclusive release-year window when the user constrains time ("90s" -> \
1990-1999, "after 2010" -> start 2010, no end). Omit when unconstrained.
- min_rating: 0-10 scale; only when the user asks for quality ("highly rated" -> 7.5). \
Omit otherwise.
- similarity_text: one or two sentences describing the plot, tone, and themes the user \
is after, written like a movie blurb — this string is embedded for semantic search, so \
make it descriptive, not a restatement of the filters."""


class QueryParser(Protocol):
    """Anything that turns a free-text query into a validated QuerySpec."""

    name: str

    def parse(self, query: str) -> QuerySpec: ...


class ClaudeParser:
    """Claude Haiku structured-output parser (official anthropic SDK)."""

    name = PARSER_CLAUDE

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = config.CLAUDE_PARSE_MODEL,
    ) -> None:
        if client is None:  # pragma: no cover - exercised in live mode only
            import anthropic as anthropic_sdk

            client = anthropic_sdk.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._client = client
        self.model = model

    def parse(self, query: str) -> QuerySpec:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=config.CLAUDE_PARSE_MAX_TOKENS,
            system=CLAUDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
            output_format=QuerySpec,
        )
        spec = response.parsed_output
        if spec is None:
            stop_reason = getattr(response, "stop_reason", "?")
            raise ValueError(f"Claude returned no parseable spec (stop_reason={stop_reason})")
        if not spec.similarity_text.strip():
            spec.similarity_text = query.strip()
        return spec


# --- Heuristic fallback -----------------------------------------------------

# genre keyword -> canonical MovieLens genre. Multi-word keys are matched as
# phrases; single words on word boundaries. Longest keys match first so
# "science fiction" wins over "action" never being inside it, etc.
GENRE_KEYWORDS: dict[str, str] = {
    "science fiction": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "space": "Sci-Fi",
    "funny": "Comedy",
    "funnier": "Comedy",
    "hilarious": "Comedy",
    "comedy": "Comedy",
    "comedies": "Comedy",
    "scary": "Horror",
    "scarier": "Horror",
    "horror": "Horror",
    "spooky": "Horror",
    "romantic": "Romance",
    "romance": "Romance",
    "rom-com": "Romance",
    "romcom": "Romance",
    "thriller": "Thriller",
    "thrillers": "Thriller",
    "suspense": "Thriller",
    "animated": "Animation",
    "animation": "Animation",
    "anime": "Animation",
    "cartoon": "Animation",
    "documentary": "Documentary",
    "documentaries": "Documentary",
    "drama": "Drama",
    "dramas": "Drama",
    "action": "Action",
    "adventure": "Adventure",
    "crime": "Crime",
    "heist": "Crime",
    "gangster": "Crime",
    "fantasy": "Fantasy",
    "western": "Western",
    "westerns": "Western",
    "war movie": "War",
    "war movies": "War",
    "war film": "War",
    "war films": "War",
    "musical": "Musical",
    "musicals": "Musical",
    "mystery": "Mystery",
    "whodunit": "Mystery",
    "noir": "Film-Noir",
    "kids": "Children",
    "family": "Children",
    "children": "Children",
}

# Words within this many tokens BEFORE a genre keyword that flip it to exclude.
_NEGATORS = {"no", "not", "without", "nothing", "non", "avoid", "skip", "minus"}
_NEGATION_WINDOW = 3

MOOD_WORDS: tuple[str, ...] = (
    "funnier",
    "darker",
    "lighter",
    "scarier",
    "sadder",
    "happier",
    "weirder",
    "slower",
    "faster",
    "grittier",
    "sweeter",
    "smarter",
    "more romantic",
    "more serious",
    "more violent",
    "less violent",
    "more uplifting",
    "feel-good",
)

_LIKE_RE = re.compile(
    r"\b(?:like|similar to|reminds me of|in the vein of)\s+(.+?)"
    r"(?=\s+but\b|\s+and\b|\s+with\b|\s+from\b|[,.;!?]|$)",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", re.IGNORECASE)
_AFTER_RE = re.compile(r"\b(?:after|since|post|newer than)\s+(\d{4})\b", re.IGNORECASE)
_BEFORE_RE = re.compile(r"\b(?:before|until|pre|older than)\s+(\d{4})\b", re.IGNORECASE)
_DECADE_RE = re.compile(r"\b(?:the\s+)?(\d{2}|\d{4})s\b", re.IGNORECASE)
_IN_YEAR_RE = re.compile(r"\b(?:in|from)\s+(\d{4})\b", re.IGNORECASE)
_RATED_RE = re.compile(
    r"\b(?:highly rated|well[- ]reviewed|critically acclaimed|top rated|great reviews)\b",
    re.IGNORECASE,
)
_HEURISTIC_MIN_RATING = 7.5


def _decade_to_range(token: str) -> tuple[int, int] | None:
    """'90' or '1990' (from '90s'/'1990s') -> (1990, 1999)."""
    if len(token) == 2:
        num = int(token)
        start = (1900 if num >= 30 else 2000) + num  # '20s' means 2020s in practice
    elif len(token) == 4 and token.endswith("0"):
        start = int(token)
    else:
        return None
    return start, start + 9


def _extract_year_range(query: str) -> YearRange | None:
    if m := _BETWEEN_RE.search(query):
        return YearRange(start=int(m.group(1)), end=int(m.group(2)))
    start: int | None = None
    end: int | None = None
    if m := _AFTER_RE.search(query):
        start = int(m.group(1))
    if m := _BEFORE_RE.search(query):
        end = int(m.group(1))
    if start is None and end is None:
        if (m := _DECADE_RE.search(query)) and (decade := _decade_to_range(m.group(1))):
            return YearRange(start=decade[0], end=decade[1])
        if m := _IN_YEAR_RE.search(query):
            year = int(m.group(1))
            return YearRange(start=year, end=year)
        return None
    return YearRange(start=start, end=end)


def _extract_genres(query: str) -> tuple[list[str], list[str]]:
    """(genres_include, genres_exclude) from the keyword map + negation window."""
    lowered = query.lower()
    tokens = re.findall(r"[a-z0-9'-]+", lowered)
    include: list[str] = []
    exclude: list[str] = []
    for keyword, genre in sorted(GENRE_KEYWORDS.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b")
        m = pattern.search(lowered)
        if not m:
            continue
        first_word = keyword.split()[0]
        negated = False
        for i, token in enumerate(tokens):
            if token == first_word:
                window = tokens[max(0, i - _NEGATION_WINDOW) : i]
                negated = any(w in _NEGATORS for w in window)
                break
        target = exclude if negated else include
        if genre not in target:
            target.append(genre)
    include = [g for g in include if g not in exclude]
    return include, exclude


class HeuristicParser:
    """Deterministic offline parser — same contract as ClaudeParser."""

    name = PARSER_HEURISTIC

    def parse(self, query: str) -> QuerySpec:
        query = query.strip()
        references = [m.group(1).strip() for m in _LIKE_RE.finditer(query)]
        include, exclude = _extract_genres(query)
        moods = [m for m in MOOD_WORDS if re.search(r"\b" + re.escape(m) + r"\b", query.lower())]
        return QuerySpec(
            reference_titles=references,
            mood_adjustments=moods,
            genres_include=include,
            genres_exclude=exclude,
            year_range=_extract_year_range(query),
            min_rating=_HEURISTIC_MIN_RATING if _RATED_RE.search(query) else None,
            similarity_text=query,
        )


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_parser() -> QueryParser:
    """ClaudeParser when a key exists, otherwise the offline heuristic."""
    if claude_available():
        return ClaudeParser()
    return HeuristicParser()


def parse_query(query: str) -> tuple[QuerySpec, str]:
    """Parse with the best available parser; degrade Claude errors gracefully.

    Returns ``(spec, parser_name)`` — the name lands in the API response so
    clients always know which parser produced the interpretation.
    """
    parser = get_parser()
    if parser.name == PARSER_CLAUDE:
        try:
            return parser.parse(query), parser.name
        except Exception:
            logger.exception(
                "Claude query parsing failed — serving the deterministic heuristic parse instead"
            )
            fallback = HeuristicParser()
            return fallback.parse(query), fallback.name
    logger.info(
        "OFFLINE MODE: ANTHROPIC_API_KEY not set — using the heuristic query parser. "
        "Add ANTHROPIC_API_KEY to .env (https://console.anthropic.com/settings/keys) and "
        "restart to enable Claude parsing (model: %s).",
        config.CLAUDE_PARSE_MODEL,
    )
    return parser.parse(query), parser.name
