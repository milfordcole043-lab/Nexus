"""Regex-based entity extraction from document content."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    """An entity extracted from text content."""

    name: str
    type: str
    confidence: float
    context_snippet: str


# Compiled regex patterns for each entity type
_LANGUAGE_NAMES = (
    r"Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Scala|"
    r"Haskell|Elixir|Clojure|Lua|Perl|R|Julia|Dart|Zig|Nim|OCaml|Erlang|MATLAB|SQL|"
    r"HTML|CSS|Bash|Shell|PowerShell|Assembly"
)

_LIBRARY_NAMES = (
    r"React|FastAPI|Django|Flask|Express|Next\.js|Svelte|Vue|Angular|Spring|Rails|"
    r"Laravel|Gin|Echo|Actix|Axum|Rocket|numpy|pandas|scipy|matplotlib|seaborn|"
    r"scikit-learn|sklearn|TensorFlow|PyTorch|Keras|transformers|LangChain|"
    r"SQLAlchemy|Pydantic|uvicorn|aiohttp|httpx|requests|pytest|unittest|"
    r"Tailwind|Bootstrap|Material-UI|Prisma|Drizzle|tRPC|GraphQL|gRPC|"
    r"Redis|PostgreSQL|MongoDB|MySQL|SQLite|Elasticsearch|Celery|RabbitMQ|Kafka|"
    r"watchdog|aiosqlite|LanceDB|ChromaDB|Pinecone|Weaviate|FAISS"
)

_TOOL_NAMES = (
    r"Docker|Kubernetes|Git|GitHub|GitLab|Bitbucket|Terraform|Ansible|Jenkins|"
    r"CircleCI|Travis|Vercel|Netlify|AWS|Azure|GCP|Heroku|Nginx|Apache|"
    r"Webpack|Vite|ESBuild|Rollup|Babel|npm|yarn|pnpm|pip|poetry|conda|"
    r"Ollama|Groq|OpenAI|Anthropic|Claude|VS\s?Code|Neovim|Vim|Emacs|"
    r"Prometheus|Grafana|Datadog|Sentry|Linux|macOS|Windows"
)

_FILE_REF_PATTERN = re.compile(
    r"(?:^|[\s\"'`(])("
    r"(?:[\w./-]+/)?"  # optional directory prefix
    r"[\w.-]+"  # filename stem
    r"\.(?:py|js|ts|tsx|jsx|rs|go|java|rb|php|c|cpp|h|hpp|css|html|json|yaml|yml|toml|md|txt|sh|sql|env|cfg|ini|xml|csv)"
    r")"
    r"(?:[\s\"'`),:;.]|$)",
    re.MULTILINE,
)

_PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b")

# Common false positives for person detection
_PERSON_STOPLIST = frozenset({
    "The End", "New York", "San Francisco", "Los Angeles", "United States",
    "Open Source", "Pull Request", "Merge Request", "Code Review",
    "Stack Overflow", "Visual Studio", "Machine Learning", "Deep Learning",
    "Natural Language", "Source Code", "Data Science", "Web Assembly",
    "Hello World", "Read Only", "Read Write", "Black Box", "White Box",
    "Best Practice", "Design Pattern", "System Design", "High Performance",
    "Real Time", "Type Script", "Key Value", "End Point",
})

PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("language", re.compile(rf"\b({_LANGUAGE_NAMES})\b"), 0.95),
    ("library", re.compile(rf"\b({_LIBRARY_NAMES})\b"), 0.9),
    ("tool", re.compile(rf"\b({_TOOL_NAMES})\b"), 0.85),
    ("file_ref", _FILE_REF_PATTERN, 0.8),
]


def _get_context_snippet(content: str, match_start: int, match_end: int, window: int = 80) -> str:
    """Extract surrounding text as context snippet."""
    start = max(0, match_start - window)
    end = min(len(content), match_end + window)
    snippet = content[start:end].strip()
    # Clean up to sentence-ish boundaries
    if start > 0:
        first_space = snippet.find(" ")
        if first_space > 0 and first_space < 20:
            snippet = snippet[first_space + 1:]
    if end < len(content):
        last_space = snippet.rfind(" ")
        if last_space > len(snippet) - 20:
            snippet = snippet[:last_space]
    return snippet


class EntityExtractor:
    """Extracts entities from text using compiled regex patterns."""

    def extract(self, content: str) -> list[ExtractedEntity]:
        """Extract entities from text content. Pure function, no async."""
        seen: dict[tuple[str, str], ExtractedEntity] = {}

        # Named patterns (language, library, tool, file_ref)
        for entity_type, pattern, confidence in PATTERNS:
            for match in pattern.finditer(content):
                name = match.group(1).strip()
                if not name or len(name) < 2:
                    continue
                key = (name, entity_type)
                if key not in seen:
                    snippet = _get_context_snippet(content, match.start(), match.end())
                    seen[key] = ExtractedEntity(
                        name=name,
                        type=entity_type,
                        confidence=confidence,
                        context_snippet=snippet,
                    )

        # Person names
        for match in _PERSON_PATTERN.finditer(content):
            name = match.group(1)
            if name in _PERSON_STOPLIST:
                continue
            # Skip if it overlaps with a known entity
            key = (name, "person")
            if key not in seen:
                snippet = _get_context_snippet(content, match.start(), match.end())
                seen[key] = ExtractedEntity(
                    name=name,
                    type="person",
                    confidence=0.6,
                    context_snippet=snippet,
                )

        return list(seen.values())
