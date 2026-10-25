from functools import lru_cache
from pathlib import Path

import pytest

from eventwise.extract import extract_document

DATA = Path(__file__).resolve().parent.parent / "data"
SINGLE = DATA / "1 - single document"
MULTI = DATA / "2 - multiple documents"
FESTIVAL = DATA / "3 - example festival documents" / "Festival_Documents"
CREW = DATA / "3 - example festival documents" / "Crew_Onboarding" / "02_crew_agreements"
AI_INDEX = SINGLE / "ai_index_report_2026.pdf"


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"dataset file not available: {path}")
    return path


@lru_cache(maxsize=None)
def extracted(path: Path, pages: tuple[int, ...] | None = None):
    """Extraction is deterministic, so cache it across tests."""
    return extract_document(require(path), pages=list(pages) if pages else None)
