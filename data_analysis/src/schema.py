"""
Canonical record shape for the source<->synthetic analysis pipeline.

Every file loader in loader.py yields Record instances, so every downstream
stage (metadata extraction, text stats, embedding, contamination) works on
one shape regardless of which of the six input files the data came from.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


Pair = str   # "moove" | "meditron" | "guidelines"
Split = str  # "source" | "synthetic"


@dataclass
class Record:
    # --- identity ---
    id: str
    pair: Pair
    split: Split
    source_file: str

    # --- content ---
    question: Optional[str]
    answer: Optional[str]
    full_text: str

    # --- task-format markers ---
    is_mcq: bool = False
    is_paired: bool = False
    is_document: bool = False
    has_think_block: bool = False

    # --- provenance / filtering flags ---
    parse_failed: bool = False

    # --- raw payload ---
    raw: dict = field(default_factory=dict)

    # --- optional pair-specific metadata ---
    vote: Optional[str] = None
    country: Optional[str] = None
    organization: Optional[str] = None
    working_group: Optional[str] = None
    mcq_label: Optional[str] = None
    guideline_source: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def to_jsonl(self, drop_raw=True):
        d = self.to_dict()
        if drop_raw:
            d.pop("raw", None)
        return json.dumps(d, ensure_ascii=False)


def needs_document_prompt(rec):
    return rec.is_document


def is_valid_for_qa_metadata(rec):
    return rec.question is not None and rec.answer is not None
