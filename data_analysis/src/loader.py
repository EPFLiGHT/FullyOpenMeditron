"""
Unified loader for the source<->synthetic analysis pipeline.

One top-level function:  load(path, pair, split) -> Iterator[Record]

Loaders handle multiple real-world schema variants per pair:
  - MOOVE source: Firestore-nested dict, auto-detects JSON vs JSONL by content
  - Guidelines source: 6 schema variants across aafp/cps/drugs/idsa/mayo/rch
    - magic.jsonl is skipped by default (all records are login-page scrapes)
"""

import json
import random
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

from schema import Record


# ---------------------------------------------------------------------------
# File-reading helpers
# ---------------------------------------------------------------------------

def _sniff_json_or_jsonl(path):
    """
    Return 'json' if the file's first non-whitespace char is '{' followed
    by a quoted key on the same visual structure as a JSONL line would not
    have (i.e., a single big JSON object with many keys), else 'jsonl'.
    We detect by trying to parse the whole file as JSON first; if that works
    AND the result is a dict with more than one top-level key, it's a big JSON.
    Otherwise treat as JSONL.
    """
    # Cheap heuristic: if the file fits in memory and parses as a dict with >1 key, it's JSON.
    # Most MOOVE sources are under a few hundred MB; safe.
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(2048).lstrip()
    except Exception:
        return "jsonl"

    # Multiple lines of JSON objects is the giveaway for JSONL
    if head.count("\n") > 1:
        # Could still be a single pretty-printed JSON object. Check first line.
        first_line = head.split("\n", 1)[0].strip()
        # A JSONL line is self-contained JSON. A pretty-printed JSON starts with just "{"
        if first_line == "{" or first_line.endswith(","):
            return "json"
        # Try parsing the first line alone
        try:
            json.loads(first_line)
            return "jsonl"
        except Exception:
            return "json"

    # Single line or very short head: try JSON
    return "json"


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("{}: line {} not valid JSON: {}".format(path, i, e))


def _read_big_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _convo_extract(rec):
    """Extract (human, assistant) from OpenAI-style conversations wrapper.
    Accepts 'from' values of 'human'/'user' and 'assistant'/'gpt'."""
    convo = rec.get("conversations") or []
    human = None
    assistant = None
    for turn in convo:
        who = str(turn.get("from") or "").lower()
        val = turn.get("value")
        if who in ("human", "user") and human is None:
            human = val
        elif who in ("assistant", "gpt", "ai") and assistant is None:
            assistant = val
    return human, assistant


def _has_think_block(text):
    if not text:
        return False
    return "<think>" in text and "</think>" in text


def _looks_like_mcq(question):
    if not question:
        return False
    markers = sum(1 for m in ("A)", "B)", "C)", "D)", "E)") if m in question)
    return markers >= 2


# ---------------------------------------------------------------------------
# MOOVE source adapter
# ---------------------------------------------------------------------------

def _moove_pick_answer(rec, rid):
    a1 = rec.get("firstAnswer") or ""
    a2 = rec.get("secondAnswer") or ""
    vote = str(rec.get("vote") or "").strip()

    if vote == "1":
        return (a1 or None), vote
    if vote == "2":
        return (a2 or None), vote
    if vote == "12":
        if a1 and a2:
            return (a1 + "\n\n---\n\n" + a2), vote
        return (a1 or a2 or None), vote

    rng = random.Random(rid)
    choice = rng.choice([a1, a2])
    return (choice or None), (vote or "unknown")


def load_moove_source(path, source_file=None):
    """Handles both JSON (one big dict) and JSONL MOOVE source files.
    Auto-detects by content, not extension."""
    source_file = source_file or path.name
    fmt = _sniff_json_or_jsonl(path)

    if fmt == "jsonl":
        def _stream():
            for obj in _read_jsonl(path):
                if isinstance(obj, dict) and len(obj) == 1 and isinstance(
                    next(iter(obj.values())), dict
                ):
                    rid, rec = next(iter(obj.items()))
                    yield rid, rec
                else:
                    rid = obj.get("contribution_id") or obj.get("id") or ""
                    yield rid, obj
        stream = _stream()
    else:
        big = _read_big_json(path)
        if not isinstance(big, dict):
            raise ValueError(
                "MOOVE source expected dict at top level, got {}".format(type(big).__name__)
            )
        stream = iter(big.items())

    for rid, rec in stream:
        rid = rid or rec.get("contribution_id") or ""
        if not rid:
            continue
        question = (rec.get("question") or "").strip() or None
        answer, vote_norm = _moove_pick_answer(rec, rid)
        parts = [p for p in (
            question,
            rec.get("firstAnswer") or "",
            rec.get("secondAnswer") or "",
        ) if p]
        full_text = "\n\n".join(parts).strip()

        yield Record(
            id="moove_src_" + str(rid),
            pair="moove",
            split="source",
            source_file=source_file,
            question=question,
            answer=answer,
            full_text=full_text,
            is_mcq=_looks_like_mcq(question),
            is_paired=True,
            is_document=False,
            has_think_block=False,
            parse_failed=False,
            raw=rec,
            vote=vote_norm,
            country=rec.get("country"),
            organization=rec.get("organization_name"),
            working_group=rec.get("workingGroup_name"),
        )


# ---------------------------------------------------------------------------
# Conversations-format adapters
# ---------------------------------------------------------------------------

def _load_conversations_file(path, pair, split, id_prefix,
                             has_think_hint=False, expects_mcq_label=False,
                             respect_parse_failed=False):
    source_file = path.name
    for i, rec in enumerate(_read_jsonl(path)):
        parse_failed = False
        if respect_parse_failed:
            parse_failed = bool(rec.get("parse_failed", False))

        question, answer = _convo_extract(rec)

        # Fallbacks: some sources may store Q/A at top level instead of in conversations
        if question is None:
            for k in ("question", "instruction", "prompt", "input"):
                v = rec.get(k)
                if v:
                    question = v
                    break
        if answer is None:
            for k in ("answer", "output", "response", "completion"):
                v = rec.get(k)
                if v:
                    answer = v
                    break

        rid = rec.get("id") or rec.get("question_id") or "{:07d}".format(i)
        full_parts = [p for p in (question, answer) if p]
        full_text = "\n\n".join(full_parts).strip()

        think = has_think_hint and _has_think_block(answer)
        mcq = _looks_like_mcq(question)
        mcq_label = rec.get("label_letter") if expects_mcq_label else None

        yield Record(
            id="{}_{}".format(id_prefix, rid),
            pair=pair,
            split=split,
            source_file=source_file,
            question=question,
            answer=answer,
            full_text=full_text,
            is_mcq=mcq,
            is_paired=False,
            is_document=False,
            has_think_block=think,
            parse_failed=parse_failed,
            raw=rec,
            mcq_label=mcq_label,
        )


def load_moove_synthetic(path):
    return _load_conversations_file(path, "moove", "synthetic", "moove_syn",
                                     False, False, False)


def load_meditron_source(path):
    return _load_conversations_file(path, "meditron", "source", "meditron_src",
                                     False, False, False)


def load_meditron_synthetic(path):
    return _load_conversations_file(path, "meditron", "synthetic", "meditron_syn",
                                     True, True, False)


def load_guidelines_synthetic(path):
    return _load_conversations_file(path, "guidelines", "synthetic", "guidelines_syn",
                                     False, False, True)


# ---------------------------------------------------------------------------
# Guidelines source: six file-specific schemas
# ---------------------------------------------------------------------------

# Files that produce no useful content (every record is a login page or similar)
SKIP_GUIDELINE_FILES = {"magic"}


def _flatten_content(value):
    """Coerce rec['content'] into a single markdown string.

    Mayo-style: dict of {section_name: markdown_string} — join section values.
    RCH-style:  plain string — return as-is.
    CPS-old:    plain string — return as-is.
    Other:      empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for section_name, section_body in value.items():
            if isinstance(section_body, str) and section_body.strip():
                parts.append("## " + str(section_name) + "\n\n" + section_body)
            elif isinstance(section_body, dict):
                parts.append(_flatten_content(section_body))
        return "\n\n".join(parts)
    if isinstance(value, list):
        return "\n\n".join(_flatten_content(x) for x in value if x)
    return ""


def _guideline_extract_body(rec):
    """Return (title, body_text, url, schema_variant) for any guideline record."""
    # Title can be under 'title' or 'name'
    title = (rec.get("title") or rec.get("name") or "").strip()
    # URL can be under 'url' or 'link'
    url = rec.get("url") or rec.get("link")

    # Body: try in priority order
    # 1. clean_text (CPS-new, richest)
    # 2. text (drugs, aafp, idsa)
    # 3. content (mayo dict / rch string / cps-old string)
    # 4. raw_text (CPS fallback — includes nav chrome)
    ct = rec.get("clean_text")
    if isinstance(ct, str) and ct.strip():
        return title, ct.strip(), url, "clean_text"

    t = rec.get("text")
    if isinstance(t, str) and t.strip():
        return title, t.strip(), url, "text"

    c = rec.get("content")
    flat = _flatten_content(c).strip()
    if flat:
        # Distinguish mayo-style (dict) from rch/cps-old (string) for diagnostics
        variant = "content_dict" if isinstance(c, dict) else "content_str"
        return title, flat, url, variant

    rt = rec.get("raw_text")
    if isinstance(rt, str) and rt.strip():
        return title, rt.strip(), url, "raw_text"

    return title, "", url, "empty"


def load_guidelines_source(path, include_magic=False):
    """Load a single guideline file. Skips files in SKIP_GUIDELINE_FILES
    unless include_magic=True."""
    source_file = path.name
    src_tag = path.stem.lower()

    if src_tag in SKIP_GUIDELINE_FILES and not include_magic:
        return  # generator returns nothing

    for i, rec in enumerate(_read_jsonl(path)):
        title, body, url, variant = _guideline_extract_body(rec)
        rid = rec.get("id") or "{}_{:07d}".format(src_tag, i)

        # Assemble full_text
        if title and body:
            full_text = title + "\n\n" + body
        else:
            full_text = title or body or ""
        full_text = full_text.strip()

        if not full_text:
            # Still yield, but flag it so sanity can count empties
            # Comment out the next line if you'd rather drop them silently:
            # continue
            pass

        yield Record(
            id="guidelines_src_" + str(rid),
            pair="guidelines",
            split="source",
            source_file=source_file,
            question=None,
            answer=None,
            full_text=full_text,
            is_mcq=False,
            is_paired=False,
            is_document=True,
            has_think_block=False,
            parse_failed=(variant == "empty"),
            raw=rec,
            guideline_source=src_tag,
            url=url,
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

LOADER_REGISTRY = {
    ("moove", "source"):        load_moove_source,
    ("moove", "synthetic"):     load_moove_synthetic,
    ("meditron", "source"):     load_meditron_source,
    ("meditron", "synthetic"):  load_meditron_synthetic,
    ("guidelines", "source"):   load_guidelines_source,
    ("guidelines", "synthetic"):load_guidelines_synthetic,
}


def load(path, pair, split):
    path = Path(path)
    key = (pair, split)
    if key not in LOADER_REGISTRY:
        raise KeyError("No loader registered for " + str(key))
    return LOADER_REGISTRY[key](path)


def load_guidelines_source_dir(dir_path, include_magic=False):
    dir_path = Path(dir_path)
    for f in sorted(dir_path.glob("*.jsonl")):
        for rec in load_guidelines_source(f, include_magic=include_magic):
            yield rec
