#!/usr/bin/env python3
# auto_moove/moove_meta.py

"""Pair identity and case metadata for MOOVE rows.

The MOOVE exports this repo consumes have two shapes, and neither is stable
enough to read field names inline at every call site:

  all_reviews_no_dup_test*.jsonl -- one row per human review, with
      pair_key / left_ID / right_ID, prefix ("Malawi PHC"), care_level,
      income_tier, "Working group name", left_Model / right_Model,
      majority_winner (an answer ID).

  full_moove.jsonl -- the raw Firestore export, with contribution_id,
      country, organization_name, workingGroup_name, firstModel /
      secondModel, vote ("1" / "2" / "12").

extract_meta() maps both onto one small vocabulary so that everything
downstream -- results JSONL, the Inspect log, any grouping in analysis --
speaks the same field names. Deliberately pair-level only: reviewer-specific
columns (User name, per-reviewer scores) would be wrong to attach to a row
that was deduplicated down to one per pair.

pair_key_for() is the identity used to join those rows back to auto_moove
results, and must stay byte-identical to what auto_moove.py wrote.
"""

import hashlib

# canonical name -> source columns to try, in order
FIELD_ALIASES = {
    # prefixes / care_levels: list-valued, in the HF judge-calibration export
    "source_set":    ("prefix", "source", "dataset", "prefixes"),
    "country":       ("country", "Country"),
    "care_level":    ("care_level", "Care level", "level_of_care", "care_levels"),
    "income_tier":   ("income_tier", "Income tier"),
    "working_group": ("Working group name", "workingGroup_name", "working_group"),
    "organization":  ("organization_name", "Organization name"),
}

# the original MOOVE pair the question came from -- the two answers humans
# compared, which are NOT the two models auto_moove generates. Named apart
# from model_1 / model_2 so the two can never be read as the same thing.
_LEFT_MODEL = ("left_Model", "firstModel")
_RIGHT_MODEL = ("right_Model", "secondModel")
_LEFT_ID = ("left_ID", "Left answer ID")
_RIGHT_ID = ("right_ID", "Right answer ID")

# one model under two export names: Kenya PHC served gpt-oss through Bedrock,
# Tanzania and Malawi PHC by its HF id. Left apart, a leaderboard pooling
# events ranks them as two models.
MODEL_ALIASES = {"openai.gpt-oss-120b-1:0": "openai/gpt-oss-120b"}


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)

_VOTE_LABEL = {"1": "left", "left": "left", "first": "left",
               "2": "right", "right": "right", "second": "right",
               "12": "tie", "tie": "tie", "equal": "tie"}


def _first(row, keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, "", []):
            return v
    return None


def question_of(row):
    """The question text, whichever column this export calls it."""
    return (row.get("prompt")
            or row.get("question")
            or row.get("left_question_text")
            or row.get("right_question_text"))


def pair_key_for(row):
    """Stable pair identity. Mirrors auto_moove.py's original compute_pair_key:
    pair_key (list joined by "::") > id/ID/pair_id > sha256 of the question.
    source_id / weak_id / silver_id / gold_id are the HF judge-calibration
    export's ids; no MOOVE export carries them, so no key already written
    changes."""
    pk = row.get("pair_key")
    if pk:
        if isinstance(pk, (list, tuple)):
            return "::".join(str(x) for x in pk)
        return str(pk)
    for k in ("id", "ID", "pair_id", "source_id", "weak_id", "silver_id", "gold_id"):
        if row.get(k):
            return str(row[k])
    text = question_of(row) or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- one human review of a pair, for judge-vs-human validation ------------
# The MOOVE review export names the sides left_/right_answer_text and the
# vote "Vote" (left / right / equal). The HF judge-calibration export names
# them model_1_answer / model_2_answer and "label" (Model 1 / Model 2 / Equal).

_LEFT_ANSWER = ("left_answer_text", "model_1_answer")
_RIGHT_ANSWER = ("right_answer_text", "model_2_answer")
_REVIEW_VOTE = {"left": "1", "right": "2", "equal": "12",
                "model 1": "1", "model 2": "2"}


def answers_of(row):
    """(left, right) texts of the two answers the humans compared; "" if missing."""
    def text(keys):
        v = _first(row, keys)
        return v if isinstance(v, str) else ""
    return text(_LEFT_ANSWER), text(_RIGHT_ANSWER)


def questions_match(row):
    """False when the two sides of a pair carry different question texts."""
    left, right = row.get("left_question_text"), row.get("right_question_text")
    if not (isinstance(left, str) and isinstance(right, str)):
        return True
    left, right = left.strip(), right.strip()
    return not (left and right) or left == right


def review_vote_of(row):
    """This one review's vote as a MOOVE vote code ("1" / "2" / "12").
    Deliberately not human_winner_of(): that prefers majority_winner, which
    would replace every reviewer's vote with the pair's majority."""
    v = _first(row, ("Vote", "label"))
    return None if v is None else _REVIEW_VOTE.get(str(v).strip().lower())


def human_winner_of(row):
    """Which side of the ORIGINAL MOOVE pair the human reviewers preferred,
    as "left" / "right" / "tie". Two encodings in the wild: an answer ID
    (majority_winner / winner_id) or a vote code (vote / Vote)."""
    wid = _first(row, ("majority_winner", "winner_id"))
    if wid is not None:
        if str(wid) == str(_first(row, _LEFT_ID)):
            return "left"
        if str(wid) == str(_first(row, _RIGHT_ID)):
            return "right"
    vote = _first(row, ("vote", "Vote"))
    if vote is not None:
        return _VOTE_LABEL.get(str(vote).strip().lower())
    return None


def extract_meta(row, extra=()):
    """Case metadata for one MOOVE row: only keys that are actually present,
    so a log never carries a column of Nones. `extra` names raw source columns
    to pass through verbatim, for exports with fields this vocabulary misses."""
    meta = {}
    for name, aliases in FIELD_ALIASES.items():
        v = _first(row, aliases)
        if v is not None:
            meta[name] = v

    for name, aliases in (("moove_left_model", _LEFT_MODEL),
                          ("moove_right_model", _RIGHT_MODEL)):
        v = _first(row, aliases)
        if v is not None:
            meta[name] = v
    hw = human_winner_of(row)
    if hw is not None:
        meta["moove_human_winner"] = hw

    for k in extra:
        if k in row and row[k] not in (None, "", []):
            meta[k] = row[k]
    return meta


def index_by_pair(path, extra=()):
    """{pair_key: metadata} for a MOOVE export. First row per pair wins --
    duplicates are repeat reviews of the same pair and carry the same
    pair-level fields."""
    import json

    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = pair_key_for(row)
            if key not in index:
                meta = extract_meta(row, extra=extra)
                if meta:
                    index[key] = meta
    return index
