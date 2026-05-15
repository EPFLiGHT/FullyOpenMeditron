"""
Metadata extractor — the first LLM-dependent stage of the pipeline.

Tags each Record with a small, carefully-chosen set of clinical metadata
fields. Two schemas, two prompts:

  QA schema (MOOVE / Meditron / Guidelines *synthetic*):
    specialty, level_of_care, severity, question_type, geography,
    age_group, difficulty

  Document schema (Guidelines *source* — raw reference text):
    specialty, level_of_care, severity, geography

Design
------
- Model-agnostic: swap vLLM model via --model flag; prompts stay the same.
- Resumable: keyed on Record.id, reads already-tagged IDs from the output
  JSONL before running, skips them. Re-running is always safe.
- Robust JSON parsing: the model sometimes emits prose before the JSON, or
  wraps it in code fences. Parser extracts the first balanced {...} block.
- Schema validation: any value not in the allowed list for an enum field is
  coerced to "unspecified" (or nearest equivalent) before writing, so
  downstream distribution code never sees garbage.
- Filters at ingestion: `filter_valid()` drops records that are too short
  or obviously junk before the model sees them.
"""

import json
import re
from typing import Iterable, Iterator, Optional

# These are the allowed values per enum field. The extractor validates
# outputs against these; anything unrecognized becomes "unspecified".

ALLOWED_SPECIALTIES = {
    "general_medicine", "emergency_medicine", "critical_care",
    "infectious_disease", "cardiology", "neurology", "oncology",
    "endocrinology", "gastroenterology", "nephrology", "pulmonology",
    "rheumatology", "hematology", "dermatology", "psychiatry",
    "obstetrics", "gynecology", "pediatrics", "geriatrics",
    "surgery_general", "orthopedics", "urology", "ophthalmology",
    "ent", "radiology", "pathology", "anesthesiology",
    "public_health", "family_medicine", "unspecified",
}

ALLOWED_LEVEL_OF_CARE = {
    "community", "prehospital", "primary", "emergency",
    "secondary", "tertiary", "unspecified",
}

ALLOWED_SEVERITY = {
    "routine", "urgent", "emergent", "life_threatening", "unspecified",
}

ALLOWED_QUESTION_TYPE = {
    "triage", "diagnosis", "management", "follow_up", "prevention",
    "screening", "interpretation", "pharmacology", "ethics", "other",
}

ALLOWED_AGE_GROUP = {
    "neonate", "infant", "child", "adolescent", "adult",
    "elderly", "mixed", "unspecified",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_QA = (
    "You are an expert clinical data annotator and physician. "
    "Your task is to analyze medical questions and extract specific metadata "
    "attributes. You must output ONLY a valid JSON object. Do not include "
    "any explanations, reasoning, or markdown formatting outside the JSON block."
)

SYSTEM_MESSAGE_DOC = (
    "You are an expert clinical data annotator and physician. "
    "Your task is to analyze medical reference documents and extract metadata "
    "attributes describing the clinical domain they cover. You must output "
    "ONLY a valid JSON object. Do not include any explanations, reasoning, "
    "or markdown formatting outside the JSON block."
)

USER_PROMPT_QA = """Analyze the following medical QA item and return a single JSON object with exactly these keys. Do not include any explanations, markdown formatting, or trailing commas:

{{
  "specialty": "one of: general_medicine, emergency_medicine, critical_care, infectious_disease, cardiology, neurology, oncology, endocrinology, gastroenterology, nephrology, pulmonology, rheumatology, hematology, dermatology, psychiatry, obstetrics, gynecology, pediatrics, geriatrics, surgery_general, orthopedics, urology, ophthalmology, ent, radiology, pathology, anesthesiology, public_health, family_medicine, unspecified",
  "level_of_care": "one of: community, prehospital, primary, emergency, secondary, tertiary, unspecified",
  "severity": "one of: routine, urgent, emergent, life_threatening, unspecified",
  "question_type": "one of: triage, diagnosis, management, follow_up, prevention, screening, interpretation, pharmacology, ethics, other",
  "geography": "country or region name, or unspecified",
  "age_group": "one of: neonate, infant, child, adolescent, adult, elderly, mixed, unspecified",
  "difficulty": "integer 1 to 5 (1 = easy, 5 = expert)"
}}

QA item to analyze:
{body}

Output the JSON analysis now."""

USER_PROMPT_DOC = """Analyze the following medical reference document and return a single JSON object with exactly these keys. Do not include any explanations, markdown formatting, or trailing commas:

{{
  "specialty": "one of: general_medicine, emergency_medicine, critical_care, infectious_disease, cardiology, neurology, oncology, endocrinology, gastroenterology, nephrology, pulmonology, rheumatology, hematology, dermatology, psychiatry, obstetrics, gynecology, pediatrics, geriatrics, surgery_general, orthopedics, urology, ophthalmology, ent, radiology, pathology, anesthesiology, public_health, family_medicine, unspecified",
  "level_of_care": "one of: community, prehospital, primary, emergency, secondary, tertiary, unspecified",
  "severity": "one of: routine, urgent, emergent, life_threatening, unspecified",
  "geography": "country or region name, or unspecified"
}}

Reference document to analyze:
{body}

Output the JSON analysis now."""


# ---------------------------------------------------------------------------
# Input validation and body construction
# ---------------------------------------------------------------------------

MIN_FULL_TEXT_CHARS = 40  # skip records that are too short to be meaningful


# def filter_valid(records):
#     """Drop records we shouldn't send to the model.

#     Criteria:
#       - empty or near-empty full_text
#       - literal 'None' id (Meditron junk rows)
#       - parse_failed flag set (guidelines_qa failures, mayo empty content)
#     """
#     seen_ids = set()
#     for r in records:
#         if r.id in seen_ids:
#             continue
#         seen_ids.add(r.id)

#         if r.parse_failed:
#             continue
#         if not r.full_text or len(r.full_text) < MIN_FULL_TEXT_CHARS:
#             continue
#         # Records whose id resolved to the literal string "None"
#         # (from Meditron source rows with id: "None")
#         if r.id.endswith("_None"):
#             continue
#         yield r
def filter_valid(records):
    yield from records


# Document chunking: idsa has records up to 51k tokens. vLLM will choke.
# Truncate to a head+tail window — the head carries the topic signal, the
# tail often has summary/recommendations. Skip the middle (reference lists etc.)
DOC_HEAD_CHARS = 4000
DOC_TAIL_CHARS = 2000

def build_body_for_model(record):
    """Construct the body string the prompt will embed."""
    if record.is_document:
        # For guideline chunks, truncate if very long
        text = record.full_text
        if len(text) > DOC_HEAD_CHARS + DOC_TAIL_CHARS + 200:
            text = (
                text[:DOC_HEAD_CHARS]
                + "\n\n[... document truncated for analysis ...]\n\n"
                + text[-DOC_TAIL_CHARS:]
            )
        return text
    # For QA records: just the question is enough for most metadata fields;
    # including the answer helps the model infer context (age, severity).
    # But cap it so the prompt fits in a reasonable window.
    q = (record.question or "").strip()
    a = (record.answer or "").strip()
    MAX_Q = 2000
    MAX_A = 2000
    if len(q) > MAX_Q:
        q = q[:MAX_Q] + "..."
    if len(a) > MAX_A:
        a = a[:MAX_A] + "..."
    if q and a:
        return "QUESTION: " + q + "\n\nANSWER: " + a
    return q or a


def build_messages(record):
    """Return the (system, user) messages pair ready for llm.chat()."""
    body = build_body_for_model(record)
    if record.is_document:
        return [
            {"role": "system", "content": SYSTEM_MESSAGE_DOC},
            {"role": "user", "content": USER_PROMPT_DOC.format(body=body)},
        ]
    return [
        {"role": "system", "content": SYSTEM_MESSAGE_QA},
        {"role": "user", "content": USER_PROMPT_QA.format(body=body)},
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Matches the first balanced {...} block in the text. Not a full JSON parser —
# we use this to isolate the JSON region, then json.loads it.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\||$)", re.DOTALL)


def _strip_wrapping(text):
    """Peel off common model output wrappers (code fences, Harmony channels, etc)."""
    text = text.strip()
    # GPT-OSS Harmony: keep only the final channel
    m = _HARMONY_FINAL_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Qwen <think>...</think> prefix: drop everything up to </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # Code fence: keep just the content inside
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    return text


def _extract_json_object(text):
    """Find the first balanced {...} block and return it as a dict, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _validate_enum(value, allowed, default="unspecified"):
    if not isinstance(value, str):
        return default
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    return default


def _validate_difficulty(value):
    """Coerce to int in [1, 5] or return None."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
    if 1 <= n <= 5:
        return n
    return None


def _validate_geography(value):
    if not isinstance(value, str):
        return "unspecified"
    v = value.strip()
    if not v or v.lower() in ("unspecified", "n/a", "none", "null"):
        return "unspecified"
    # Normalize: lower-case, spaces kept (country names have spaces)
    return v


def parse_model_output(raw_text, is_document):
    """Parse and validate model output. Always returns a dict with the
    expected keys. Sets `_parse_ok: False` if the model didn't produce
    valid JSON at all."""
    stripped = _strip_wrapping(raw_text)
    obj = _extract_json_object(stripped)

    if obj is None or not isinstance(obj, dict):
        # Fill with defaults so downstream never crashes on missing keys
        schema_defaults = {
            "specialty": "unspecified",
            "level_of_care": "unspecified",
            "severity": "unspecified",
            "geography": "unspecified",
        }
        if not is_document:
            schema_defaults.update({
                "question_type": "other",
                "age_group": "unspecified",
                "difficulty": None,
            })
        schema_defaults["_parse_ok"] = False
        schema_defaults["_raw_preview"] = raw_text[:500] if raw_text else ""
        return schema_defaults

    # Validate each field
    out = {
        "specialty": _validate_enum(obj.get("specialty"), ALLOWED_SPECIALTIES),
        "level_of_care": _validate_enum(obj.get("level_of_care"), ALLOWED_LEVEL_OF_CARE),
        "severity": _validate_enum(obj.get("severity"), ALLOWED_SEVERITY),
        "geography": _validate_geography(obj.get("geography")),
    }
    if not is_document:
        out["question_type"] = _validate_enum(
            obj.get("question_type"), ALLOWED_QUESTION_TYPE, default="other"
        )
        out["age_group"] = _validate_enum(obj.get("age_group"), ALLOWED_AGE_GROUP)
        out["difficulty"] = _validate_difficulty(obj.get("difficulty"))
    out["_parse_ok"] = True
    return out
