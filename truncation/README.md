# Truncated answers

The first generation scripts capped generation at `--max-new-tokens 2048`
(`data_gen/distill_with_retries_v3.py`, `data_gen/distill_synthetic_qa_v3.py`) and kept whatever came
out. Nothing checked vLLM's `finish_reason`, so every completion that hit the cap was written to disk as
a finished assistant turn. `finish_reason == "length"` is a near-perfect detector: a set filtered on it
alone came back at 0.02% truncated under an independent LM judge.

`regenerate_truncated.py` repairs the three question sets before decontamination. The answers are
regenerated again later anyway, by pass@8 rejection sampling (`resampling/`), which keeps no capped
sample; this step matters for the questions it drops and for the sets as they enter decontamination.

## Regenerating (`regenerate_truncated.py`)

```
bash new_launch.sh --requirements none truncation/regenerate_truncated.py \
    --model openai/gpt-oss-120b --outdir data/fixed \
    --input data/curated_qa/gpt_oss_8retries_v3.jsonl data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl \
            data/guidelines_qa/guidelines_qa_v3_full.jsonl
```

A turn is regenerated if the row carries `finish_reason == "length"`, or if its answer is flagged in
`results/train/truncated.jsonl` (an LM-judge audit of the training sets). Flags are matched on answer
text. A row is overwritten only when the new sample stopped on its own with a non-empty final channel;
otherwise the original is kept and marked `regen_failed`. Retries grow the token budget (16384 by
default, `--grow`). Everything not truncated is passed through byte for byte.

| dataset | rows | truncated | regenerated | dropped |
|---|---|---|---|---|
| gpt_oss_8retries_v3 (Curated QA) | 216719 | 3833 | 3833 | 0 |
| meditron_4_synthetic_qa_v3 | 214654 | 24258 | 24252 | 6 |
| guidelines_qa_v3_full | 145681 | 320 | 6 | 314 |

All regenerated turns stopped on their own on the first attempt, so the 2048-token cap was the whole
problem. The synthetic vignettes of the paper (v4) were generated and answered after this fix and never
went through this step.

### Dropped rows

98% of the truncated Guidelines QA rows (314/320) have a corrupted *question*: the generator writes ten
vignettes per call, and when the call was cut off the parser salvaged the generator's reasoning as the
question. There is nothing to answer, so these rows are dropped. `unusable_question()` applies the rule
generally (`--keep-unusable` disables it); it also drops the 6 Synthetic Curated QA rows whose question is
empty of content. The same corruption in untruncated Guidelines QA rows (3941) is dropped later by
`resampling/resample_answers.py`.

### Answer format

Synthetic Curated QA was generated question and answer together, in exam format, so a regenerated
multiple-choice answer must end on an option letter: those rows get `MCQ_SUFFIX` appended to the
question. The set also holds open-ended rows that carry a leftover `label_letter`; `is_mcq()` reads the
explicit `is_labeled` flag instead, so the suffix never reaches an open-ended row. The script prints
`<n> MCQ / <n> free-form` per dataset before generating, and `verify_fixed.py` compares each regenerated
subgroup's answer-format rate with the rest of the same subgroup, in both directions.
