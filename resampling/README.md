# Resampling

Every answer of the training corpus is generated here, by rejection-sampling distillation, after the
questions have been decontaminated. The questions are not changed.

## The scripts

- `resample_answers.py` — pass@k answer generation over any of the four sets, in waves. A
  multiple-choice row keeps a sample, chosen uniformly at random among those that reach the row's
  label; a row with none after k samples goes to `<name>_rejected.jsonl`. Only a sample that stopped
  on its own and reached the final channel is eligible; a sample that hit the token cap is resampled
  with a larger budget and never kept. Open-ended rows (the synthetic vignettes and the open-ended part
  of Synthetic Curated QA) have no verifier and take one complete sample.
- `common.py` — sampling, checkpointing and the schema helpers. The multiple-choice contract
  (`is_mcq`, `MCQ_SUFFIX`, `chosen_letter`) is imported from `truncation/regenerate_truncated.py`.
- `merge_shards.py` — joins the per-shard outputs in the input's order.
- `moove_seed_pool.py` — the few-shot pool of the synthetic vignettes, with every evaluation question
  (the unanimous-576 set and the held-out review split) removed, including near-copies.
- `corpora.py` — the registry of finished corpora (`data/corpora.yaml`).

`resample_answers.py` takes `--selfcheck` (offline unit checks) and `--smoke N` (N rows per input).

Multiple-choice rows whose question lists no options cannot be answered as written; `--no-options
drop` (the default) sends them to the rejected file. Options written as unlettered JSON are rewritten
as a lettered list (`options_letterized: true`).

## Waves, not flat pass@k


`--k-schedule 1,3,4`: one sample for every row, then 3 more only for the rows
that failed, then 4 more. Flat pass@8 pays 8x decode on the rows that pass
first try; the waves pay it only where it is needed, in 3 batched rounds rather
than 8 sequential single-sample ones. `--k-schedule 8` gives the flat version.

The truncation retry is a separate axis from the waves, deliberately: a capped
sample is *unusable*, not *wrong*, so it is resampled with a grown token budget
(`--length-retries`, `--grow`) and does not count against k.

Free-form rows (the MOOVE set, and the open-ended block of
`meditron_4_synthetic_qa_v3`) have no verifier. They take one complete sample
and are checked for completeness only. They do not get `MCQ_SUFFIX`.

## Running it

Both scripts resume: every resolved row is appended to `<outdir>/checkpoint.jsonl` as it lands.
Relaunch with the same `--outdir` until nothing is pending.

```
bash new_launch.sh --requirements none resampling/resample_answers.py \
    --model openai/gpt-oss-120b \
    --input data/decontaminated/gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl \
    --input data/decontaminated/meditron_4_synthetic_qa_v3_fixed_decontam.jsonl \
    --input data/decontaminated/guidelines_qa_v3_full_fixed_decontam.jsonl \
    --outdir $SCRATCH/meditronfo/resampling/answers_gptoss_full
```

`--model` is the answering model; `--model Qwen/Qwen3.8-27B` gives the second teacher's corpus. The
harmony system message and final-channel handling switch off for a model that is not gpt-oss.

## Multi-node


One vLLM spanning nodes would be the wrong tool: gpt-oss-120b fits on a single
node's 4 GH200s, so tensor-parallel across nodes buys nothing and pays NCCL for
it. What scales here is data parallelism — shard the rows, one model replica
per node.

`--shard i/N` processes every Nth row; `--shard auto` reads the shard from the
Slurm array task (or from srun ranks). Each shard writes its own output,
checkpoint and report, and every row carries `source_row` so the merge restores
the input's order exactly.

```
# 8 independent 1-node jobs, each taking an eighth of the rows
bash new_launch.sh --requirements none --array 8 resampling/resample_answers.py --model openai/gpt-oss-120b \
    --input data/decontaminated/meditron_4_synthetic_qa_v3_fixed_decontam.jsonl \
    --outdir $SCRATCH/meditronfo/resampling/answers_gptoss_full --shard auto

python3 resampling/merge_shards.py --outdir $SCRATCH/meditronfo/resampling/answers_gptoss_full
```

An array is preferred over one N-node allocation: tasks start as nodes free up
instead of waiting for N at once, which matters when jobs pend on node limits. `debug` will not take a
wide array (2 submitted jobs per user), so arrays go to `normal` or `low`.

The merge refuses to write a short corpus: a missing shard or a row claimed by
two shards is an error, not a warning. Merge only after every shard reports
done; shards that hit the walltime resume where they stopped when relaunched
with the same `--outdir` and the same `--shard`.

Outputs go to scratch, not the home directory: these files
are hundreds of MB each.

`--model` is the *answering* model and nothing in either script is specific to
gpt-oss, so distilling from Apertus or Gemma-4 is a flag change. The harmony
system message and final-channel handling switch themselves off for a model
that is not gpt-oss, and `<end_of_turn>` / `<|assistant_end|>` are added as stop
tokens when the tokenizer has them.

## Schema


Outputs are normalised before they are written: no `null` anywhere, columns
that are null in every row dropped, typed empties (`""`, `False`, `0`)
everywhere else. The input sets are not clean this way — `synthetic_moove`
carries `exact_match: null` on every row — and a column Arrow infers as `null`
kills the first real value later in the file.

`resample_answers.py` also drops `raw_output` by default (`--drop-cols`): it is
the question generator's stale harmony text, and 8x the guidelines file size.

## Registry


Finished corpora are catalogued in `data/corpora.yaml` by `resampling/corpora.py`,
and live under `data/corpora/<corpus>/<dataset>.jsonl` (keep them on persistent
storage, not scratch).

```
python3 resampling/corpora.py register <name> --model <answering model> \
    --file curated_qa=<path> --file synthetic_qa=<path> \
    --file guidelines_qa=<path> --file synthetic_moove=<path> --job <id> --note "<caveat>"
python3 resampling/corpora.py list
python3 resampling/corpora.py check            # every file still there, same rows and sha256
python3 resampling/corpora.py axolotl <name>   # the datasets: block for a training config
```

Each dataset gets a role name (`curated_qa`, ...) rather than its upstream file
name, so a Qwen answer file is not called `gpt_oss_8retries_...`. A name is
registered once: re-registering it with different bytes is refused, so a version
always means one content, and a changed corpus is a new version. Write the
caveats a trainer needs into `--note`; they travel with the corpus.

Give each corpus its own `dataset_prepared_path` in the axolotl config: axolotl
silently reuses a stale tokenised cache otherwise (CLAUDE.md).

