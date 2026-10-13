#!/usr/bin/env python3
# auto_moove/auto_moove_common.py

import re, gc, shelve, threading
import os, json, time, random, hashlib
from pathlib import Path

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

from auto_moove_prompt import MOOVE, CHECKLIST, criteria_for, judge_prompt, verdict_model

# Answer cache. AUTO_MOOVE_CACHE_ROOT (env or .env) points a run at its own,
# e.g. off the cluster or without write access here. Still single-writer. The
# default lives on scratch, which the cluster purges: answers not reused for a while
# are gone (most pre-September ones already are), and a rerun regenerates them.
CACHE_ROOT = Path(os.environ.get("AUTO_MOOVE_CACHE_ROOT",
                                 os.path.join(os.environ.get("SCRATCH", "/tmp"), "meditronfo", "auto_moove_cache")))

# A retry after unparseable output must not resend the identical request:
# under greedy decoding and a fixed seed it gets the identical output back.
# Retries sample at this temperature with seed + attempt, so they differ from
# the first try and from each other, yet still reproduce run to run.
RETRY_TEMPERATURE = 0.7

# Sampling params a provider refused, per model name. OpenAI's reasoning
# models (o-series, GPT-5 family) reject any temperature but their default,
# so every call failed once temperature=0.0 was pinned. The refused param is
# dropped for the rest of the run, with a warning, and effective_temperature()
# lets results record that the run was not greedy.
_refused = {}
_refused_lock = threading.Lock()


def _refused_param(err):
    """The sampling param an HTTP 400 rejects as unsupported, else None."""
    if getattr(err, "status_code", None) != 400:
        return None
    msg = str(err).lower()
    for p in ("temperature", "seed"):
        if getattr(err, "param", None) == p or (p in msg and "support" in msg):
            return p
    return None


def _note_refused(model_name, param, value):
    with _refused_lock:
        if param in _refused.setdefault(model_name, set()):
            return
        _refused[model_name].add(param)
    print(f">>> WARNING: {model_name} refuses {param}={value}; using the provider's "
          f"default instead. This run is NOT greedy and will not reproduce exactly. <<<")


def effective_temperature(model_name, requested):
    """The temperature a model actually ran at: None when its provider refused
    the requested one and sampled at its own default."""
    return None if "temperature" in _refused.get(model_name, ()) else requested


def has_text(answer):
    """False for a failed (None) or empty answer, which must never be cached
    or judged as if it were the model's reply."""
    return isinstance(answer, str) and bool(answer.strip())

MAX_RETRIES = 1

PRICES = {  # $ per 1M tokens (input, output) -- update when the provider does
    # ---- OpenAI ---------------------------------------------------------
    # GPT-5.6 family, GA 2026-07-09, 1.05M context
    "gpt-5.6-sol":           (5.00,  30.00),
    "gpt-5.6-terra":         (2.00,  12.00),
    "gpt-5.6-luna":          (0.20,   1.20),
    "gpt-5.5":               (5.00,  30.00),
    "gpt-5.5-pro":          (30.00, 180.00),
    "gpt-5.4":               (2.50,  15.00),
    "gpt-5.4-pro":          (30.00, 180.00),
    "gpt-5.4-mini":          (0.75,   4.50),
    "gpt-5.4-nano":          (0.20,   1.25),
    "gpt-5.1":               (1.25,  10.00),
    "gpt-5-mini":            (0.25,   2.00),
    "gpt-5-nano":            (0.05,   0.40),
    "gpt-4.1":               (2.00,   8.00),
    "gpt-4.1-mini":          (0.40,   1.60),
    "gpt-4.1-nano":          (0.10,   0.40),
    "gpt-4o":                (2.50,  10.00),
    "gpt-4o-mini":           (0.15,   0.60),
    "o3":                    (2.00,   8.00),
    "o3-mini":               (1.10,   4.40),
    "o4-mini":               (1.10,   4.40),

    # ---- Google Gemini --------------------------------------------------
    "gemini-3.6-flash":      (1.50,   7.50),
    "gemini-3.5-flash":      (1.50,   9.00),
    "gemini-3.5-flash-lite": (0.30,   2.50),
    "gemini-3.1-pro":        (2.00,  12.00),  # >200K ctx: (4.00, 18.00)
    "gemini-3.1-flash-lite": (0.25,   1.50),
    "gemini-3-pro":          (2.00,  12.00),  # >200K ctx: (4.00, 18.00)
    "gemini-3-flash":        (0.50,   3.00),
    "gemini-2.5-pro":        (1.25,  10.00),  # >200K ctx: (2.50, 15.00)
    "gemini-2.5-flash":      (0.30,   2.50),
    "gemini-2.5-flash-lite": (0.10,   0.40),
}
_usage = []

def report_cost():
    if not _usage:
        return
    tin = sum(u[1] for u in _usage)
    tout = sum(u[2] for u in _usage)
    known = [u for u in _usage if u[0] in PRICES]
    cost = sum(i * PRICES[m][0] + o * PRICES[m][1] for m, i, o in known) / 1e6
    tag = "" if len(known) == len(_usage) else f" ({_usage[0][0]} not in PRICES)"
    print(f">>> {len(_usage)} calls | {tin:,} in / {tout:,} out | ~${cost:.2f}{tag} <<<")

def _load_vllm(model_name, *, tp, utilization, max_model_len, reasoning_parser=None,
               max_num_batched_tokens=None):
    """The engine settings every local model needs on the cluster, in one place.
    reasoning_parser: a thinking judge reasons freely, and the structured-output
    grammar applies only once its reasoning has ended."""
    import torch
    from vllm import LLM

    tp_size = tp if tp > 0 else torch.cuda.device_count()
    print(f"\n>>> Loading {model_name} <<<")
    kv_cache_dtype = "fp8" if "deepseek-v4" in model_name.lower() else "auto"
    extra = {"reasoning_parser": reasoning_parser} if reasoning_parser else {}
    if max_num_batched_tokens:
        extra["max_num_batched_tokens"] = max_num_batched_tokens
    return LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=utilization,
        trust_remote_code=True,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        compilation_config={"pass_config": {"fuse_allreduce_rms": False}},
        **extra,
    )


def _release_vllm():
    """Free the GPUs for the next model. Call after dropping the last LLM reference."""
    import torch
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def _no_thinking(model_name, tokenizer):
    """Chat-template kwargs that switch hybrid thinking off: Qwen3, and
    Nemotron-3.5, whose template thinks unless told not to (it then opens the
    turn with an empty <think></think>)."""
    name = model_name.lower()
    if "qwen" in name or "qwen" in type(tokenizer).__name__.lower() or "nemotron" in name:
        return {"enable_thinking": False}
    if "muse-glimmer" in name:
        return {"reasoning_strength": "low"}
    return {}


# Judges that have no thinking switch, only a reasoning channel the model
# chooses to open. Muse-Glimmer writes its reasoning as an assistant message
# "to=self" and its answer "to=user"; opening the answer message in the prompt
# skips the reasoning, as Nemotron's empty <think></think> does.
ANSWER_PREFILL = {"muse-glimmer": " to=user<|message|>"}


def _answer_prefill(model_name):
    return next((v for k, v in ANSWER_PREFILL.items() if k in model_name.lower()), "")


def _render_judge_prompt(tokenizer, model_name, msgs, chat_kwargs, *, tokenize):
    """The judge's chat prompt, with the answer prefill of a thinking-only
    model; text, or token ids when tokenize."""
    prefill = _answer_prefill(model_name)
    if not prefill:
        if not tokenize:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 **chat_kwargs)
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, **chat_kwargs)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         **chat_kwargs) + prefill
    # the template writes the BOS itself
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)] if tokenize else text


# Judges that can think (--thinking): vLLM reasoning parser by model-name
# substring. Both templates read enable_thinking; Gemma-4's defaults it to off.
REASONING_PARSERS = {"gemma-4": "gemma4", "qwen3": "qwen3"}
# (start, end) of the reasoning in the decoded output. Qwen3 puts <think> in
# the prompt, so its output may carry only the end tag.
REASONING_TAGS = (("<|channel>", "<channel|>"), ("<think>", "</think>"))


# gpt-oss writes harmony: its reasoning in an analysis channel, then the answer
# in a final channel. Decoded with special tokens, the raw text carries both.
HARMONY_FINAL = "<|channel|>final<|message|>"


def harmony_final(text):
    """gpt-oss: the final channel only. Reasoning that never reached it is no
    answer (""), so it is neither cached nor judged, and a judge retries."""
    if HARMONY_FINAL in text:
        text = text.rpartition(HARMONY_FINAL)[2]
        for tok in ("<|return|>", "<|end|>"):
            text = text.split(tok)[0]
        return text.strip()
    return "" if "<|channel|>" in text else text


def _reasoning_parser(model_name):
    name = model_name.lower()
    for key, parser in REASONING_PARSERS.items():
        if key in name:
            return parser
    raise ValueError(f"No reasoning parser known for {model_name}: add it to REASONING_PARSERS")


def generate_answers(model_name, prompts, *, max_new_tokens, tp, utilization,
                     schema=None, max_model_len=8192, temperature=0.0, seed=42,
                     samples=1, thinking=False):
    """schema: pydantic model class to constrain output, or None for free text.
    samples > 1: every prompt is decoded that many times, and each result is
    the list of its samples. thinking: the model reasons first; its reasoning
    stays in the returned text, and extract_judge_data skips it."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # One request per sample, regrouped per prompt at the end.
    n_prompts = len(prompts)
    prompts = [p for p in prompts for _ in range(samples)]

    llm = _load_vllm(model_name, tp=tp, utilization=utilization, max_model_len=max_model_len,
                     reasoning_parser=_reasoning_parser(model_name) if thinking else None)
    tokenizer = llm.get_tokenizer()
    # A judge has to answer in JSON straight away, so hybrid-thinking Qwen3
    # models get thinking off (from a co-author's PR). Templates that never read the
    # flag ignore it -- Qwen3-30B-A3B-Instruct-2507's doesn't read it -- and
    # answer generation keeps each model's default. thinking turns it on
    # instead; the reasoning parser holds the grammar back until it ends.
    chat_kwargs = ({"enable_thinking": True} if thinking else
                   _no_thinking(model_name, tokenizer) if schema is not None else {})

    # a judge answering straight away gets a thinking-only model's answer prefill
    prefill_model = model_name if (schema is not None and not thinking) else ""

    def render(idxs):
        out = []
        for i in idxs:
            p = prompts[i]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            out.append(_render_judge_prompt(tokenizer, prefill_model, msgs, chat_kwargs,
                                            tokenize=False))
        return out

    # gpt-oss ignores the structured-outputs backend, so it gets prompt-only JSON
    # plus retries; everything else is grammar-constrained.
    grammar = (schema is not None and model_name != "openai/gpt-oss-120b")
    # vLLM defaults to temperature=1.0, i.e. sampling. Left unset, both answer
    # generation and judging were stochastic: the same inputs gave different
    # win rates run to run, so no result could be reproduced and no refactor
    # could be verified against a previous one. Pin greedy decoding and a seed.
    #
    # Sampled decoding gets a distinct seed per request. vLLM seeds a generator
    # per request, so requests sharing a seed draw the same noise at every
    # step: all prompts then tilt the same way at the verdict token, and the AB
    # and BA orders lean to the same position -- a spurious disagreement.
    # Greedy decoding keeps seed + attempt, so earlier retries reproduce.
    per_request = samples > 1 or temperature > 0

    def sampling_params(attempt, request):
        return SamplingParams(
            max_tokens=max_new_tokens,
            # the first pass decodes as asked; a retry resamples (RETRY_TEMPERATURE)
            temperature=max(temperature, RETRY_TEMPERATURE) if attempt else temperature,
            seed=None if seed is None else
                 seed + attempt + (1000 * request if per_request else 0),
            skip_special_tokens=False,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
            structured_outputs=StructuredOutputsParams(json=schema.model_json_schema())
            if grammar else None,
        )

    # A template that opens the reasoning in the prompt (Qwen3.8 ends it with
    # <think>) leaves it out of the output; put it back, so the text is whole
    # and extract_judge_data can tell reasoning that never ended from none.
    opened = None
    if thinking and prompts:
        head = render([0])[0].rstrip()
        opened = next((s for s, _ in REASONING_TAGS if head.endswith(s)), None)

    # Without this an answer from gpt-oss handed the judge its reasoning too,
    # and a judge's JSON was read from the first draft in its reasoning.
    harmony = "gpt-oss" in model_name.lower()

    results = [None] * len(prompts)
    pending = list(range(len(prompts)))
    max_attempts = (MAX_RETRIES + 1) if (schema is not None and not grammar) else 1

    for attempt in range(max_attempts):
        if not pending:
            break
        if attempt:
            print(f"\n--- Retry {attempt}/{MAX_RETRIES} for {len(pending)} prompts ---")
        outputs = llm.generate(prompts=render(pending),
                               sampling_params=[sampling_params(attempt, i) for i in pending])
        still = []
        for idx, out in zip(pending, outputs):
            results[idx] = out.outputs[0].text.strip()
            if harmony:
                results[idx] = harmony_final(results[idx])
            if opened and opened not in results[idx]:
                results[idx] = opened + results[idx]
            if schema is not None and extract_judge_data(results[idx], schema) is None:
                still.append(idx)
        pending = still

    if pending:
        print(f">>> WARNING: {len(pending)} judge outputs unparseable after retries <<<")

    del llm
    _release_vllm()
    if samples > 1:
        results = [results[i * samples:(i + 1) * samples] for i in range(n_prompts)]
    return results



PROVIDERS = {
    "openai": dict(base="https://api.openai.com/v1",
                   key="OPENAI_API_KEY", strict=True,  tok="max_completion_tokens"),
    "gemini": dict(base="https://generativelanguage.googleapis.com/v1beta/openai/",
                   key="GEMINI_API_KEY", strict=False, tok="max_tokens"),
    "rcp":    dict(base="https://<openai-compatible-endpoint>/v1",
                   key="RCP_API_KEY",    strict=False, tok="max_tokens"),
    "local":  dict(base="http://localhost:8000/v1",
                   key="OPENAI_API_KEY", strict=True,  tok="max_tokens"),  # vllm serve
}


def _strict_schema(schema):
    """OpenAI strict mode: every object closed, every property required."""
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                n["additionalProperties"] = False
                n["required"] = list(n.get("properties", {}))
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    s = schema.model_json_schema()
    walk(s)
    return s


def extract_judge_data(text, schema):
    """First JSON object in `text` that validates against `schema`, else None.
    A thinking judge's reasoning is skipped: a draft verdict inside it is not
    the answer, and reasoning that never ended has no answer at all."""
    if not text:
        return None
    for start, end in REASONING_TAGS:
        if end in text:
            text = text.rpartition(end)[2]
        elif start in text:
            return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)      # balanced, unlike a greedy regex
        except ValueError:
            continue
        try:
            return schema.model_validate(obj)
        except Exception:
            continue
    return None


def _api_kwargs(cfg, *, max_new_tokens, schema, temperature, seed, reasoning_effort):
    """Request params shared by generate_answers_api and generate_answers_batch."""
    kwargs = {cfg["tok"]: max_new_tokens, "temperature": temperature}
    if seed is not None:
        kwargs["seed"] = seed          # honoured by OpenAI-compatible servers
    if reasoning_effort is not None:
        # not among _refused_param's params: a model that rejects it fails
        # loudly instead of silently reasoning at (and billing for) its default
        kwargs["reasoning_effort"] = reasoning_effort
    if schema is not None:
        kwargs["response_format"] = (
            {"type": "json_schema", "json_schema":
                {"name": "verdict", "strict": True, "schema": _strict_schema(schema)}}
            if cfg["strict"] else {"type": "json_object"})
    return kwargs


def generate_answers_api(model_name, prompts, *, max_new_tokens, schema=None,
                         max_workers=8, max_retries=4, temperature=0.0, seed=42,
                         reasoning_effort=None):
    """model_name is 'provider:model', e.g. 'gemini:gemini-2.5-pro'."""
    provider, _, model = model_name.partition(":")
    cfg = PROVIDERS[provider]
    client = OpenAI(base_url=cfg["base"], api_key=os.environ[cfg["key"]])

    kwargs = _api_kwargs(cfg, max_new_tokens=max_new_tokens, schema=schema,
                         temperature=temperature, seed=seed,
                         reasoning_effort=reasoning_effort)

    def one(p):
        messages = p if isinstance(p, list) else [{"role": "user", "content": p}]
        text, err, invalid = None, None, False
        # one retry loop for both transport failures and unparseable JSON
        for attempt in range(max_retries + 1):
            refused = _refused.get(model_name, ())
            params = {k: v for k, v in kwargs.items() if k not in refused}
            if invalid:                          # resample, see RETRY_TEMPERATURE
                if "temperature" in params:
                    params["temperature"] = max(temperature, RETRY_TEMPERATURE)
                if "seed" in params:
                    params["seed"] = seed + attempt
            try:
                r = client.chat.completions.create(model=model, messages=messages, **params)
                if r.usage:
                    _usage.append((model, r.usage.prompt_tokens, r.usage.completion_tokens))
                text = (r.choices[0].message.content or "").strip()
                if schema is None or extract_judge_data(text, schema) is not None:
                    return text
                err, invalid = "output did not validate against schema", True
            except Exception as e:
                bad = _refused_param(e)
                if bad in params:
                    _note_refused(model_name, bad, params[bad])
                    continue                     # resend at once, without it
                err = repr(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt + random.random())   # jitter for 429s
        print(f"  !! giving up after {max_retries} retries: {err}")
        return text          # keep the last text so the failure is inspectable

    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, p): i for i, p in enumerate(prompts)}
        for fut in futures:
            results[futures[fut]] = fut.result()
    report_cost()
    return results


BATCH_POLL_SECONDS = 30


def generate_answers_batch(model_name, prompts, *, max_new_tokens, schema=None,
                           temperature=0.0, seed=42, reasoning_effort=None,
                           state_path):
    """generate_answers_api through the OpenAI Batch API: about half the price,
    but asynchronous (up to 24 h). One synchronous probe goes first, because a
    batch cannot drop a refused param and resend. The batch id is kept in
    state_path, so rerunning the same command collects a batch that was
    interrupted while waiting instead of paying for it twice. Requests the
    batch failed, or whose output does not parse, are redone synchronously."""
    provider, _, model = model_name.partition(":")
    cfg = PROVIDERS[provider]
    client = OpenAI(base_url=cfg["base"], api_key=os.environ[cfg["key"]])
    api = dict(max_new_tokens=max_new_tokens, schema=schema, temperature=temperature,
               seed=seed, reasoning_effort=reasoning_effort)
    messages = [p if isinstance(p, list) else [{"role": "user", "content": p}]
                for p in prompts]
    digest = hashlib.sha256(json.dumps(
        [model, messages, max_new_tokens, reasoning_effort, schema and schema.__name__]
    ).encode()).hexdigest()

    state_file = Path(state_path)
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    if state and state["requests_sha256"] != digest:
        raise SystemExit(f"{state_path} belongs to a different set of requests; "
                         f"delete it to submit a new batch")
    if state is None:
        if not has_text(generate_answers_api(model_name, prompts[:1], **api)[0]):
            raise SystemExit("the probe request failed; not submitting the batch")
        refused = sorted(_refused.get(model_name, ()))
        params = {k: v for k, v in _api_kwargs(cfg, **api).items() if k not in refused}
        input_file = Path(f"{state_path}.input.jsonl")
        input_file.write_text("".join(
            json.dumps({"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions",
                        "body": {"model": model, "messages": m, **params}}) + "\n"
            for i, m in enumerate(messages)))
        with open(input_file, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id, completion_window="24h",
                                      endpoint="/v1/chat/completions")
        state = {"batch_id": batch.id, "requests_sha256": digest,
                 "requests": len(prompts), "refused": refused}
        state_file.write_text(json.dumps(state, indent=2))
        shown = ", ".join(f"{k}={v}" for k, v in params.items() if k != "response_format")
        print(f">>> submitted batch {batch.id}: {len(prompts)} requests ({shown}); "
              f"rerun the same command to collect it if this process stops <<<", flush=True)
    for param in state.get("refused", []):          # so results record them on a rerun
        _refused.setdefault(model_name, set()).add(param)

    last = None
    while True:
        batch = client.batches.retrieve(state["batch_id"])
        c = batch.request_counts
        now = (batch.status, c.completed if c else 0, c.failed if c else 0)
        if now != last:
            print(f"  batch {batch.id}: {batch.status}, {now[1]}/{c.total if c else '?'} "
                  f"done, {now[2]} failed", flush=True)
            last = now
        if batch.status == "failed":
            raise SystemExit(f"batch {batch.id} failed: {batch.errors}")
        if batch.status in ("completed", "expired", "cancelled"):
            break
        time.sleep(BATCH_POLL_SECONDS)

    results, tin, tout, reasoning = [None] * len(prompts), 0, 0, 0
    for file_id in (batch.output_file_id, batch.error_file_id):
        if not file_id:
            continue
        for line in client.files.content(file_id).text.splitlines():
            row = json.loads(line)
            resp = row.get("response") or {}
            if resp.get("status_code") != 200:
                continue
            body = resp["body"]
            results[int(row["custom_id"])] = (body["choices"][0]["message"]["content"]
                                              or "").strip()
            usage = body.get("usage") or {}
            tin += usage.get("prompt_tokens", 0)
            tout += usage.get("completion_tokens", 0)
            reasoning += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    price = PRICES.get(model)
    cost = (f"~${(tin * price[0] + tout * price[1]) / 1e6 / 2:.2f} at half the PRICES rate"
            if price else f"{model} not in PRICES")
    print(f">>> batch {batch.status}: {sum(r is not None for r in results)} of "
          f"{len(prompts)} answered | {tin:,} in / {tout:,} out ({reasoning:,} reasoning) "
          f"| {cost} <<<")

    redo = [i for i, t in enumerate(results)
            if not has_text(t) or (schema is not None and extract_judge_data(t, schema) is None)]
    if len(redo) > max(20, len(prompts) // 10):
        raise SystemExit(f"{len(redo)} of {len(prompts)} requests missing from batch "
                         f"{batch.id}; not redoing that many at full price")
    if redo:
        print(f">>> {len(redo)} requests failed in the batch or did not parse; "
              f"redoing them synchronously <<<")
        for i, text in zip(redo, generate_answers_api(model_name,
                                                      [prompts[i] for i in redo], **api)):
            results[i] = text
    return results


def generate(model_name, prompts, *, max_new_tokens, tp=0, utilization=0.85,
             schema=None, max_model_len=8192, temperature=0.0, seed=42, samples=1,
             reasoning_effort=None, batch_state=None, thinking=False):
    """API if the name is prefixed with a known provider, local vLLM otherwise.
    batch_state: where generate_answers_batch keeps its batch id; None calls
    the API directly. thinking: local models only (API: reasoning_effort)."""
    if model_name.partition(":")[0] in PROVIDERS:
        if samples > 1:
            raise ValueError("samples > 1 is implemented for local vLLM models only")
        if thinking:
            raise ValueError("thinking is for local vLLM models; API models take reasoning_effort")
        if batch_state is not None:
            return generate_answers_batch(model_name, prompts, max_new_tokens=max_new_tokens,
                                          schema=schema, temperature=temperature, seed=seed,
                                          reasoning_effort=reasoning_effort,
                                          state_path=batch_state)
        return generate_answers_api(model_name, prompts, max_new_tokens=max_new_tokens,
                                    schema=schema, temperature=temperature, seed=seed,
                                    reasoning_effort=reasoning_effort)
    return generate_answers(model_name, prompts, max_new_tokens=max_new_tokens,
                            tp=tp, utilization=utilization, schema=schema,
                            max_model_len=max_model_len,
                            temperature=temperature, seed=seed, samples=samples,
                            thinking=thinking)


def score_judge_candidates(model_name, prompts, *, tp=0, utilization=0.85, max_model_len=8192):
    """Score each candidate verdict instead of generating one -- a co-author's
    calibration protocol; winner rubric, local vLLM only. For every label the
    judge reads the prompt followed by '{"Winner":"<label>"}', and the score
    is the mean log-probability of those completion tokens.

    Returns one {"Model 1": s, "Model 2": s, "Equal": s} dict per prompt.

    ponytail: the MEAN is kept as calibrated, so his --equal-threshold values
    stay valid. It is not a sequence log-probability: the '{"Winner":"'
    prefix all labels share does not cancel, and "Equal" has fewer tokens
    than "Model 1", which tilts Equal against the two answers. The SUM is the
    principled score (the shared prefix then cancels in the softmax); switch
    to it together with a recalibrated threshold.
    """
    from vllm import SamplingParams

    # prompt_logprobs materialises fp32 log-probs over the whole vocabulary
    # for every token of a prefill chunk, outside vLLM's memory budget: with
    # Gemma-4's 262k vocabulary a 15k-token chunk is 14.8 GiB, which OOMed
    # the judge on Apertus-70B-Instruct's long answers (2026-09-23). 4096
    # tokens per chunk bounds it at 4 GiB.
    llm = _load_vllm(model_name, tp=tp, utilization=utilization, max_model_len=max_model_len,
                     max_num_batched_tokens=4096)
    tokenizer = llm.get_tokenizer()
    chat_kwargs = _no_thinking(model_name, tokenizer)

    requests, slots = [], []        # slots: (prompt index, label, completion start, completion ids)
    for i, p in enumerate(prompts):
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        prefix = _render_judge_prompt(tokenizer, model_name, msgs, chat_kwargs, tokenize=True)
        for label in ("Model 1", "Model 2", "Equal"):
            completion = json.dumps({"Winner": label}, ensure_ascii=False, separators=(",", ":"))
            ids = [int(t) for t in tokenizer.encode(completion, add_special_tokens=False)]
            requests.append({"prompt_token_ids": prefix + ids})
            slots.append((i, label, len(prefix), ids))

    # Nothing generated matters (max_tokens=1): prompt_logprobs=1 returns the
    # log-prob of every prompt token, which is all the scoring needs.
    outputs = llm.generate(prompts=requests,
                           sampling_params=SamplingParams(temperature=0.0, max_tokens=1,
                                                          prompt_logprobs=1))
    scores = [{} for _ in prompts]
    for out, (i, label, start, ids) in zip(outputs, slots):
        lps = []
        for off, tok in enumerate(ids):
            lp = out.prompt_logprobs[start + off][tok]
            lps.append(float(getattr(lp, "logprob", lp)))
        scores[i][label] = sum(lps) / len(lps)

    del llm
    _release_vllm()
    return scores


def schema_tag(schema):
    """Cache-key component: two rubrics must not share a cache entry."""
    if schema is None:
        return "free"
    blob = json.dumps(schema.model_json_schema(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _prompt_hash(prompt) -> str:
    if isinstance(prompt, list):
        payload = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    else:
        payload = prompt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_answers_cached(model_name, prompts, args):

    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    # Decoding params are part of the cache identity. Without them a greedy
    # run would be silently served answers sampled at temperature 1.0, and the
    # "deterministic" result would not be deterministic at all. Answers cached
    # before this change carry the bare "n<tokens>" tag and are simply missed.
    _temp = getattr(args, "temperature", 0.0)
    _seed = getattr(args, "seed", 42)
    marker = "__provider_refused_temperature__"

    def keys_for(temp):
        tag = f"n{args.max_new_tokens}_t{temp}_s{_seed}"
        return [f"{tag}_{_prompt_hash(p)}" for p in prompts]

    cache_dir = CACHE_ROOT / _sanitize(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with shelve.open(str(cache_dir / "answers")) as cache:  # ponytail: single-writer; use diskcache if jobs share a cache concurrently
        # A provider that refused the pinned temperature sampled at its own
        # default, so those answers are filed under t"default", never under a
        # temperature they did not run at. The marker carries that across runs;
        # without it every later run would look under the wrong key and pay again.
        if cache.get(marker):
            _note_refused(model_name, "temperature", _temp)
        refused = "temperature" in _refused.get(model_name, ())
        keys = keys_for("default" if refused else _temp)
        results = [cache.get(k) for k in keys]
        miss = [i for i, r in enumerate(results) if not has_text(r)]
        if not miss:
            print(f">>> Full cache hit for {model_name} ({len(prompts)} prompts) <<<")
            return results

        print(f">>> Cache: {len(prompts) - len(miss)} hits, {len(miss)} misses for {model_name} <<<")
        miss_prompts = [prompts[i] for i in miss]
        # generate() sends provider-prefixed names (rcp:, gemini:, openai:) to
        # their API and everything else to local vLLM. This used to special-case
        # rcp: through generate_answers_rcp_api, which the provider refactor
        # removed: rcp: answer models raised NameError, and gemini:/openai: ones
        # were handed to vLLM as if they were HF model ids.
        outs = generate(model_name, miss_prompts, max_new_tokens=args.max_new_tokens,
                        tp=args.tp, utilization=args.utilization, schema=None,
                        max_model_len=getattr(args, "max_model_len", 8192),
                        temperature=_temp, seed=_seed)
        if not refused and "temperature" in _refused.get(model_name, ()):
            cache[marker] = True
            keys = keys_for("default")

        # A failed (None) or empty answer is never cached: it would be served
        # to every later run as if it were the model's reply.
        empty = 0
        for i, out in zip(miss, outs):
            results[i] = out
            if has_text(out):
                cache[keys[i]] = out
            else:
                empty += 1
        if empty:
            print(f">>> WARNING: {empty} of {len(miss)} answers from {model_name} are missing "
                  f"or empty; not cached, and their pairs are not judged <<<")
    return results


def safe_int(v):
    """int(v) or None. Judge output fields arrive as whatever the model emitted."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def winner_to_moove_vote(winner, swap):
    """Judge winner -> MOOVE vote code ("1" / "2" / "12"), position un-swapped.

    The judge always names a position ("Model 1"), never a model, so the swap
    has to be undone before the verdict means anything. Matches the encoding in
    full_moove.jsonl, which moove_agreement.VOTE_MAP decodes.
    """
    actual = unswap_winner(winner, swap)
    return {"Model 1": "1", "Model 2": "2", "Equal": "12", "Tie": "12"}.get(actual)


def build_judge_messages(prompt, a1, a2, swap, rubric="both"):
    m1_ans = a2 if swap else a1
    m2_ans = a1 if swap else a2
    user_content = (
        f"{judge_prompt(rubric)}\n\n"
        f"--- MEDICAL PROMPT ---\n{prompt}\n\n"
        f"--- MODEL 1 ANSWER ---\n{m1_ans}\n\n"
        f"--- MODEL 2 ANSWER ---\n{m2_ans}\n\n"
    )
    return [{"role": "user", "content": user_content}]


def unswap_winner(winner, swap):
    if not swap or winner not in ("Model 1", "Model 2"):
        return winner
    return "Model 2" if winner == "Model 1" else "Model 1"


def unswap_scores(verdict, swap, rubric="both"):
    """Return (left_scores, right_scores) keyed by criterion name."""
    empty = {c: None for c in criteria_for(rubric)}
    if verdict is None or rubric == "winner":        # winner: no criteria at all
        return dict(empty), dict(empty)
    if rubric == "preference":
        L, R = {}, {}
        for c, w in verdict.pref.model_dump().items():
            w = unswap_winner(w, swap)              # reuse the existing bookkeeping
            L[c], R[c] = {"Model 1": ("YES", "NO"),
                          "Model 2": ("NO", "YES")}.get(w, ("NA", "NA"))
        return L, R
    s1 = verdict.m1.as_criteria_dict()
    s2 = verdict.m2.as_criteria_dict()
    return (s2, s1) if swap else (s1, s2)


if __name__ == "__main__":
    # ponytail: the only logic here that can silently corrupt results is the
    # swap bookkeeping. One check, no framework.
    assert unswap_winner("Model 1", True) == "Model 2"
    assert unswap_winner("Model 1", False) == "Model 1"
    assert unswap_winner("Equal", True) == "Equal"

    V = verdict_model("moove")
    v = V.model_validate({
        "m1": {c: 2 for c in MOOVE},
        "m2": {c: -1 for c in MOOVE},
        "winner": "Model 1",
    })
    L, R = unswap_scores(v, swap=True, rubric="moove")
    assert L["safety"] == -1 and R["safety"] == 2, (L, R)
    L, R = unswap_scores(v, swap=False, rubric="moove")
    assert L["safety"] == 2 and R["safety"] == -1
    assert extract_judge_data('noise {"m1": {"safety": 9}} noise', V) is None
    print("ok")