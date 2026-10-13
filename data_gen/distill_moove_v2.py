#!/usr/bin/env python3
#distill_moove_v2.py

import argparse
import hashlib
import json
import os
from datetime import datetime
import random
import re
import torch
from vllm import LLM
from distill_common import add_length_args, generate_complete

SYSTEM_MESSAGE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

DEV_MESSAGE = (
    "You are an expert medical educator and physician "
    "tasked with creating high-quality, clinically accurate content. "
    "Your task is to generate a new, unique, and realistic medical scenario or question prompt. "
    "The content must reflect realistic clinical presentations, inquiries from colleagues, "
    "or patient encounters. The timeline and objective progress should always be clear and detailed."
    "Include clear context about site and where people travelled etc."
    "You will be provided with 5 examples. Use them strictly to understand the desired "
    "format, diagnostic difficulty, and clinical depth. DO NOT copy them. "
    "Generate a completely new question that would be unconditionally "
    "approved by a medical review board."
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input .jsonl dataset (DPO format)")
    p.add_argument("--output", required=True, help="Path to output .jsonl file for synthetic prompts")
    p.add_argument("--model", required=True, help="HF model path or name")
    p.add_argument("--ratio", type=float, default=1.0, help="Ratio of synthetic prompts to generate based on input size")
    p.add_argument("--temperature", type=float, default=0.7, help="Slightly higher temperature for diversity")
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--reasoning", type=str, default="low")
    # questions are written in chunks and appended as they land, so a walltime
    # kill (debug is capped at 1h30) or a preemption loses one chunk, not the run
    p.add_argument("--chunk-size", type=int, default=2000,
                   help="prompts per vLLM call; the output is appended after each")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the few-shot draws (with the resume point, so a "
                        "resumed run is reproducible too)")
    p.add_argument("--resume", action="store_true",
                   help="keep the questions already in --output and generate the rest")
    add_length_args(p)
    return p.parse_args()

def question_id(text):
    """Stable id of a seed question: sha1 of its text, case/space/punctuation-free.

    Stored on every generated row as few_shot_ids, so which questions a
    generation was shown is a lookup, not a guess. v3 recorded nothing, and when
    its pool turned out to hold near-copies of eval questions there was no way
    to tell which generations had seen one.
    """
    norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def build_prompt(tokenizer, system_msg, examples):
    user_content = "Here are example prompts to model your format and clinical depth on:\n\n"
    for i, prompt_text in enumerate(examples):
        user_content += f"--- Example {i+1} ---\n<question>\n{prompt_text}\n</question>\n\n"
    
    user_content += (
        "Now, acting as an expert medical educator, generate a brand new, unique, "
        "and clinically accurate medical scenario or question. "
        "Wrap your generated scenario strictly within <question> and </question> tags."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "developer", "content": DEV_MESSAGE},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    args = parse_args()
    
    num_gpus = torch.cuda.device_count()
    tp_size = args.tp if args.tp > 0 else num_gpus

    print(f"--- Initializing vLLM ({tp_size} GPUs) ---")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.utilization,
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()
    
    print(f"--- Loading Source Dataset: {args.input} ---")
    system_message = SYSTEM_MESSAGE.format(reasoning=args.reasoning, date=datetime.now().strftime("%Y-%m-%d"))

    # Bypass HF datasets to avoid strict schema matching errors
    source_prompts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Check if 'question' exists, is a string, and isn't empty
                if "question" in row and isinstance(row["question"], str) and row["question"].strip():
                    source_prompts.append(row["question"].strip())
            except json.JSONDecodeError:
                continue

    num_source = len(source_prompts)
    
    if num_source == 0:
        raise ValueError("Source dataset must contain at least 1 valid 'question' field.")

    target_count = int(num_source * args.ratio)
    print(f"Extracted {num_source} source prompts.")
    print(f"--- Generating {target_count} synthetic prompts ---")

    success_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            success_count = sum(1 for line in f if line.strip())
        print(f"--- Resuming: {success_count} questions already in {args.output} ---")
    elif os.path.exists(args.output):
        open(args.output, "w").close()          # fresh run: start from empty

    print(f"--- Starting vLLM Generation ({args.chunk_size} prompts per call) ---")
    attempted, empty_streak = 0, 0
    rng = random.Random(f"{args.seed}:{success_count}")   # reproducible from any resume point
    while success_count < target_count:
        want = min(args.chunk_size, target_count - success_count)
        prompts, shown = [], []
        for _ in range(want):
            samples = rng.sample(source_prompts, min(5, num_source))
            prompts.append(build_prompt(tokenizer, system_message, samples))
            shown.append([question_id(q) for q in samples])
        if attempted == 0:
            print(prompts[0])

        outputs = generate_complete(llm, prompts, temperature=args.temperature,
                                    max_new_tokens=args.max_new_tokens,
                                    stop_token_ids=[tokenizer.eos_token_id],
                                    length_retries=args.length_retries, grow=args.grow)
        attempted += want

        kept = 0
        with open(args.output, "a", encoding="utf-8") as f:   # append: a kill loses
            for (final_text, _), ids in zip(outputs, shown):  # at most this chunk
                if final_text is None:      # never finished within the token budget
                    continue
                match = re.search(r"<question>(.*?)</question>", final_text,
                                  re.DOTALL | re.IGNORECASE)
                if not match:
                    continue
                extracted_prompt = match.group(1).strip()
                if not extracted_prompt:
                    continue
                f.write(json.dumps({"conversations": [
                    {"from": "human", "value": extracted_prompt},
                    {"from": "assistant", "value": ""}],
                    "few_shot_ids": ids}, ensure_ascii=False) + "\n")
                kept += 1
        success_count += kept
        print(f"--- {success_count}/{target_count} questions written "
              f"({kept}/{want} kept this chunk) -> {args.output} ---", flush=True)
        # Stop on a real failure, not on one unlucky tail chunk: a small chunk
        # can legitimately yield nothing when a single generation is unparsable.
        empty_streak = empty_streak + 1 if kept == 0 else 0
        if empty_streak >= 3:
            print("--- 3 chunks in a row produced nothing usable; stopping ---")
            break
        if attempted > 3 * target_count:
            print(f"--- attempted {attempted} prompts for {target_count} questions; "
                  f"stopping rather than looping ---")
            break

    print(f"✅ Extraction complete! {success_count}/{target_count} synthetic prompts "
          f"in {args.output}.")

if __name__ == "__main__":
    main()