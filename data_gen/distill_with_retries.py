#!/usr/bin/env python3
#distill_with_retries.py

import argparse
import json
from datetime import datetime
import torch
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

SYSTEM_MESSAGE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input .jsonl file")
    p.add_argument("--output", required=True, help="Path to output .jsonl file")
    p.add_argument("--model", required=True, help="HF model path or name")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tries", type=int, default=8)
    p.add_argument("--reasoning", type=str, default="low")
    p.add_argument("--resume", action="store_true", help="Resume from existing output file")
    return p.parse_args()

def get_indices(conversations):
    user_idx, assistant_idx = None, None
    for i, turn in enumerate(conversations):
        role = turn.get("from")
        if user_idx is None and role in ["human", "user"]:
            user_idx = i
        elif user_idx is not None and role == "assistant":
            assistant_idx = i
            break
    return user_idx, assistant_idx

MMLU_PRO_PATTERN = re.compile(r'answer is \(?([A-J])\)?', re.IGNORECASE)
MEDMCQA_PATTERN  = re.compile(r'(?i)Answer[^A-J]*(?::)*[^A-J]*(?:boxed)*[^A-J]*([A-J])\b')

def evaluate_exact_match(text, ground_truth_label):
    if not ground_truth_label:
        return None, None, None

    m = MMLU_PRO_PATTERN.findall(text)
    if m:
        found = m[-1].upper()
        return (found == ground_truth_label.upper()), found, "mmlu_pro"

    m = MEDMCQA_PATTERN.findall(text)
    if m:
        found = m[-1].upper()
        return (found == ground_truth_label.upper()), found, "medmcqa"

    return False, "INVALID", None

def main():
    args = parse_args()
    gptoss = args.model == "openai/gpt-oss-120b"
    medgemma = args.model == "google/medgemma-27b-text-it"
    apertus = "Apertus" in args.model 
    
    num_gpus = torch.cuda.device_count()
    tp_size = args.tp if args.tp > 0 else num_gpus
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.utilization,
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()

    stop_tokens = [tokenizer.eos_token_id]
    if "<end_of_turn>" in tokenizer.vocab and medgemma:
        stop_tokens.append(tokenizer.vocab["<end_of_turn>"])  
    if "<|assistant_end|>" in tokenizer.vocab and apertus:
        stop_tokens.append(tokenizer.vocab["<|assistant_end|>"])
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False, 
        stop_token_ids=stop_tokens,
    )

    print(f"--- Loading Dataset: {args.input} ---")
    ds = load_dataset("json", data_files=args.input, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows = [dict(x) for x in ds]
    
    # --- ADDED: Clean up conversations and set default tracking vars ---
    for row in rows:
        if "conversations" in row:
            row["conversations"] = [turn for turn in row["conversations"] if turn.get("from") != "system"]
            
        if not row.get("exact_match"):
            row["exact_match"] = False
        if not row.get("try_count"):
            row["try_count"] = 0
    # ------------------------------------------------------------------

    pattern_counts = {"mmlu_pro": 0, "medmcqa": 0, "none": 0}
    for attempt in range(1, args.max_tries + 1):
        pending_indices = []
        prompt_token_ids = []

        for i, row in enumerate(rows):
            if (attempt == 1 and not args.resume) or (not row.get("exact_match") and row.get("label_letter")): #if 1st attempt and false or if labeled and false
                uidx, aidx = get_indices(row.get("conversations", []))
                if uidx is not None:
                    user_text = row["conversations"][uidx].get("value", "").strip()
                    messages = []
                    if gptoss:
                        system_message = SYSTEM_MESSAGE.format(reasoning=args.reasoning, date=datetime.now().strftime("%Y-%m-%d"))
                        messages.append({"role": "system", "content": system_message})
                    messages.append({"role": "user", "content": user_text})
                    token_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt_token_ids.append(token_ids)
                    pending_indices.append((i, aidx))

        if not pending_indices:
            print(f"--- All items matched or no labels found. Ending early at attempt {attempt} ---")
            break

        print(f"--- Attempt {attempt}/{args.max_tries}: Processing {len(pending_indices)} samples ---")
        
        print(prompt_token_ids[0])
        outputs = llm.generate(prompts=prompt_token_ids, sampling_params=sampling_params)

        for (row_idx, assistant_idx), out in zip(pending_indices, outputs):
            res_text = out.outputs[0].text.strip()
            
            # --- ADDED: Extract only the final channel message ---
            if "<|channel|>final<|message|>" in res_text:
                res_text = res_text.split("<|channel|>final<|message|>")[-1].strip()
            # -----------------------------------------------------
                
            label = rows[row_idx].get("label_letter")
            
            is_match, found_label, pattern_used = evaluate_exact_match(res_text, label)
            if is_match:
                pattern_counts[pattern_used or "none"] += 1

            
            rows[row_idx]["conversations"][assistant_idx]["value"] = res_text
            rows[row_idx]["exact_match"] = is_match
            rows[row_idx]["try_count"] = attempt

        total_labeled = sum(1 for r in rows if r.get("label_letter"))
        total_correct = sum(1 for r in rows if r.get("exact_match") is True)
    

        if total_labeled > 0:
            accuracy_pct = (total_correct / total_labeled) * 100
            if attempt < args.max_tries:
                print(f"\nAccuracy after Attempt {attempt}: {accuracy_pct:.2f}% ({total_correct}/{total_labeled})")
            else:
                print(f"\n✅ Final Accuracy after {args.max_tries} tries: {accuracy_pct:.2f}% ({total_correct}/{total_labeled})")
        else:
            status = "Attempt" if attempt < args.max_tries else "Final result"
            print(f"\n✅ {status} {attempt}/{args.max_tries} complete. (No labels available to calculate accuracy)")
        print(f"--- Saving Results to: {args.output} ---")
        print(f"\n--- Pattern match counts across all attempts ---")
        print(f"  mmlu_pro: {pattern_counts['mmlu_pro']}")
        print(f"  medmcqa:  {pattern_counts['medmcqa']}")
        print(f"  none/invalid: {pattern_counts['none']}")
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
