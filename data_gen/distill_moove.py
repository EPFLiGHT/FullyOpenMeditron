#!/usr/bin/env python3
#distill_moove.py

import argparse
import json
from datetime import datetime
import random
import re
import torch
from vllm import LLM, SamplingParams

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
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7, help="Slightly higher temperature for diversity")
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--reasoning", type=str, default="low")
    return p.parse_args()

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
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False, 
        stop_token_ids=[tokenizer.eos_token_id],
    )

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

    prompt_configs = []

    # Build the required number of generation prompts
    for _ in range(target_count):
        k = min(5, num_source)
        samples = random.sample(source_prompts, k)
        token_ids = build_prompt(tokenizer, system_message, samples)
        prompt_configs.append(token_ids)

    print(f"--- Starting vLLM Generation ---")
    
    formatted_prompts = prompt_configs
    print(formatted_prompts[0])
    outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params)

    print(f"--- Processing and Saving Results to: {args.output} ---")
    success_count = 0
    
    with open(args.output, "w", encoding="utf-8") as f:
        for out in outputs:
            generated_text = out.outputs[0].text.strip()
            
            # Isolate the final message if using multi-channel outputs
            final_text = generated_text
            if "<|channel|>final<|message|>" in final_text:
                final_text = final_text.split("<|channel|>final<|message|>")[-1]
            
            # Extract the new prompt using regex
            match = re.search(r"<question>(.*?)</question>", final_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                extracted_prompt = match.group(1).strip()
                
                if extracted_prompt:
                    synthetic_row = {
                        "conversations": [
                            {"from": "human", "value": extracted_prompt},
                            {"from": "assistant", "value": ""}
                        ]
                    }
                    f.write(json.dumps(synthetic_row, ensure_ascii=False) + "\n")
                    success_count += 1

    print(f"✅ Extraction complete! Saved {success_count}/{target_count} valid synthetic prompts.")

if __name__ == "__main__":
    main()
