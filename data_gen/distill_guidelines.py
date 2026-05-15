#!/usr/bin/env python3
#distill_guidelines.py

import argparse
import json
from datetime import datetime
import re
import torch
from collections import Counter
from vllm import LLM, SamplingParams

MODEL_CTX = 131072
RESERVED_FOR_OUTPUT = 8192
TEMPLATE_OVERHEAD_BUFFER = 512   # rough headroom for chat template tokens
MAX_PROMPT_TOKENS = MODEL_CTX - RESERVED_FOR_OUTPUT - TEMPLATE_OVERHEAD_BUFFER


SYSTEM_MESSAGE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

DEV_MESSAGE = (
    "You are an expert medical educator and physician "
    "tasked with creating high-quality, clinically accurate content for a medical exam. "
    "Your task is to generate clinical vignette-style questions along with its correct answer. "
    "based STRICTLY on the provided medical guideline. "
    "Focus on realistic patient presentations (age, symptoms, physical exam findings), "
    "identifying 'red flags', and diagnostic reasoning highlighted in the text. "
    "The timeline and objective progress should always be clear and detailed in the vignettes."
    "Include clear context about site and where people travelled etc."
    "Do not include outside information or unproven treatments. "
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input .jsonl dataset containing guidelines")
    p.add_argument("--output", required=True, help="Path to output .jsonl file for synthetic QA data")
    p.add_argument("--model", required=True, help="HF model path or name")
    p.add_argument("--max-new-tokens", type=int, default=8192, help="High limit to fit 10 vignettes")
    p.add_argument("--temperature", type=float, default=0.6, help="Lower temp to keep it grounded in the text")
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--reasoning", type=str, default="low")
    p.add_argument("--limit", type=int, default=None)

    return p.parse_args()

def extract_chosen_answer(text):
    """
    Extracts the chosen answer by looking for 'Answer: X', '**X.**', 
    or just 'X.' at the beginning of the text.
    """

    # 1. Look for explicit "Answer: [A-E]" or similar
    pattern_answer = r'answer is \(?([A-J])\)?'
    matches = re.findall(pattern_answer, text)
    if matches:
        return matches[-1].upper()

    pattern_answer = r"(?i)Answer[^A-E]*(?::)*[^A-E]*(?:boxed)*[^A-E]*([A-E])\b"
    matches = re.findall(pattern_answer, text)
    if matches:
        return matches[-1].upper()
    
    # 2. Look for "**A.**", "**B -", etc.
    pattern_bold = r"\*\*\s*([A-E])\s*[\.\-\–]"
    matches = re.findall(pattern_bold, text)
    if matches:
        return matches[0].upper()
    
    # 3. Look for "Option [1-5]" and map it to A-E
    pattern_option = r"(?i)Option\s*([1-5])"
    matches = re.findall(pattern_option, text)
    if matches:
        # Map 1->A, 2->B, 3->C, 4->D, 5->E
        return chr(int(matches[0]) + 64) 

    return "Unknown"

def build_prompt(tokenizer, system_message, guideline_text):
    user_content = (
        f"Here is the medical guideline:\n\n"
        f"=== GUIDELINE START ===\n{guideline_text}\n=== GUIDELINE END ===\n\n"
        "Based ONLY on the guideline above, generate exactly 10 unique MULTIPLE-CHOICE clinical vignette questions and their answers. "
        "Each question should present a realistic patient scenario that tests the diagnostic or management principles in the text. "
        "For each vignette, provide 4-5 plausible multiple-choice options (A-E). "
        "Ensure distractors represent common diagnostic pitfalls or 'next best steps' "
        "that are incorrect based strictly on the provided guideline.\n\n"
        "You MUST format EACH of the 10 items exactly as follows, using these specific XML tags:\n\n"
        "<qa>\n"
        "<question>\n"
        "Patient scenario and the specific question here.\n"
        "A) [Option 1]\n"
        "B) [Option 2]\n"
        "C) [Option 3]\n"
        "D) [Option 4]\n"
        "</question>\n"
        "<answer>The rationale explaining your chain of thought without mentioning the guideline and then Answer: correct answer</answer>\n"
        "</qa>"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "developer", "content": DEV_MESSAGE},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    args = parse_args()
    
    num_gpus = torch.cuda.device_count()
    tp_size = args.tp if args.tp > 0 else num_gpus

    print(f"--- Initializing vLLM (Model: {args.model}, TP: {tp_size}) ---")
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

    system_message = SYSTEM_MESSAGE.format(reasoning=args.reasoning, date=datetime.now().strftime("%Y-%m-%d"))

    print(f"--- Loading Source Guidelines from: {args.input} ---")
    prompt_configs = []
    skipped_guideline = 0
    
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            # Adjust the key "text" if your JSONL uses a different key for the guideline body
            guideline_text = data.get("clean_text", "").strip()
            
            if guideline_text:
                token_ids = build_prompt(tokenizer, system_message, guideline_text)
                
                ids = tokenizer.encode(token_ids, add_special_tokens=False)
                if len(ids) > MAX_PROMPT_TOKENS:
                    print(f"[skip] guideline {i}: {len(ids)} tokens > budget {MAX_PROMPT_TOKENS}")
                    skipped_guideline += 1
                    print(f"{skipped_guideline} skipped guidelines")
                    continue
                prompt_configs.append(token_ids)

    if args.limit:
        guideline_count = args.limit
        prompt_configs = prompt_configs[:args.limit]

    print(f"Loaded {len(prompt_configs)} guidelines, skipped {skipped_guideline}. "
      f"Generating ~{len(prompt_configs) * 10} synthetic QAs...")

    print(f"--- Starting vLLM Generation ---")
    formatted_prompts = prompt_configs
    
    print(formatted_prompts[0])
    outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params)

    print(f"--- Processing and Saving Results to: {args.output} ---")
    total_extracted_qas = 0
    label_distribution = Counter()
    
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, out in enumerate(outputs):
            generated_text = out.outputs[0].text.strip()
            
            # Find all instances of <qa> blocks in the output
            # re.DOTALL allows the dot (.) to match newlines
            qa_matches = re.findall(
                r"<qa>\s*<question>(.*?)</question>\s*<answer>(.*?)</answer>\s*</qa>", 
                generated_text, 
                re.DOTALL | re.IGNORECASE
            )
            
            if not qa_matches:
                print(f"Warning: No valid <qa> blocks found in output for guideline index {i}.")
                continue
                
            for q_text, a_text in qa_matches:
                q_text = q_text.strip()
                a_text = a_text.strip()
                
                if q_text and a_text:
                    choice = extract_chosen_answer(a_text)
                    label_distribution[choice] += 1
                    synthetic_row = {
                        "label_letter": choice,
                        "conversations": [
                            {"from": "human", "value": q_text},
                            {"from": "assistant", "value": a_text}
                        ],
                        "parse_failed": False,
                        "raw_output": generated_text
                    }
                    out_f.write(json.dumps(synthetic_row, ensure_ascii=False) + "\n")
                    total_extracted_qas += 1

    print(f"✅ Extraction complete! Saved {total_extracted_qas}/{int(10*len(outputs))} valid synthetic QA pairs to {args.output}.")
    
    print("\n--- Label Distribution (Assistant Choices) ---")
    total_extracted = sum(label_distribution.values())
    for label in sorted(label_distribution.keys()):
        count = label_distribution[label]
        percentage = (count / total_extracted) * 100 if total_extracted > 0 else 0
        print(f"Option {label}: {count} ({percentage:.2f}%)")
    print(f"Total processed labels: {total_extracted}")

if __name__ == "__main__":
    main()
