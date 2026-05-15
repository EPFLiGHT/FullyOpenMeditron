#!/usr/bin/env python3
# healthbench/healthbench.py
"""
This script evaluates the performance of a model on the HealthBench dataset.

To run HealthBench, HealthBench Consensus, or HealthBench Hard, use the simple-evals script:
- `python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_consensus --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_hard --model=gpt-4.1`

You can also evaluate physician ideal completions or reference completions against the HealthBench rubrics. To do so, run the following command:
- To evaluate physician ideal completions: `python -m simple-evals.healthbench_eval --run_mode=physician_completions`
- To evaluate reference model completions used by physicians: `python -m simple-evals.healthbench_eval --run_mode=physician_completion_references`
"""

import argparse
import gc
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import urllib.request
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

@dataclass
class SingleEvalResult:
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str = ""
    convo: list[dict] = field(default_factory=list)
    example_level_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    score: float | None
    metrics: dict[str, float]
    htmls: list[str] = field(default_factory=list)
    convos: list[list[dict]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


class RubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(
            rubric_items, grading_response_list, strict=True
        )
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


def calculate_length_adjusted_score(
    score: float,
    response_text: str,
    *,
    center: float,
    penalty_per_500_chars: float,
) -> float:
    return score - penalty_per_500_chars * ((len(response_text) - center) / 500.0)


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": response_usage.input_tokens_details.cached_tokens
            if hasattr(response_usage.input_tokens_details, "cached_tokens")
            else response_usage.input_tokens_details["cached_tokens"],
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": response_usage.output_tokens_details.reasoning_tokens
            if hasattr(response_usage.output_tokens_details, "reasoning_tokens")
            else response_usage.output_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": response_usage.prompt_tokens_details.cached_tokens
            if hasattr(response_usage.prompt_tokens_details, "cached_tokens")
            else response_usage.prompt_tokens_details["cached_tokens"],
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": response_usage.completion_tokens_details.reasoning_tokens
            if hasattr(response_usage.completion_tokens_details, "reasoning_tokens")
            else response_usage.completion_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }


PHYSICIAN_COMPLETION_MODES = {
    "Group 1": {
        "description": "No reference completions were provided to the physicians.",
        "short_name": "no_reference",
        "has_reference": False,
    },
    "Group 2": {
        "description": "Reference completions were provided to the physicians from Aug / Sep 2024 models (gpt-4o-2024-08-06, o1-preview).",
        "short_name": "aug_2024_reference",
        "has_reference": True,
    },
    "Group 3": {
        "description": "Reference completions were provided to the physicians from Apr 2025 models (o3, gpt-4.1).",
        "short_name": "apr_2025_reference",
        "has_reference": True,
    },
}


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples for final HealthBench scoring."""
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            _compute_clipped_stats(list(s), "mean") for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f"Unknown {stat =}")


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:
    """
    Aggregate multiple SingleEvalResults into a single EvalResult for HealthBench.
    For each metric, returns the stats in _compute_clipped_stats.
    """
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )



def free_vram():
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def vllm_batch_generate(
    model_name: str,
    prompts: list,  # list of message-lists OR list of strings
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int = 0,
    gpu_memory_utilization: float = 0.90,
) -> list[str]:
    """Load model, batch-generate, free VRAM. Returns list of response texts."""
    tp = tensor_parallel_size if tensor_parallel_size > 0 else torch.cuda.device_count()
    
    print(f"\n>>> Loading {model_name} (tp={tp}) <<<")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Format prompts via chat template
    formatted = []
    for p in prompts:
        if isinstance(p, list):
            text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        else:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    
    print(f">>> Generating {len(formatted)} completions <<<")
    outputs = llm.generate(prompts=formatted, sampling_params=sampling_params)
    results = [out.outputs[0].text.strip() for out in outputs]
    
    del llm
    free_vram()
    return results

class HealthBenchVLLMEval:
    def __init__(
        self,
        model_name: str,
        grader_name: str,
        num_examples: int | None = None,
        n_repeats: int = 1,
        subset_name: Literal["hard", "consensus"] | None = None,
        input_path: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        grader_max_tokens: int = 2048,
        grader_temperature: float = 0.1,
        grader_max_retries: int = 3,
        tensor_parallel_size: int = 0,
        gpu_memory_utilization: float = 0.90,
    ):
        self.model_name = model_name
        self.grader_name = grader_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.grader_max_tokens = grader_max_tokens
        self.grader_temperature = grader_temperature
        self.grader_max_retries = grader_max_retries
        self.tp = tensor_parallel_size
        self.util = gpu_memory_utilization
        
        # Resolve input path
        if input_path is None:
            if subset_name == "hard":
                input_path = INPUT_PATH_HARD
            elif subset_name == "consensus":
                input_path = INPUT_PATH_CONSENSUS
            else:
                input_path = INPUT_PATH
        
        if input_path.startswith(("http://", "https://")):
            with urllib.request.urlopen(input_path) as f:
                content = f.read().decode("utf-8")
            examples = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                examples = [json.loads(line) for line in f if line.strip()]
        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]
        
        rng = random.Random(0)
        if num_examples is not None and num_examples < len(examples):
            examples = rng.sample(examples, num_examples)
        
        self.examples = examples * n_repeats
    
    def run(self) -> EvalResult:
        # === PHASE 1: Generate model-under-test completions ===
        prompts = [row["prompt"] for row in self.examples]
        completions = vllm_batch_generate(
            self.model_name,
            prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=self.util,
        )
        
        # Build the (example_idx, rubric_idx, grader_prompt) flat list
        grader_prompts = []
        prompt_index = []  # (example_idx, rubric_idx)
        for ex_idx, (row, completion) in enumerate(zip(self.examples, completions)):
            convo_with_response = row["prompt"] + [
                {"role": "assistant", "content": completion}
            ]
            convo_str = "\n\n".join(
                f"{m['role']}: {m['content']}" for m in convo_with_response
            )
            for r_idx, rubric_item in enumerate(row["rubrics"]):
                grader_prompt_text = (
                    GRADER_TEMPLATE
                    .replace("<<conversation>>", convo_str)
                    .replace("<<rubric_item>>", str(rubric_item))
                )
                grader_prompts.append([{"role": "user", "content": grader_prompt_text}])
                prompt_index.append((ex_idx, r_idx))
        
        print(f"\n>>> Total grader calls needed: {len(grader_prompts)} <<<")
        
        # === PHASE 2: Run grader with retries on parse failures ===
        grading_texts = [None] * len(grader_prompts)
        pending = list(range(len(grader_prompts)))
        
        # We load the grader ONCE and reuse for retries by keeping the LLM alive
        tp = self.tp if self.tp > 0 else torch.cuda.device_count()
        print(f"\n>>> Loading grader {self.grader_name} (tp={tp}) <<<")
        grader_llm = LLM(
            model=self.grader_name,
            tensor_parallel_size=tp,
            gpu_memory_utilization=self.util,
            trust_remote_code=True,
        )
        grader_tok = grader_llm.get_tokenizer()
        
        parsed_results = [None] * len(grader_prompts)
        
        for attempt in range(self.grader_max_retries + 1):
            if not pending:
                break
            if attempt > 0:
                print(f">>> Grader retry {attempt}/{self.grader_max_retries} on {len(pending)} prompts <<<")
            
            current_temp = min(1.0, self.grader_temperature + attempt * 0.2)
            sp = SamplingParams(temperature=current_temp, max_tokens=self.grader_max_tokens)
            
            formatted = [
                grader_tok.apply_chat_template(grader_prompts[i], tokenize=False, add_generation_prompt=True)
                for i in pending
            ]
            outputs = grader_llm.generate(prompts=formatted, sampling_params=sp)
            
            new_pending = []
            for i, out in zip(pending, outputs):
                text = out.outputs[0].text.strip()
                grading_texts[i] = text
                parsed = parse_json_to_dict(text)
                if "criteria_met" in parsed and isinstance(parsed["criteria_met"], bool):
                    parsed_results[i] = parsed
                else:
                    new_pending.append(i)
            pending = new_pending
        
        # Anything still unparsed: fill with criteria_met=False so scoring proceeds
        for i in pending:
            parsed_results[i] = {"criteria_met": False, "explanation": "Failed to parse after retries"}
        
        del grader_llm
        free_vram()
        
        # === PHASE 3: Score and aggregate ===
        # Regroup parsed_results by example
        per_example_grades: dict[int, list] = defaultdict(lambda: [])
        for (ex_idx, r_idx), parsed in zip(prompt_index, parsed_results):
            # ensure ordering matches rubric order
            per_example_grades[ex_idx].append((r_idx, parsed))
        
        single_results = []
        for ex_idx, row in enumerate(self.examples):
            grades_with_idx = sorted(per_example_grades[ex_idx], key=lambda x: x[0])
            grading_response_list = [g for _, g in grades_with_idx]
            rubric_items = row["rubrics"]
            
            overall_score = calculate_score(rubric_items, grading_response_list)
            assert overall_score is not None
            metrics = {"overall_score": overall_score}
            
            # example-level tags
            for tag in row["example_tags"]:
                metrics[tag] = overall_score
            
            # rubric-level tags
            rubric_tag_items_grades = defaultdict(list)
            for ri, gr in zip(rubric_items, grading_response_list):
                for tag in ri.tags:
                    rubric_tag_items_grades[tag].append((ri, gr))
            for tag, items_grades in rubric_tag_items_grades.items():
                items, grades = zip(*items_grades)
                s = calculate_score(items, grades)
                if s is not None:
                    metrics[tag] = s
            
            # build readable explanation
            rubric_items_with_grades = []
            readable = []
            for ri, gr in zip(rubric_items, grading_response_list):
                expl = gr.get("explanation", "No explanation provided")
                cm = gr["criteria_met"]
                readable.append(f"[{cm}] {ri}\n\tExplanation: {expl}")
                rubric_items_with_grades.append({**ri.to_dict(), "criteria_met": cm, "explanation": expl})
            readable.sort(key=lambda x: x.startswith("[False]"), reverse=True)
            readable_str = "\n\n" + "\n\n".join(readable)
            
            convo = row["prompt"] + [{"role": "assistant", "content": completions[ex_idx]}]
            html = ""
            
            single_results.append(SingleEvalResult(
                html=html,
                score=overall_score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": overall_score,
                    "rubric_items": rubric_items_with_grades,
                    "readable_explanations": readable_str,
                    "prompt": row["prompt"],
                    "completion": [{"role": "assistant", "content": completions[ex_idx]}],
                    "prompt_id": row["prompt_id"],
                    "completion_id": hashlib.sha256(
                        (row["prompt_id"] + completions[ex_idx]).encode("utf-8")
                    ).hexdigest(),
                },
            ))
        
        return _aggregate_get_clipped_mean(single_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF path/name for model under test")
    parser.add_argument("--grader", required=True, help="HF path/name for grader")
    parser.add_argument("--subset", choices=["hard", "consensus"], default=None)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--grader-max-tokens", type=int, default=2048)
    parser.add_argument("--grader-temperature", type=float, default=0.1)
    parser.add_argument("--grader-max-retries", type=int, default=3)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--utilization", type=float, default=0.90)
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to results_<model_name>.json")
    parser.add_argument("--output-dir", default=".", help="Directory for default output file")
    parser.add_argument("--summary-file", default=None, help="Append a one-line CSV summary here. Defaults to <output-dir>/healthbench_summary.csv")    
    args = parser.parse_args()

    if args.output is None:
        model_name = Path(args.model).name  # last path component, works for both HF IDs and local paths
        subset_tag = f"_{args.subset}" if args.subset else ""
        args.output = str(Path(args.output_dir) / f"results_{model_name}{subset_tag}.json")
        print(f"Output path: {args.output}")
    
    eval_obj = HealthBenchVLLMEval(
        model_name=args.model,
        grader_name=args.grader,
        num_examples=args.examples,
        n_repeats=args.n_repeats,
        subset_name=args.subset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        grader_max_tokens=args.grader_max_tokens,
        grader_temperature=args.grader_temperature,
        grader_max_retries=args.grader_max_retries,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.utilization,
    )
    result = eval_obj.run()
    
    Path(args.output).write_text(json.dumps({
        "score": result.score,
        "metrics": {k: float(v) for k, v in result.metrics.items()},
        "metadata": result.metadata,
    }, indent=2, default=str))
    if args.summary_file is None:
        args.summary_file = str(Path(args.output_dir) / "healthbench_summary.csv")

    subset_tag = args.subset if args.subset else "none"
    model_name = Path(args.model).name
    summary_line = f"healthbench,{subset_tag},{model_name},{result.score}\n"

    with open(args.summary_file, "a", encoding="utf-8") as f:
        f.write(summary_line)
    print(f"Appended summary to {args.summary_file}: {summary_line.strip()}")
    print(f"\nFinal score: {result.score}")
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()

