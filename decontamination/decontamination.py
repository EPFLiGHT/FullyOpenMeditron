#!/usr/bin/env python3
"""
decontamination.py
──────────────────
Original Source: https://github.com/swiss-ai/posttraining-data/blob/main/04-decontamination/decontamination.py
Project: Swiss AI - Post-Training Data Pipeline

This script is based on the decontamination tool originally developed by the Swiss AI project.
──────────────────
Dataset decontamination tool for removing evaluation benchmark contamination.

This script identifies and removes training samples that overlap with evaluation
benchmarks to prevent data leakage. Uses n-gram overlap detection and exact
matching to find contaminated samples across multiple benchmark datasets.

Key features:
- N-gram overlap detection with configurable thresholds
- Support for multiple benchmark formats
- Memory-efficient processing for large datasets
- Detailed contamination reporting and statistics
- Preserves dataset metadata and processing history
"""
import argparse
import difflib
from datasets import load_from_disk, load_dataset, DatasetDict
import shutil
from transformers import AutoTokenizer
from multiprocessing import Pool
from collections import defaultdict
from functools import partial
import json
import os
import time
import gc
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

"""
Code modified from: https://github.com/huggingface/cosmopedia/blob/main/decontamination/decontaminate.py
Enhanced with pre-filtering and parallelization optimizations.
V2: Optimizes n-gram matching by pre-filtering training samples that have no n-gram overlap.
V3: Parallelizes Step 3 (contamination verification) using shared memory to avoid serialization overhead.
"""


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(train_data, output_path: Path, input_path: Path,
                              contaminated_ids: set, processed_benchmarks: list,
                              tokenizer_name: str, ngram_length: int, diff_threshold: float):
    """Save filtered dataset and update metadata with processing log."""
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    print(f"Saving filtered dataset to {output_path}...")
    train_data.save_to_disk(str(output_path))

    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}

    # Calculate samples removed per split if DatasetDict
    samples_by_split = {}
    total_samples_before = 0
    total_samples_after = 0

    if hasattr(train_data, 'keys'):  # DatasetDict
        for split_name, split_data in train_data.items():
            samples_by_split[split_name] = len(split_data)
            total_samples_after += len(split_data)
    else:  # Single Dataset
        total_samples_after = len(train_data)

    # Add processing log entry
    processing_entry = {
        "operation": "decontamination",
        "script": "decontamination.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokenizer_name": tokenizer_name,
        "ngram_length": ngram_length,
        "diff_threshold": diff_threshold,
        "benchmarks_processed": len(processed_benchmarks),
        "benchmark_names": processed_benchmarks,
        "contaminated_samples_removed": len(contaminated_ids),
        "samples_after_filtering": total_samples_after,
        "decontamination_success": True
    }

    # Add split information if DatasetDict
    if samples_by_split:
        processing_entry["samples_by_split"] = samples_by_split

    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return set(zip(*[tokens[i:-(n - i)] for i in range(n)]))


def process_tokens(x, ngram_length):
    return {
        "ngram": get_ngrams(x, ngram_length),
        "tokens": x
    }


shared_eval_ngram_to_eval_idx = None
shared_eval_ngram_to_eval_idx_key_set = None


def init_get_eval_match_indices(ngram_to_eval_idx, ):
    global shared_eval_ngram_to_eval_idx, shared_eval_ngram_to_eval_idx_key_set
    shared_eval_ngram_to_eval_idx = ngram_to_eval_idx
    shared_eval_ngram_to_eval_idx_key_set = set(ngram_to_eval_idx.keys())


def get_eval_match_indices(train_sample):
    global shared_eval_ngram_to_eval_idx, shared_eval_ngram_to_eval_idx_key_arr
    shared_ngrams = train_sample["ngram"] & shared_eval_ngram_to_eval_idx_key_set
    if len(shared_ngrams) == 0:
        return tuple([])
    matches = []
    for ngram in shared_ngrams:
        matches.extend(shared_eval_ngram_to_eval_idx.get(ngram))
    return tuple(set(matches))


shared_train_ngrams = None
shared_eval_ngrams = None


def init_check_matching(train_ngrams, eval_ngrams):
    global shared_train_ngrams, shared_eval_ngrams
    shared_train_ngrams = train_ngrams
    shared_eval_ngrams = eval_ngrams


def check_matching(train_idx, eval_indices):
    global shared_train_ngrams, shared_eval_ngrams
    # train_idx, eval_indices = inputs
    eval_idx_match = None
    for eval_idx in eval_indices:
        matcher = difflib.SequenceMatcher(
            None,
            shared_train_ngrams[train_idx]["tokens"],
            shared_eval_ngrams[eval_idx],
            autojunk=False
        )
        matching_blocks = matcher.get_matching_blocks()
        match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
        del matcher, matching_blocks
        if match_length / len(shared_eval_ngrams[eval_idx]["tokens"]) >= args.diff_threshold:
            eval_idx_match = eval_idx
            break
    return train_idx, eval_idx_match


shared_eval_ngrams_dict = None


def init_contamination_worker(eval_ngrams_dict):
    """Initialize worker process with shared eval_ngrams dictionary."""
    global shared_eval_ngrams_dict
    shared_eval_ngrams_dict = eval_ngrams_dict


def verify_contamination_worker_optimized(args_tuple):
    """
    Optimized worker function for parallel contamination verification using shared memory.
    
    Args:
        args_tuple: (train_idx, eval_indices, train_tokens, threshold, train_conversation_id)
    
    Returns:
        tuple: (train_conversation_id, eval_idx, match_ratio, train_tokens_full, eval_tokens_full) if contaminated, None otherwise
    """
    global shared_eval_ngrams_dict
    train_idx, eval_indices, train_tokens, threshold, train_conversation_id = args_tuple

    for eval_idx in eval_indices:
        eval_tokens = shared_eval_ngrams_dict[eval_idx]["tokens"]
        matcher = difflib.SequenceMatcher(None, train_tokens, eval_tokens, autojunk=False)
        matching_blocks = matcher.get_matching_blocks()
        match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
        del matcher, matching_blocks

        match_ratio = match_length / len(eval_tokens)
        if match_ratio >= threshold:
            return (train_conversation_id, eval_idx, match_ratio, train_tokens, eval_tokens)

    return None


def verify_contamination_worker(args_tuple):
    """
    Worker function for parallel contamination verification.
    
    Args:
        args_tuple: (train_idx, eval_indices, train_tokens, eval_ngrams_dict, threshold, train_conversation_id)
    
    Returns:
        tuple: (train_conversation_id, eval_idx, match_ratio, train_tokens_full, eval_tokens_full) if contaminated, None otherwise
    """
    train_idx, eval_indices, train_tokens, eval_ngrams_dict, threshold, train_conversation_id = args_tuple

    for eval_idx in eval_indices:
        eval_tokens = eval_ngrams_dict[eval_idx]["tokens"]
        matcher = difflib.SequenceMatcher(None, train_tokens, eval_tokens, autojunk=False)
        matching_blocks = matcher.get_matching_blocks()
        match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
        del matcher, matching_blocks

        match_ratio = match_length / len(eval_tokens)
        if match_ratio >= threshold:
            return (train_conversation_id, eval_idx, match_ratio, train_tokens, eval_tokens)

    return None


def get_benchmark_cache_key(benchmark_name, tokenizer_name, ngram_length):
    """Generate cache key for benchmark n-grams."""
    key_string = f"{benchmark_name}_{tokenizer_name}_{ngram_length}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_benchmark_cache_path(cache_dir, cache_key):
    """Get cache file path for benchmark n-grams."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"benchmark_ngrams_{cache_key}.pkl")


def load_cached_benchmark_ngrams(cache_path):
    """Load cached benchmark n-grams if they exist."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
    return None


def save_benchmark_ngrams_to_cache(ngrams, lookup_table, cache_path):
    """Save benchmark n-grams to cache."""
    try:
        cache_data = {
            'ngrams': ngrams,
            'lookup_table': lookup_table
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Cached n-grams to {cache_path}")
    except (pickle.PickleError, IOError) as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")


def compute_benchmark_ngrams(eval_dataset, tokenizer, ngram_length, num_proc):
    """Compute n-grams for a benchmark dataset."""
    print(f"  Tokenizing {len(eval_dataset)} evaluation prompts...")
    eval_data_tokens = tokenizer(eval_dataset["prompt"][:])["input_ids"]
    print(f"  Generating n-grams...")
    with Pool(num_proc) as p:
        eval_ngrams = p.map(partial(process_tokens, ngram_length=ngram_length), eval_data_tokens)
    del eval_data_tokens

    print(f"  Building n-gram lookup table...")
    eval_ngram_to_eval_idx = defaultdict(list)
    append = eval_ngram_to_eval_idx.__getitem__
    for idx, s in enumerate(eval_ngrams):
        for element in s["ngram"]:
            append(element).append(idx)
    print(f"  Created lookup table with {len(eval_ngram_to_eval_idx)} unique n-grams")

    return eval_ngrams, eval_ngram_to_eval_idx


def main(args):
    print(f"\n=== DECONTAMINATION PIPELINE ===")

    # 1. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # 2. Load & Format Evaluation Data
    if args.eval_format == "raw_json" or args.decontamination_prompts.endswith(('.json', '.jsonl')):
        print(f"Loading raw JSON/JSONL evaluation data from: {args.decontamination_prompts}")
        raw_eval = load_dataset("json", data_files=args.decontamination_prompts, split="train")
        eval_data = DatasetDict({
            "custom_eval_benchmark": raw_eval.map(lambda x: {"prompt": x[args.eval_prompt_key]})
        })
        benchmark_list = ["custom_eval_benchmark"]
    else:
        print(f"Loading HF dataset evaluation data from: {args.decontamination_prompts}")
        eval_data = load_from_disk(args.decontamination_prompts)
        benchmark_list = args.benchmark_name if args.benchmark_name else list(eval_data.keys())
        # Map custom key if it's not "prompt"
        if args.eval_prompt_key != "prompt":
            for bench in benchmark_list:
                eval_data[bench] = eval_data[bench].map(lambda x: {"prompt": x[args.eval_prompt_key]})
                
    print(f"Total benchmarks to process: {len(benchmark_list)}\n")

    if not os.path.exists(args.report_path):
        print(f"Creating contamination reports directory: {args.report_path}")
        os.makedirs(args.report_path, exist_ok=True)

    # 3. Load & Format Training Data
    print(f"Loading training data from: {args.dataset_path}")
    if args.dataset_path.endswith(('.json', '.jsonl')):
        train_data = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        train_data = load_from_disk(args.dataset_path)

    # Flatten DatasetDict if necessary
    if hasattr(train_data, 'keys'):
        from datasets import concatenate_datasets
        train_data = concatenate_datasets([train_data[split] for split in train_data.keys()])

    # Standardize format and tokenize
    def format_and_tokenize(example, idx):
        # Generate a deterministic ID if it doesn't exist
        conv_id = str(example.get("conversation_id", f"auto_id_{idx}"))
        
        # Extract text based on the provided format flag
        text = ""
        if args.train_format == "sharegpt":
            text = next((msg["value"] for msg in example.get("conversations", []) if msg.get("from") == "human"), "")
        elif args.train_format == "flat":
            text = example.get(args.train_prompt_key, "")
        else: # Default swiss-ai format
            try:
                text = example.get("initial_prompt", [{}])[0].get("content", "")
            except:
                text = ""

        return {
            "conversation_id": conv_id,
            "prompt_token_ids": tokenizer(text)["input_ids"]
        }

    print("Formatting and tokenizing training prompts...")
    train_data = train_data.map(format_and_tokenize, with_indices=True)

    # Cache the standardized dataset so we can safely delete it from RAM and reload it during the final filtering step
    formatted_train_path = os.path.join(args.output, "_tmp_formatted_train")
    train_data.save_to_disk(formatted_train_path)

    train_data_prompts_token_ids = train_data["prompt_token_ids"]
    train_conversation_ids = train_data["conversation_id"]
    
    del train_data  # Free up memory for multiprocessing
    gc.collect()

    print(f"Generating n-grams from training data ({len(train_data_prompts_token_ids)} prompts)...")
    start_time = time.time()
    with Pool(args.num_proc) as p:
        train_ngrams = []
        total_processed = 0
        for result in p.imap(partial(process_tokens, ngram_length=args.ngram_length),
                             train_data_prompts_token_ids,
                             chunksize=1000):
            train_ngrams.append(result)
            total_processed += 1
            if total_processed % 1000 == 0 or total_processed == len(train_data_prompts_token_ids):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(train_data_prompts_token_ids) - total_processed) / rate if rate > 0 else 0
                print(f"  Processed {total_processed}/{len(train_data_prompts_token_ids)} prompts "
                      f"({total_processed * 100 // len(train_data_prompts_token_ids)}%) - "
                      f"{rate:.0f} prompts/sec - ETA: {int(eta)}s", end='\r')
        print()

    elapsed = time.time() - start_time
    print(f"Training data preprocessing complete in {int(elapsed / 60)}m {elapsed % 60:.1f}s")
    gc.collect()

    # Iterate over benchmarks for decontamination.
    processed_benchmarks = []
    for i, eval_dataset_name in enumerate(benchmark_list, 1):
        start_time = time.time()
        print(f"\n[{i}/{len(benchmark_list)}] Processing benchmark: {eval_dataset_name}")

        # Load and process evaluation prompts
        output_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )  # Reports are saved inside the input dataset's directory
        if os.path.exists(output_path) and not args.overwrite:
            print("contamination_report already exists, skipping decontamination")
            continue
        # Step 1: Load/compute benchmark n-grams
        step_start = time.time()
        cache_key = get_benchmark_cache_key(eval_dataset_name, args.tokenizer_name, args.ngram_length)
        cache_path = get_benchmark_cache_path(args.cache_dir, cache_key)

        cached_data = load_cached_benchmark_ngrams(cache_path)
        if cached_data is not None:
            print(f"  Loading cached n-grams from {cache_path}")
            eval_ngrams = cached_data['ngrams']
            eval_ngram_to_eval_idx = cached_data['lookup_table']
            print(f"  Loaded {len(eval_ngrams)} cached n-grams with {len(eval_ngram_to_eval_idx)} unique entries")
        else:
            # Compute n-grams and cache them
            eval_dataset = eval_data[eval_dataset_name]
            eval_ngrams, eval_ngram_to_eval_idx = compute_benchmark_ngrams(
                eval_dataset, tokenizer, args.ngram_length, args.num_proc
            )
            save_benchmark_ngrams_to_cache(eval_ngrams, eval_ngram_to_eval_idx, cache_path)
            del eval_dataset

        step1_time = time.time() - step_start
        print(f"  Step 1 (Load/compute n-grams): {step1_time:.1f}s")

        # Step 2: Find n-gram matches (with pre-filtering optimization)
        step_start = time.time()
        print(f"  Finding n-gram matches with training data...")

        # Pre-filtering: Only process training samples with ANY n-gram overlap
        benchmark_ngrams_set = set(eval_ngram_to_eval_idx.keys())
        train_samples_with_overlap = []
        train_indices_with_overlap = []

        print(f"  Pre-filtering training samples for n-gram overlap...")
        prefilter_start = time.time()
        for idx, train_sample in enumerate(train_ngrams):
            if train_sample["ngram"] & benchmark_ngrams_set:  # Fast set intersection
                train_samples_with_overlap.append(train_sample)
                train_indices_with_overlap.append(idx)

        prefilter_time = time.time() - prefilter_start
        overlap_ratio = len(train_samples_with_overlap) / len(train_ngrams) * 100
        print(
            f"  Pre-filtering complete: {len(train_samples_with_overlap)}/{len(train_ngrams)} samples have overlap ({overlap_ratio:.1f}%) - {prefilter_time:.1f}s")

        # Only process samples with potential matches
        if len(train_samples_with_overlap) > 0:
            with Pool(args.num_proc, initializer=init_get_eval_match_indices, initargs=(eval_ngram_to_eval_idx,)) as p:
                ngram_match_results = p.map(get_eval_match_indices, train_samples_with_overlap)

            # Map back to original indices
            train_idx_match_indices = {}
            for local_idx, matches in enumerate(ngram_match_results):
                if len(matches) > 0:
                    original_idx = train_indices_with_overlap[local_idx]
                    train_idx_match_indices[original_idx] = matches
            del ngram_match_results
        else:
            train_idx_match_indices = {}

        del train_samples_with_overlap, train_indices_with_overlap, benchmark_ngrams_set, eval_ngram_to_eval_idx
        step2_time = time.time() - step_start
        print(f"  Found potential matches for {len(train_idx_match_indices)} training samples")
        print(f"  Step 2 (Find n-gram matches): {step2_time:.1f}s")

        # Step 3: Verify contamination with sequence matching (parallelized with shared memory)
        step_start = time.time()
        print(f"  Checking {len(train_idx_match_indices)} potential matches against threshold (parallelized)...")

        # Prepare optimized tasks for parallel processing (no eval_ngrams in each task)
        verification_tasks = []
        for train_idx, eval_indices in train_idx_match_indices.items():
            verification_tasks.append((
                train_idx,
                eval_indices,
                train_ngrams[train_idx]["tokens"],
                args.diff_threshold,
                train_conversation_ids[train_idx]
            ))

        # Execute parallel contamination verification with shared memory
        contamination_mapping = {}
        contaminated_samples_shown = 0

        if len(verification_tasks) > 0:
            # Use shared memory initialization to avoid serializing eval_ngrams to each worker
            with Pool(args.num_proc, initializer=init_contamination_worker, initargs=(eval_ngrams,)) as p:
                results = p.map(verify_contamination_worker_optimized, verification_tasks)

            # Process results
            for result in results:
                if result is not None:
                    train_conversation_id, eval_idx, match_ratio, train_tokens_full, eval_tokens_full = result
                    contamination_mapping[train_conversation_id] = eval_idx

                    # Show contaminated samples if requested
                    if args.show_contaminated and contaminated_samples_shown < 10:
                        print(f"\n  [CONTAMINATED] Sample {train_conversation_id}")
                        print(f"    Training prompt: {tokenizer.decode(train_tokens_full)}")
                        print(f"    Benchmark prompt: {tokenizer.decode(eval_tokens_full)}")
                        print(f"    Match ratio: {match_ratio:.2f}")
                        contaminated_samples_shown += 1
        step3_time = time.time() - step_start
        print(f"  Found {len(contamination_mapping)} contaminated samples")
        print(f"  Step 3 (Verify contamination): {step3_time:.1f}s")

        del eval_ngrams, train_idx_match_indices
        gc.collect()

        # Step 4: Save report
        step_start = time.time()
        json.dump(contamination_mapping, open(output_path, "w"), separators=(",", ":"))
        step4_time = time.time() - step_start
        print(f"  Step 4 (Save report): {step4_time:.1f}s")

        # Total time summary
        running_time = time.time() - start_time
        print(f"  Total benchmark time: {int(running_time / 60)}m {running_time % 60:.1f}s")
        print(f"    - Load/compute n-grams: {step1_time:.1f}s ({step1_time / running_time * 100:.1f}%)")
        print(f"    - Find n-gram matches: {step2_time:.1f}s ({step2_time / running_time * 100:.1f}%)")
        print(f"      * Pre-filtering: {prefilter_time:.1f}s")
        print(f"    - Verify contamination: {step3_time:.1f}s ({step3_time / running_time * 100:.1f}%)")
        print(f"    - Save report: {step4_time:.1f}s ({step4_time / running_time * 100:.1f}%)")

        processed_benchmarks.append(eval_dataset_name)
        del contamination_mapping
        gc.collect()

    # Load the contamination reports and filter the training data
    print(f"\n=== FINAL FILTERING ===\nCombining contamination reports from {len(processed_benchmarks)} processed benchmarks...")
    del eval_data
    gc.collect()
    
    # LOAD FROM CACHE INSTEAD OF RAW PATH
    formatted_train_path = os.path.join(args.output, "_tmp_formatted_train")
    train_data = load_from_disk(formatted_train_path)
    
    contaminated_ids = set()
    contamination_summary = []
    for eval_dataset_name in processed_benchmarks:
        report_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )
        with open(report_path, "r") as f:
            report = json.load(f)
        contaminated_ids = contaminated_ids.union(set(report.keys()))
        if len(report) > 0:
            contamination_summary.append((eval_dataset_name, len(report)))
        print(f"  - {eval_dataset_name}: {len(report)} contaminated samples")

    print(f"\n=== CONTAMINATION SUMMARY ===")
    print(f"Total contaminated samples to remove: {len(contaminated_ids)}")
    if contamination_summary:
        print(f"\nBenchmarks with contamination:")
        contamination_summary.sort(key=lambda x: x[1], reverse=True)
        for benchmark, count in contamination_summary[:20]:  # Show top 20
            print(f"  {benchmark}: {count} samples")
        if len(contamination_summary) > 20:
            print(f"  ... and {len(contamination_summary) - 20} more benchmarks")
    # Original single dataset filtering (commented out for DatasetDict compatibility):
    # train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)
    # train_data.save_to_disk(args.output)

    # Filter out contaminated samples
    train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)

    # Save dataset with metadata
    save_dataset_and_metadata(
        train_data,
        Path(args.output),
        Path(args.dataset_path),
        contaminated_ids,
        processed_benchmarks,
        args.tokenizer_name,
        args.ngram_length,
        args.diff_threshold
    )

    # Clean up the temporary cache directory
    shutil.rmtree(formatted_train_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a decontamination report for a dataset with optimized pre-filtering and parallel verification."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path or name of the training dataset to process. Must include a column with 'messages' that "
             "contains the standard HuggingFace chat format of List[Dict[str, str]]",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for filtered dataset",
    )
    parser.add_argument(
        "--decontamination_prompts",
        type=str,
        default=None,
        help="Name of the dataset with benchmark samples to use for decontamination. "
             "It expects a DatasetDict with multiple benchmark samples init, each with columns:"
             " config_name, split_name, prompts",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="alehc/swissai-tokenizer",
        help="Name of the tokenizer to use for decontamination. "
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="Name of the benchmark samples to use for decontamination. Use either a single benchmark's name or a list with comma separation."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        required=True,
        help="Path for the output JSON with decontamination report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate contamination reports even if they already exist. Without this flag, existing reports are skipped.",
    )
    parser.add_argument(
        "--ngram_length",
        type=int,
        default=8,
        help="Length of the n-grams to consider.",
    )
    parser.add_argument(
        "--diff_threshold",
        type=float,
        default=0.5,
        help="Threshold for filtering based on difference ratio.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes to use for map operations.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./benchmark_cache",
        help="Directory to store cached benchmark n-grams (default: ./benchmark_cache).",
    )
    parser.add_argument(
        "--show_contaminated",
        action="store_true",
        help="Display examples of contaminated samples during processing (shows first 10 per benchmark).",
    )

    parser.add_argument(
        "--train_format",
        type=str,
        default="default",
        choices=["default", "sharegpt", "flat"],
        help="Schema of the training data. 'sharegpt' reads the first human message in 'conversations'. 'flat' reads a single text column. 'default' reads 'initial_prompt' dicts."
    )
    parser.add_argument(
        "--train_prompt_key",
        type=str,
        default="text",
        help="Column name containing the training prompt if train_format is 'flat'."
    )
    parser.add_argument(
        "--eval_format",
        type=str,
        default="hf",
        choices=["hf", "raw_json"],
        help="Format of the evaluation data. 'raw_json' automatically wraps your JSON/JSONL into a DatasetDict."
    )
    parser.add_argument(
        "--eval_prompt_key",
        type=str,
        default="prompt",
        help="Column name containing the evaluation question/prompt (e.g., 'question')."
    )

    args = parser.parse_args()
    main(args)