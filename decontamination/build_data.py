from datasets import load_dataset, Dataset, DatasetDict

def medqa_prompt(ex):
    opts = "\n".join(
        f"{letter}. {ex[f'ending{i}']}"
        for i, letter in enumerate("ABCD")
    )
    return f"{ex['sent1']}\n{opts}"

def medxpert_prompt(ex):
    # `question` already includes "Answer Choices: (A) ... (B) ..." inline,
    # so we don't need to re-append options.
    return ex["question"]

def pubmedqa_prompt(ex):
    ctx = " ".join(ex["CONTEXTS"]) if ex.get("CONTEXTS") else ""
    return f"{ctx}\n{ex['QUESTION']}"

def medmcqa_prompt(ex):
    opts = (
        f"A. {ex['opa']}\n"
        f"B. {ex['opb']}\n"
        f"C. {ex['opc']}\n"
        f"D. {ex['opd']}"
    )
    return f"{ex['question']}\n{opts}"

benches = {}

medqa = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
benches["medqa_test"] = Dataset.from_dict(
    {"prompt": [medqa_prompt(x) for x in medqa]}
)

medxpert = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
benches["medxpertqa_test"] = Dataset.from_dict(
    {"prompt": [medxpert_prompt(x) for x in medxpert]}
)

pubmed = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test")
benches["pubmedqa_test"] = Dataset.from_dict(
    {"prompt": [pubmedqa_prompt(x) for x in pubmed]}
)

medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
benches["medmcqa_val"] = Dataset.from_dict(
    {"prompt": [medmcqa_prompt(x) for x in medmcqa]}
)

DatasetDict(benches).save_to_disk("data/decontam_prompts/med_benchmarks_evalonly")

# Quick sanity check
for name, ds in benches.items():
    print(f"\n=== {name} ({len(ds)} prompts) ===")
    print(ds[0]["prompt"][:400])