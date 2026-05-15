"""Data-path configuration for the distribution-shift analysis pipeline.

Resolves the six dataset paths from three sources, in priority order:

  1. CLI flag (e.g., --moove-source /path/to/file)
  2. YAML config file (e.g., moove_source: path/to/file)
  3. Built-in default (this module's DEFAULTS dict)

Relative paths in the config or defaults are resolved against ``--root``;
absolute paths are used as-is.

Usage in an orchestrator script::

    parser.add_argument("--config", type=str,
                        default="data_analysis/configs/default.yaml")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--moove-source", type=str, default=None)
    parser.add_argument("--moove-synthetic", type=str, default=None)
    parser.add_argument("--meditron-source", type=str, default=None)
    parser.add_argument("--meditron-synthetic", type=str, default=None)
    parser.add_argument("--guidelines-dir", type=str, default=None)
    parser.add_argument("--guidelines-synthetic", type=str, default=None)
    args = parser.parse_args()

    paths = load_data_paths(args)
    moove_src = paths["moove_source"]            # Path object
    curated_syn = paths["curated_synthetic"]     # Path object
    # ...
"""

from pathlib import Path

# Defaults target the FullyOpenMeditron data layout. Override via the
# YAML config or CLI flags for other projects.
DEFAULTS = {
    "moove_source":         "data/moove/full_moove.jsonl",
    "moove_synthetic":      "data/synthetic_moove/synthetic_moove_v2_gpt_oss_1try.jsonl",
    "curated_source":       "data/curated_qa/meditron_4_cleaned.jsonl",
    "curated_synthetic":    "data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl",
    "guidelines_dir":       "data/guidelines",
    "guidelines_synthetic": "data/guidelines_qa/guidelines_qa_v3_full.jsonl",
}

# Map argparse-Namespace attribute name -> canonical config key.
# Legacy flag names (--meditron-source, --meditron-synthetic) point at the
# Curated QA pair; the canonical keys use the paper's terminology.
_FLAG_TO_KEY = {
    "moove_source":         "moove_source",
    "moove_synthetic":      "moove_synthetic",
    "meditron_source":      "curated_source",
    "meditron_synthetic":   "curated_synthetic",
    "guidelines_dir":       "guidelines_dir",
    "guidelines_synthetic": "guidelines_synthetic",
}


def _load_yaml(path):
    """Read a YAML config file; return {} for None or missing paths.

    Lazy-imports yaml so the dependency is only required when a config is used.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        # Default config path may not exist (e.g. when running from a
        # different working directory) — fall back to defaults silently.
        return {}
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load a config file. Install with "
            "`pip install pyyaml`, or pass --config '' to skip the config "
            "and use built-in defaults."
        ) from e
    with p.open() as f:
        return yaml.safe_load(f) or {}


def load_data_paths(args):
    """Resolve all six dataset paths from CLI args, config file, and defaults.

    Returns a dict mapping {key: Path}, where keys are:
        moove_source, moove_synthetic,
        curated_source, curated_synthetic,
        guidelines_dir, guidelines_synthetic
    """
    root = Path(getattr(args, "root", ".") or ".")
    config_path = getattr(args, "config", None)
    cfg = _load_yaml(config_path)

    resolved = {}
    for flag_attr, key in _FLAG_TO_KEY.items():
        cli_value = getattr(args, flag_attr, None)
        cfg_value = cfg.get(key)
        default = DEFAULTS[key]

        chosen = cli_value or cfg_value or default
        path = Path(chosen)
        if not path.is_absolute():
            path = root / path
        resolved[key] = path

    return resolved
