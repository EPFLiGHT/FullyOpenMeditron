#!/usr/bin/env python3
# resampling/corpora.py
"""
The registry of training corpora: where each file lives, what produced it, and
whether it is still the file that was registered.

data/corpora.yaml is the catalogue, tracked in git. The corpora themselves live
under data/corpora/<corpus>/<dataset>.jsonl (a symlink to persistent project
storage in our setup; not the home directory, whose quota is small, and not
scratch, which is purged).

A corpus is registered once. Re-registering a name with different content is an
error, so "meditronfo_qwen38_v1" always means the same bytes; a changed corpus
is a new version.

usage:
python3 resampling/corpora.py register meditronfo_qwen38_v1 --model Qwen/Qwen3.8-27B \\
    --file curated_qa=<path> --file synthetic_qa=<path> \\
    --file guidelines_qa=<path> --file synthetic_moove=<path> \\
    --job 3459326 --note "labels not relabelled"
python3 resampling/corpora.py list
python3 resampling/corpora.py check                   # every corpus: exists, rows, sha256
python3 resampling/corpora.py check --quick           # exists and size only
python3 resampling/corpora.py axolotl meditronfo_qwen38_v1 > datasets_block.yaml

    --in-place   register files where they are instead of copying them into the store
                 (for a corpus that already lives somewhere persistent)
"""

import argparse
import copy
import datetime
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "corpora.yaml"
STORE = Path(__file__).resolve().parents[1] / "data" / "corpora"

# One name per dataset role, whatever the file was called upstream: a Qwen
# answer file must not be called gpt_oss_8retries_... because its questions
# came from there.
ROLES = ("curated_qa", "synthetic_qa", "guidelines_qa", "synthetic_moove")

AXOLOTL_DATASET = {
    "ds_type": "json",
    "type": "chat_template",
    "field_messages": "conversations",
    "message_property_mappings": {"role": "from", "content": "value"},
    "roles": {"system": ["system"], "user": ["user", "human"],
              "assistant": ["assistant", "gpt"]},
}


def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return ""


def load(registry):
    if Path(registry).exists():
        with open(registry, encoding="utf-8") as f:
            return yaml.safe_load(f) or {"corpora": {}}
    return {"corpora": {}}


def save(reg, registry):
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Training corpora. Written by resampling/corpora.py -- register with it\n"
                "# rather than editing by hand, so rows and sha256 stay true.\n")
        yaml.safe_dump(reg, f, sort_keys=False, allow_unicode=True, width=100)
    tmp.replace(registry)


def parse_files(specs):
    files = {}
    for spec in specs:
        role, sep, path = spec.partition("=")
        if not sep or not path:
            raise SystemExit(f"--file {spec!r}: expected ROLE=PATH")
        if role not in ROLES:
            raise SystemExit(f"--file {spec!r}: role must be one of {', '.join(ROLES)}")
        if role in files:
            raise SystemExit(f"--file: role {role} given twice")
        if not Path(path).is_file():
            raise SystemExit(f"--file {spec!r}: no such file")
        files[role] = Path(path).resolve()
    return files


def place(src, dest):
    """Copy src to dest atomically; a partial copy never sits at dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)


def cmd_register(args):
    reg = load(args.registry)
    files = parse_files(args.file)
    if not files:
        raise SystemExit("register: give at least one --file ROLE=PATH")

    digests = {role: sha256(files[role]) for role in ROLES if role in files}

    # Decide before touching the store: a refused registration must not have
    # already copied its files over the ones the registry vouches for.
    old = reg["corpora"].get(args.name)
    if old:
        if {r: e["sha256"] for r, e in old["files"].items()} == digests:
            print(f"--- {args.name}: unchanged, registry not rewritten ---")
            return
        if not args.force:
            raise SystemExit(f"{args.name} is already registered with different content. "
                             f"Register a new version (e.g. _v2) instead; --force overwrites.")

    entries = {}
    for role, digest in digests.items():                  # ROLES order: stable file
        src = files[role]
        if args.in_place:
            dest = src
        else:
            dest = Path(args.store) / args.name / f"{role}.jsonl"
            if dest.exists() and sha256(dest) == digest:
                print(f"  {role}: already in place")
            else:
                print(f"  {role}: copying {src} -> {dest}")
                place(src, dest)
        entries[role] = {"path": str(dest), "rows": count_rows(dest),
                         "bytes": dest.stat().st_size, "sha256": digest,
                         "source": str(src)}

    new = {"answer_model": args.model,
           "created": datetime.date.today().isoformat(),
           "git_commit": git_commit(),
           "jobs": [str(j) for j in args.job],
           "notes": list(args.note),
           "files": entries}

    reg["corpora"][args.name] = new
    save(reg, args.registry)
    total = sum(e["rows"] for e in entries.values())
    print(f"--- registered {args.name}: {len(entries)} files, {total} rows -> {args.registry} ---")


def cmd_note(args):
    """Append a dated note. Files and hashes are never touched: a note corrects
    what the registry *says* about a corpus, never what the corpus *is*."""
    reg = load(args.registry)
    c = reg["corpora"].get(args.name)
    if c is None:
        raise SystemExit(f"{args.name}: not registered")
    c.setdefault("notes", []).append(f"[{datetime.date.today().isoformat()}] {args.text}")
    save(reg, args.registry)
    print(f"--- {args.name}: note added ---")


def cmd_list(args):
    reg = load(args.registry)
    if not reg["corpora"]:
        print("(no corpora registered)")
        return
    for name, c in reg["corpora"].items():
        rows = sum(e["rows"] for e in c["files"].values())
        print(f"{name:32} {c['answer_model']:28} {rows:>8} rows  "
              f"{','.join(c['files'])}  ({c['created']})")


def cmd_check(args):
    reg = load(args.registry)
    names = args.names or list(reg["corpora"])
    bad = 0
    for name in names:
        c = reg["corpora"].get(name)
        if c is None:
            print(f"FAIL {name}: not registered"); bad += 1; continue
        for role, e in c["files"].items():
            p = Path(e["path"]) if Path(e["path"]).is_absolute() else REPO / e["path"]
            problem = None
            if not p.is_file():
                problem = "missing"
            elif p.stat().st_size != e["bytes"]:
                problem = f"size {p.stat().st_size} != {e['bytes']}"
            elif not args.quick and sha256(p) != e["sha256"]:
                problem = "sha256 changed"
            elif not args.quick and count_rows(p) != e["rows"]:
                problem = "row count changed"
            if problem:
                print(f"FAIL {name}/{role}: {problem} ({p})"); bad += 1
            else:
                print(f"ok   {name}/{role}: {e['rows']} rows")
    if bad:
        raise SystemExit(f"{bad} problem(s)")


def cmd_axolotl(args):
    reg = load(args.registry)
    c = reg["corpora"].get(args.name)
    if c is None:
        raise SystemExit(f"{args.name}: not registered")
    # deepcopy per entry: a shared dict makes safe_dump emit &id001 anchors,
    # valid YAML but unreadable once pasted into a config
    block = {"datasets": [{"path": c["files"][r]["path"], **copy.deepcopy(AXOLOTL_DATASET)}
                          for r in ROLES if r in c["files"]]}
    text = (f"# {args.name}: {c['answer_model']}, registered {c['created']} "
            f"(data/corpora.yaml)\n" + yaml.safe_dump(block, sort_keys=False, width=100))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"--- wrote {args.out} ---")
    else:
        sys.stdout.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=str(REGISTRY))
    p.add_argument("--store", default=str(STORE))
    p.add_argument("--selfcheck", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("register")
    r.add_argument("name")
    r.add_argument("--model", required=True, help="the model that wrote the answers")
    r.add_argument("--file", action="append", default=[], help="ROLE=PATH")
    r.add_argument("--job", action="append", default=[], help="Slurm job id(s) that produced it")
    r.add_argument("--note", action="append", default=[], help="a caveat a trainer should know")
    r.add_argument("--in-place", action="store_true")
    r.add_argument("--force", action="store_true")

    nt = sub.add_parser("note", help="append a dated note (e.g. a correction)")
    nt.add_argument("name")
    nt.add_argument("text")

    sub.add_parser("list")
    c = sub.add_parser("check")
    c.add_argument("names", nargs="*")
    c.add_argument("--quick", action="store_true", help="existence and size only")
    a = sub.add_parser("axolotl")
    a.add_argument("name")
    a.add_argument("--out")

    args = p.parse_args()
    if args.selfcheck:
        return _selfcheck()
    if not args.cmd:
        p.print_help()
        return
    {"register": cmd_register, "note": cmd_note, "list": cmd_list, "check": cmd_check,
     "axolotl": cmd_axolotl}[args.cmd](args)


def _selfcheck():
    import tempfile
    from types import SimpleNamespace as NS
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "gpt_oss_8retries_resampled.jsonl"          # misleading upstream name
        src.write_text('{"conversations": []}\n{"conversations": []}\n')
        reg, store = d / "corpora.yaml", d / "store"
        base = dict(registry=reg, store=store, model="m", job=["1"], note=["n"],
                    in_place=False, force=False)
        cmd_register(NS(name="c_v1", file=[f"curated_qa={src}"], **base))
        entry = load(reg)["corpora"]["c_v1"]["files"]["curated_qa"]
        assert entry["path"].endswith("c_v1/curated_qa.jsonl"), entry  # honest name
        assert entry["rows"] == 2 and Path(entry["path"]).read_text() == src.read_text()
        cmd_register(NS(name="c_v1", file=[f"curated_qa={src}"], **base))  # idempotent
        # same name, different bytes: refused, so a version always means one content
        registered = Path(entry["path"]).read_text()
        src.write_text('{"conversations": []}\n')
        try:
            cmd_register(NS(name="c_v1", file=[f"curated_qa={src}"], **base))
            raise AssertionError("re-registering changed content was allowed")
        except SystemExit as e:
            assert "different content" in str(e)
        # ...and the refusal must leave the registered file untouched
        assert Path(entry["path"]).read_text() == registered, "refused register overwrote the store"
        # check catches a file changed after registration
        Path(entry["path"]).write_text("tampered\n")
        try:
            cmd_check(NS(registry=reg, names=["c_v1"], quick=False))
            raise AssertionError("tampering was not caught")
        except SystemExit:
            pass
        # a note is appended, dated, and leaves the file entries alone
        before = load(reg)["corpora"]["c_v1"]["files"]
        cmd_note(NS(registry=reg, name="c_v1", text="correction"))
        after = load(reg)["corpora"]["c_v1"]
        assert after["files"] == before and after["notes"][-1].endswith("] correction")
        try:
            parse_files([f"moove={src}"])
            raise AssertionError("unknown role accepted")
        except SystemExit as e:
            assert "role must be" in str(e)
    print("selfcheck ok")


if __name__ == "__main__":
    main()
