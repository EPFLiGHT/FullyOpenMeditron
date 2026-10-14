#!/usr/bin/env python3
# auto_moove/moove_criteria_agreement.py

"""Judge-vs-physician agreement per MOOVE criterion on the unanimous-576 set.

The set's rows carry only the final verdict. Each physician also scored both
answers on the five MOOVE dimensions; those reviews are joined back from
moove/all_reviews.jsonl through annotation_ids, and a review's preference on
a criterion is the sign of left minus right on that dimension (Equal when
the two match). The dimension score is the mean of the site's sub-criteria,
with moove_build's mapping, inversion and offsets, left unrounded: the
integer scores in the export round small differences away (--rounded uses
them instead).

The judge's preferences come from auto_moove_check.py --rubric moove_pref
(j_pref_<criterion>, and the _ab / _ba single orders).

Three human labels per criterion:
  unanimous  pairs whose raters all give the same preference on it
  majority   the raters' plurality; pairs tied at the top are left out
  review     every review on its own, the judge's pair verdict repeated
Human-human: every two raters of one pair, the ceiling for the review level.
The overall verdict (--overall, any rubric with one) goes through the same
four, against the reviews' votes.

.venv/bin/python auto_moove/moove_criteria_agreement.py \
    --judged auto_moove/judge_outputs/judged_unanimous576_moove_pref_generate_bidir_gemma4_31b.jsonl \
    --overall auto_moove/judge_outputs/judged_unanimous576_winner_generate_bidir_gemma4_31b.jsonl

Event mode (--prefix, repeatable): every review of those MOOVE events in
moove/all_reviews.jsonl that scored both answers, instead of the unanimous
set. First write the judge input, one row per pair, then judge it with
auto_moove_check.py (--rubric moove_pref, and --rubric winner for --overall):

.venv/bin/python auto_moove/moove_criteria_agreement.py --prefix "Kenya PHC" \
    --prefix "Tanzania PHC" --prefix "Malawi PHC" --write-input <dir>/pairs.jsonl
.venv/bin/python auto_moove/moove_criteria_agreement.py --prefix ... \
    --judged <moove_pref output> --overall <winner output>
"""
import argparse, difflib, itertools, json, math, os, sys
from collections import Counter
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moove"))
from moove_build import (CRITERION_TO_DIMENSION, DEFAULT_OFFSET, INVERT_CRITERIA,
                         PREFIX_OFFSET)
from moove_meta import pair_key_for
from auto_moove_prompt import MOOVE
from auto_moove_check import HUMAN_LABEL, cohen_kappa, macro_f1

LABEL_VOTE = {"Model 1": "1", "Model 2": "2", "Equal": "12"}
REVIEW_VOTE = {"left": "1", "right": "2", "equal": "12"}
FLIP = {"1": "2", "2": "1", "12": "12"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/moove/moove_eval_unanimous_576.jsonl")
    p.add_argument("--reviews", default="moove/all_reviews.jsonl")
    p.add_argument("--prefix", action="append",
                   help="event mode: these MOOVE events' reviews instead of --test")
    p.add_argument("--write-input",
                   help="event mode: write the judge input (one row per pair) here and stop")
    p.add_argument("--judged", help="auto_moove_check.py --rubric moove_pref output")
    p.add_argument("--overall", help="an auto_moove_check.py output with an overall verdict "
                                     "on the same set, reported as a reference row")
    p.add_argument("--rounded", action="store_true",
                   help="compare the export's integer dimension scores instead")
    p.add_argument("--length-bias", action="store_true",
                   help="does either side follow answer length?")
    p.add_argument("--tsv", help="write the tables here")
    args = p.parse_args()
    if args.write_input and not args.prefix:
        p.error("--write-input needs --prefix")
    if not (args.judged or args.write_input):
        p.error("--judged is required, unless --write-input")
    return args


def _norm(s):
    return " ".join((s or "").split())


def _sim(a, b):
    return difflib.SequenceMatcher(None, a[:2000], b[:2000]).ratio()


def is_flipped(review, row):
    """True when the review's left answer is the set's Model 2. The set's
    texts are whitespace-collapsed and a few differ slightly, so the side is
    the closer assignment rather than an exact match."""
    L, R = _norm(review["left_answer_text"]), _norm(review["right_answer_text"])
    m1, m2 = _norm(row["model_1_answer"]), _norm(row["model_2_answer"])
    return _sim(L, m2) + _sim(R, m1) > _sim(L, m1) + _sim(R, m2)


def dimension_scores(review, side):
    """{MOOVE key: mean of the review's sub-criteria on that dimension}, the
    way moove_build.normalize_evaluations averages them, before rounding."""
    offset = PREFIX_OFFSET.get(review.get("prefix"), DEFAULT_OFFSET)
    buckets = {}
    head = f"{side} evaluations: "
    for col, v in review.items():
        if not col.startswith(head) or v is None:
            continue
        crit = col[len(head):]
        dim = CRITERION_TO_DIMENSION.get(crit)
        if dim is None:
            continue
        x = float(v) - offset
        buckets.setdefault(dim, []).append(-x if crit in INVERT_CRITERIA else x)
    return {c: (sum(buckets[d]) / len(buckets[d]) if (d := HUMAN_LABEL[c]) in buckets else None)
            for c in MOOVE}


def rounded_scores(review, side):
    return {c: (review[f"{side} score: {HUMAN_LABEL[c]}"]
                if review.get(f"{side} n_criteria: {HUMAN_LABEL[c]}") else None)
            for c in MOOVE}


def criterion_prefs(rv, scores, flip):
    """{MOOVE key: vote code} from one review's scores, in the judge's frame."""
    left, right = scores(rv, "Left"), scores(rv, "Right")
    out = {}
    for c in MOOVE:
        if left[c] is None or right[c] is None:
            out[c] = None
            continue
        pref = "1" if left[c] > right[c] else "2" if right[c] > left[c] else "12"
        out[c] = FLIP[pref] if flip else pref
    return out


def event_reviews(args):
    """Event mode: one record per review of the --prefix events that scored
    both answers, in the frame of the pair's first such review -- the row
    written to --write-input, whose left answer is the judge's Model 1."""
    prefixes = set(args.prefix)
    by_pair = {}
    with open(args.reviews, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("prefix") not in prefixes:
                continue
            sides = {k.split()[0] for k, v in d.items() if " evaluations: " in k and v is not None}
            if sides == {"Left", "Right"}:
                by_pair.setdefault(pair_key_for(d), []).append(d)
    found = {r["prefix"] for rs in by_pair.values() for r in rs}
    if prefixes - found:
        raise SystemExit(f"no scored reviews for {sorted(prefixes - found)} in {args.reviews}")
    if args.write_input:
        with open(args.write_input, "w", encoding="utf-8") as out:
            for rs in by_pair.values():
                out.write(json.dumps(rs[0], ensure_ascii=False) + "\n")
        print(f"{len(by_pair)} pairs written to {args.write_input}")
        raise SystemExit(0)

    scores = rounded_scores if args.rounded else dimension_scores
    out = []
    for pk, rs in by_pair.items():
        frame = rs[0]
        for rv in rs:
            if rv["left_ID"] == frame["left_ID"] and rv["right_ID"] == frame["right_ID"]:
                flip = False
            elif rv["left_ID"] == frame["right_ID"] and rv["right_ID"] == frame["left_ID"]:
                flip = True
            else:
                flip = is_flipped(rv, {"model_1_answer": frame["left_answer_text"],
                                       "model_2_answer": frame["right_answer_text"]})
            vote = REVIEW_VOTE[rv["Vote"].strip().lower()]
            out.append({"pair_key": pk, "review": rv["ID"], "prefix": rv["prefix"],
                        "vote": FLIP[vote] if flip else vote,
                        # the frame's sides, i.e. the judge's Model 1 / Model 2
                        "len1": words(frame["left_answer_text"]),
                        "len2": words(frame["right_answer_text"]),
                        **criterion_prefs(rv, scores, flip)})
    return pd.DataFrame(out)


def human_reviews(args):
    """One record per review: pair id, the review's overall vote and its
    preference per criterion, both in the set's Model 1 / Model 2 frame."""
    test = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    need = {a for r in test for a in r["annotation_ids"]}
    reviews = {}
    with open(args.reviews, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("ID") in need:
                reviews[d["ID"]] = d
    missing = need - reviews.keys()
    if missing:
        raise SystemExit(f"{len(missing)} annotation ids not in {args.reviews}")

    scores = rounded_scores if args.rounded else dimension_scores
    out = []
    for row in test:
        for a in row["annotation_ids"]:
            rv = reviews[a]
            flip = is_flipped(rv, row)
            vote = REVIEW_VOTE[rv["Vote"].strip().lower()]
            vote = FLIP[vote] if flip else vote
            # the set is unanimous: every review's vote is the pair's label,
            # which checks the orientation too
            assert vote == LABEL_VOTE[row["label"]], (row["gold_id"], a)
            out.append({"pair_key": row["gold_id"], "review": a, "vote": vote,
                        "prefix": rv.get("prefix"), **criterion_prefs(rv, scores, flip)})
    return pd.DataFrame(out)


def metrics(human, judge):
    human, judge = list(human), list(judge)
    n = len(human)
    decisive = [(h, j) for h, j in zip(human, judge) if h != "12"]
    return {"n": n,
            "three_way": 100 * sum(h == j for h, j in zip(human, judge)) / n if n else float("nan"),
            "binary": 100 * sum(h == j for h, j in decisive) / len(decisive) if decisive else float("nan"),
            "macro_f1": 100 * macro_f1(human, judge) if n else float("nan"),
            "kappa": cohen_kappa(human, judge) if n else float("nan"),
            "h_equal": 100 * human.count("12") / n if n else float("nan"),
            "j_equal": 100 * judge.count("12") / n if n else float("nan")}


def pair_labels(reviews, c, how):
    """{pair: label} from the raters' preferences on criterion c."""
    labels = {}
    for pk, g in reviews.groupby("pair_key"):
        prefs = [p for p in g[c] if isinstance(p, str)]   # pandas reads None as NaN
        if len(prefs) < 2:          # every pair has two raters; a missing score leaves one
            continue
        counts = Counter(prefs).most_common()
        if how == "unanimous":
            if len(counts) == 1:
                labels[pk] = counts[0][0]
        elif len(counts) == 1 or counts[0][1] > counts[1][1]:
            labels[pk] = counts[0][0]
    return labels


def human_human(reviews, c):
    """Every two raters of one pair, as (rating, rating); kappa pools both
    raters' marginals, since neither rater is a fixed role."""
    a, b = [], []
    for _, g in reviews.groupby("pair_key"):
        prefs = [p for p in g[c] if isinstance(p, str)]   # pandas reads None as NaN
        for x, y in itertools.combinations(prefs, 2):
            a += [x, y]
            b += [y, x]
    return a, b


# words, and the log length ratio of auto_moove_forest_chosen_coverage's
# answer_length feature; SIMILAR is unanimous576_h2h_tables' band
SIMILAR = (0.8, 1.25)


def words(s):
    return len((s or "").split())


def answer_lengths(args, reviews):
    """{pair_key: (words of Model 1, words of Model 2)}. Event mode carries
    the lengths of the judged frame on every review."""
    if "len1" in reviews:
        one = reviews.drop_duplicates("pair_key")
        return dict(zip(one.pair_key, zip(one.len1, one.len2)))
    out = {}
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["gold_id"]] = (words(r["model_1_answer"]), words(r["model_2_answer"]))
    return out


PREF_SCORE = {"1": 1.0, "12": 0.0, "2": -1.0}     # prefers Model 1 / neither / Model 2


def length_bias(reviews, items, lengths):
    """Per criterion: does the physicians' label, or Gemma's verdict, follow
    the longer answer? On pairs unanimous on that criterion, with both sides
    decisive and the answers of different length.

    longer_rule is the agreement of "always pick the longer answer", the
    baseline any length-following judge beats for free."""
    rows = []
    for c, name, jd in items:
        jv = {k: r.get(f"j_pref_{c}") for k, r in jd.items()}
        lab = pair_labels(reviews, c, "unanimous")
        keys = [k for k in lab if jv.get(k) and k in lengths and lengths[k][0] != lengths[k][1]]
        longer = {k: ("1" if lengths[k][0] > lengths[k][1] else "2") for k in keys}
        dec = [k for k in keys if lab[k] != "12"]           # physicians picked a side
        jdec = [k for k in keys if jv[k] != "12"]
        both = [k for k in dec if jv[k] != "12"]
        # "picks the longer answer" on the pairs where both picked a side:
        # discordant counts feed an exact McNemar test
        h_only = sum(lab[k] == longer[k] and jv[k] != longer[k] for k in both)
        j_only = sum(jv[k] == longer[k] and lab[k] != longer[k] for k in both)
        rec = {"criterion": name, "n": len(keys), "n_decisive": len(dec),
               "n_j_decisive": len(jdec), "n_both_decisive": len(both),
               "h_longer": 100 * sum(lab[k] == longer[k] for k in dec) / len(dec),
               "j_longer": 100 * sum(jv[k] == longer[k] for k in jdec) / len(jdec),
               "p_h_vs_50": binomtest(sum(lab[k] == longer[k] for k in dec), len(dec)).pvalue,
               "p_j_vs_50": binomtest(sum(jv[k] == longer[k] for k in jdec), len(jdec)).pvalue,
               "p_h_vs_j": binomtest(j_only, h_only + j_only).pvalue if h_only + j_only else 1.0,
               # always-longer as a judge: three-way over all pairs, binary over decisive
               "longer_rule_3way": 100 * sum(lab[k] == longer[k] for k in keys) / len(keys),
               "longer_rule_bin": 100 * sum(lab[k] == longer[k] for k in dec) / len(dec),
               "gemma_3way": 100 * sum(lab[k] == jv[k] for k in keys) / len(keys)}
        # Gemma's agreement where the physicians preferred the longer / shorter answer
        for lbl, sub in (("agree_h_longer", [k for k in dec if lab[k] == longer[k]]),
                         ("agree_h_shorter", [k for k in dec if lab[k] != longer[k]])):
            rec[lbl] = 100 * sum(lab[k] == jv[k] for k in sub) / len(sub) if sub else float("nan")
            rec[lbl + "_n"] = len(sub)
        # does the preference move with the length gap at all
        ratios = [math.log((lengths[k][0] + 1) / (lengths[k][1] + 1)) for k in keys]
        rec["rho_h"] = spearmanr([PREF_SCORE[lab[k]] for k in keys], ratios).statistic
        rec["rho_j"] = spearmanr([PREF_SCORE[jv[k]] for k in keys], ratios).statistic
        # Equal calls where the answers are of similar length, and where they are not
        lo, hi = SIMILAR
        near = [k for k in keys if lo <= lengths[k][0] / max(1, lengths[k][1]) <= hi]
        far = [k for k in keys if k not in set(near)]
        for lbl, sub in (("near", near), ("far", far)):
            rec[f"h_equal_{lbl}"] = 100 * sum(lab[k] == "12" for k in sub) / len(sub) if sub else float("nan")
            rec[f"j_equal_{lbl}"] = 100 * sum(jv[k] == "12" for k in sub) / len(sub) if sub else float("nan")
            rec[f"n_{lbl}"] = len(sub)
            # agreement with length controlled: near pairs give the judge no
            # length cue to ride on
            rec[f"agree_{lbl}"] = (100 * sum(lab[k] == jv[k] for k in sub) / len(sub)
                                   if sub else float("nan"))
            sd = [k for k in sub if lab[k] != "12" and jv[k] != "12"]
            rec[f"binary_{lbl}"] = (100 * sum(lab[k] == jv[k] for k in sd) / len(sd)
                                    if sd else float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("criterion")


def words_summary(lengths, reviews, items):
    """One line on how far apart the two answers are on this set."""
    keys = [k for k in lengths if k in set(reviews.pair_key)]
    m1 = pd.Series([lengths[k][0] for k in keys])
    m2 = pd.Series([lengths[k][1] for k in keys])
    ratio = (m1 / m2.clip(lower=1))
    lo, hi = SIMILAR
    return (f"{len(keys)} pairs, median words {m1.median():.0f} (Model 1) and {m2.median():.0f} (Model 2)",
            f"word ratio: median {ratio.median():.2f}, "
            f"similar length ({lo}-{hi}) {100 * ratio.between(lo, hi).mean():.1f}% of pairs, "
            f"Model 1 longer {100 * (m1 > m2).mean():.1f}%, equal {100 * (m1 == m2).mean():.1f}%")


def load_judged(path, rename=None):
    """{pair_key: record}; rename maps the overall verdict's fields onto the
    j_pref_<c> names the criteria use."""
    judged = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if rename:
                r = dict(r, **{new: r.get(old) for old, new in rename.items()})
            judged.setdefault(r["pair_key"], r)
    return judged


def main():
    args = parse_args()
    reviews = event_reviews(args) if args.prefix else human_reviews(args)
    judged = load_judged(args.judged)
    unparsed = sum(not r["parsed"] for r in judged.values())
    print(f"{reviews.pair_key.nunique()} pairs, {len(reviews)} reviews; judge: {len(judged)} pairs, "
          f"{unparsed} not parsed in both orders (left out where a criterion has no verdict).")
    print(f"Raters per pair: {dict(sorted(Counter(reviews.groupby('pair_key').size()).items()))}")

    rows = []
    items = [(c, HUMAN_LABEL[c], judged) for c in MOOVE]
    if args.overall:
        # the overall verdict as a sixth criterion: the reviews' votes against
        # the judge's, both orders under the same names
        items.append(("vote", "Overall verdict", load_judged(
            args.overall, {"judge_vote": "j_pref_vote", "winner_ab": "j_pref_vote_ab",
                           "winner_ba": "j_pref_vote_ba"})))
    for c, name, jd in items:
        both_orders = any(r.get(f"j_pref_{c}_ba") for r in jd.values())
        jv = {k: r.get(f"j_pref_{c}") for k, r in jd.items()}
        rec = {"criterion": name}
        for how in ("unanimous", "majority"):
            lab = {k: v for k, v in pair_labels(reviews, c, how).items() if jv.get(k)}
            m = metrics(lab.values(), [jv[k] for k in lab])
            rec.update({f"{how}_{k}": v for k, v in m.items()})
            if how == "unanimous" and both_orders:
                for sfx in ("ab", "ba"):
                    one = {k: jd[k].get(f"j_pref_{c}_{sfx}") for k in lab}
                    rec[f"unanimous_three_way_{sfx}"] = 100 * sum(
                        lab[k] == one[k] for k in lab) / len(lab) if lab else float("nan")
        rv = reviews[reviews[c].notna() & reviews.pair_key.map(jv).notna()]
        m = metrics(rv[c], rv.pair_key.map(jv))
        rec.update({f"review_{k}": v for k, v in m.items()})
        a, b = human_human(reviews, c)
        m = metrics(a, b)
        rec.update({"hh_n": m["n"] // 2, "hh_three_way": m["three_way"], "hh_kappa": m["kappa"]})
        if c != "vote":
            # how often a review's criterion preference is its own overall vote
            decisive = reviews[reviews[c].notna()]
            rec["matches_own_vote"] = 100 * (decisive[c] == decisive.vote).mean()
        if both_orders:
            pairs = [r for r in jd.values() if r.get(f"j_pref_{c}_ab") and r.get(f"j_pref_{c}_ba")]
            rec["abba_flips"] = 100 * sum(r[f"j_pref_{c}_ab"] != r[f"j_pref_{c}_ba"]
                                          for r in pairs) / len(pairs)
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("criterion")
    pd.set_option("display.width", 250)
    if args.length_bias:
        lengths = answer_lengths(args, reviews)
        lb = length_bias(reviews, items, lengths)
        w1, w2 = words_summary(lengths, reviews, items)
        print(f"\n=== answer length: {w1} ===\n{w2}")
        for title, cols in (
            ("Does either side follow the longer answer?",
             ["n", "n_decisive", "h_longer", "p_h_vs_50", "n_j_decisive", "j_longer",
              "p_j_vs_50", "n_both_decisive", "p_h_vs_j"]),
            ("Always pick the longer answer, as a judge",
             ["longer_rule_3way", "longer_rule_bin", "gemma_3way", "rho_h", "rho_j"]),
            ("Gemma where the physicians preferred the longer / the shorter answer",
             ["agree_h_longer_n", "agree_h_longer", "agree_h_shorter_n", "agree_h_shorter"]),
            ("Equal calls by length gap (near: word ratio 0.8-1.25)",
             ["n_near", "h_equal_near", "j_equal_near", "n_far", "h_equal_far", "j_equal_far"]),
            ("Agreement with the length gap controlled",
             ["n_near", "agree_near", "binary_near", "n_far", "agree_far", "binary_far"])):
            print(f"\n=== {title} ===")
            print(lb[cols].to_string(float_format=lambda x: f"{x:.3f}" if abs(x) < 1 else f"{x:.1f}"))
        if args.tsv:
            lb.to_csv(args.tsv.replace(".tsv", "_length.tsv"), sep="\t", float_format="%.4f")
            print(f"\nWritten to {args.tsv.replace('.tsv', '_length.tsv')}")
        return
    tabs = {
        "Pairs whose raters are unanimous on the criterion":
            ["unanimous_n", "unanimous_three_way", "unanimous_binary", "unanimous_macro_f1",
             "unanimous_kappa", "unanimous_h_equal", "unanimous_j_equal"]
            + ["unanimous_three_way_ab", "unanimous_three_way_ba", "abba_flips"],
        "Pairs labelled by the raters' majority (top ties left out)":
            ["majority_n", "majority_three_way", "majority_binary", "majority_macro_f1",
             "majority_kappa", "majority_h_equal", "majority_j_equal"],
        "Every review vs the judge, and rater vs rater":
            ["review_n", "review_three_way", "review_binary", "review_kappa",
             "hh_n", "hh_three_way", "hh_kappa", "matches_own_vote"],
    }
    for title, cols in tabs.items():
        t = df[[c for c in cols if c in df.columns]].dropna(how="all")
        print(f"\n=== {title} ===")
        print(t.to_string(float_format=lambda x: f"{x:.3f}" if abs(x) < 1 else f"{x:.1f}"))
    if args.tsv:
        df.to_csv(args.tsv, sep="\t", float_format="%.4f")
        print(f"\nWritten to {args.tsv}")


if __name__ == "__main__":
    main()
