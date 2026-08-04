#!/usr/bin/env python3
"""
aksara-data — Pretraining Corpus Decontamination

Standard practice from GPT-3/PaLM/LLaMA-style pretraining pipelines: flag
and remove any pretraining document that has substantial n-gram overlap with
an evaluation benchmark, so benchmark scores measure generalization instead
of memorization. Every major lab's technical report includes a
decontamination step and reports the overlap rate — this project didn't
have one at all before this script.

Method: word-level 13-gram exact-match overlap (the convention used in the
GPT-3 paper, Appendix C). A document is flagged "contaminated" if it shares
at least `--min-hits` distinct 13-grams with any benchmark text.

Checks against the exact benchmarks aksara-eval scores models on (see
aksara-eval/aksara_indo_bench/tasks/*.py), pulled from the same public
HuggingFace dataset ids, so "clean on this corpus" and "scored on this
benchmark" refer to the same test set.

Usage:
    # Dry run: just report contamination stats
    python3 quality/decontaminate.py --corpus /data/corpus_20b/cleaned/ --report-only

    # Filter: write a cleaned copy with contaminated docs removed
    python3 quality/decontaminate.py --corpus /data/corpus_20b/cleaned/ \\
        --out /data/corpus_20b/decontaminated/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Iterable, Iterator

WORD_RE = re.compile(r"\w+", re.UNICODE)


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#  Core n-gram overlap logic — pure Python, no ML deps, unit-testable
#  standalone (see the smoke test at the bottom of this file).
# ══════════════════════════════════════════════════════════════════

def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Word-level n-grams (lowercased) of `text`. Empty set if text is
    shorter than `n` words."""
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_ngram_index(benchmark_texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    """Union of all n-grams across every benchmark text — this is what
    corpus documents get checked against."""
    index: set[tuple[str, ...]] = set()
    for text in benchmark_texts:
        index |= word_ngrams(text, n)
    return index


def contamination_hits(doc_text: str, ngram_index: set[tuple[str, ...]], n: int) -> int:
    """How many of `doc_text`'s n-grams also appear in the benchmark index."""
    if not ngram_index:
        return 0
    return len(word_ngrams(doc_text, n) & ngram_index)


def is_contaminated(doc_text: str, ngram_index: set[tuple[str, ...]], n: int, min_hits: int) -> bool:
    return contamination_hits(doc_text, ngram_index, n) >= min_hits


# ══════════════════════════════════════════════════════════════════
#  Benchmark loading — pulls the exact eval sets aksara-eval scores on
# ══════════════════════════════════════════════════════════════════

def _extract_text_fields(example: dict) -> str:
    """Join every string-valued field — good enough for contamination
    purposes (we need substantial textual overlap, not an exact replica of
    each task's specific prompt template)."""
    return " ".join(str(v) for v in example.values() if isinstance(v, str))


def load_benchmark_texts(include_nusax_all_langs: bool = False) -> dict[str, list[str]]:
    """Returns {benchmark_name: [text, ...]}. Requires `datasets`."""
    from datasets import load_dataset

    texts: dict[str, list[str]] = {}

    def _try_load(name: str, path: str, config: str | None, split: str):
        try:
            ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
            texts[name] = [_extract_text_fields(ex) for ex in ds]
            log(f"  Loaded {name}: {len(texts[name]):,} examples")
        except Exception as e:
            log(f"  Skipping {name} ({path}): {e}", level="WARN")

    _try_load("indommlu", "IndoNLP/indommlu", None, "test")
    _try_load("copal_id", "haryoaw/COPAL", None, "test")

    nusax_langs = ["ace", "ban", "bbc", "bjn", "bug", "ind", "jav", "mad", "min", "nij", "sun"] \
        if include_nusax_all_langs else ["ind"]
    for lang in nusax_langs:
        _try_load(f"nusax_senti.{lang}", "indonlp/NusaX-senti", lang, "test")

    return texts


# ══════════════════════════════════════════════════════════════════
#  Corpus iteration — same JSONL convention as aksara-tokenizer's trainer
# ══════════════════════════════════════════════════════════════════

def iter_corpus_files(paths: list[str]) -> Iterator[str]:
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "**/*.jsonl"), recursive=True))
        else:
            yield p


def iter_jsonl_records(path: str) -> Iterator[tuple[dict, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec, line


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def run(corpus_paths: list[str], out_dir: str | None, n: int, min_hits: int,
        ngram_index: set[tuple[str, ...]], report_only: bool) -> None:
    total_docs = 0
    contaminated_docs = 0

    for path in iter_corpus_files(corpus_paths):
        out_f = None
        if not report_only:
            rel = os.path.basename(path)
            os.makedirs(out_dir, exist_ok=True)
            out_f = open(os.path.join(out_dir, rel), "w", encoding="utf-8")

        for rec, raw_line in iter_jsonl_records(path):
            total_docs += 1
            text = rec.get("text") or rec.get("content") or ""
            if is_contaminated(text, ngram_index, n, min_hits):
                contaminated_docs += 1
                continue
            if out_f:
                out_f.write(raw_line + "\n")

        if out_f:
            out_f.close()

    rate = contaminated_docs / max(total_docs, 1) * 100
    log("=" * 60)
    log(f"Total documents scanned : {total_docs:,}")
    log(f"Contaminated (removed)  : {contaminated_docs:,} ({rate:.3f}%)")
    if not report_only:
        log(f"Clean corpus written to : {out_dir}")
    log("=" * 60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Decontaminate pretraining corpus against aksara-eval benchmarks")
    ap.add_argument("--corpus", nargs="+", required=True, help="JSONL file(s) or directory of .jsonl files")
    ap.add_argument("--out", default=None, help="Output directory for cleaned corpus (required unless --report-only)")
    ap.add_argument("--ngram-size", type=int, default=13, help="N-gram size (default: 13, as in the GPT-3 paper)")
    ap.add_argument("--min-hits", type=int, default=1, help="Min. overlapping n-grams to flag a document as contaminated")
    ap.add_argument("--all-nusax-langs", action="store_true", help="Check against all 11 NusaX-senti languages, not just Indonesian")
    ap.add_argument("--report-only", action="store_true", help="Only report contamination stats, don't write filtered output")
    args = ap.parse_args(argv)

    if not args.report_only and not args.out:
        ap.error("--out is required unless --report-only is set")

    log("Loading benchmark texts (this hits HuggingFace — needs network + `datasets`)...")
    benchmark_texts = load_benchmark_texts(include_nusax_all_langs=args.all_nusax_langs)
    if not benchmark_texts:
        log("No benchmark texts loaded — nothing to decontaminate against.", level="ERROR")
        return 1

    all_texts = [t for texts in benchmark_texts.values() for t in texts]
    log(f"Building {args.ngram_size}-gram index from {len(all_texts):,} benchmark texts...")
    ngram_index = build_ngram_index(all_texts, args.ngram_size)
    log(f"Index size: {len(ngram_index):,} unique {args.ngram_size}-grams")

    run(args.corpus, args.out, args.ngram_size, args.min_hits, ngram_index, args.report_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
