#!/usr/bin/env python3
"""
aksara-data — Constitutional AI-style Self-Critique & Revision

Implements the core loop from Anthropic's Constitutional AI method
(Bai et al., 2022, "Constitutional AI: Harmlessness from AI Feedback",
https://arxiv.org/abs/2212.08073): for each draft response, a model critiques
it against a written principle, then revises it to better satisfy that
principle. Chaining this over a set of principles produces higher-quality
SFT data without needing a human label for every single issue.

The principle set ("constitution") lives in generators/constitutions/*.json
— plain data, not hardcoded in this script. Maintainers own and edit it
directly to decide what "better" means for this project; this file only
implements the generic critique -> revise loop, it has no opinion baked in
about what the principles should say. See constitutions/default.json for the
starting set (mostly about being more direct/helpful and not over-refusing
on legal topics, plus one narrow safety limit).

Usage:
    python3 generators/constitutional_revision.py \\
        --input drafts.jsonl \\
        --constitution generators/constitutions/default.json \\
        --backend huggingface --model AksaraLLM/aksarallm-mini \\
        --out revised.jsonl

`--input` is JSONL with {"prompt": ..., "response": ...} per line (e.g. raw
SFT drafts from generators/autopilot_v2.py or generators/multiturn_cot.py).

The generation backend is pluggable (see `Backend` below) — bring your own
`generate(prompt: str) -> str` by subclassing it; `HFBackend` (any local/HF
causal LM, including an AksaraLLM checkpoint exported via
`aksarallm.hf_export`) is provided as the reference implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Callable, Iterable, Iterator

CRITIQUE_TEMPLATE = """Berikut adalah sebuah percakapan:

Pertanyaan pengguna: {prompt}

Jawaban asisten: {response}

Prinsip yang harus dipatuhi: {principle}

Apakah jawaban di atas melanggar prinsip tersebut? Jika ya, jelaskan secara singkat pelanggarannya. Jika tidak ada pelanggaran, jawab persis dengan: "Tidak ada masalah."
"""

REVISION_TEMPLATE = """Berikut adalah sebuah percakapan:

Pertanyaan pengguna: {prompt}

Jawaban asisten (draf): {response}

Kritik: {critique}

Prinsip yang harus dipatuhi: {principle}

Tulis ulang jawaban asisten agar mematuhi prinsip tersebut, sambil tetap menjawab pertanyaan pengguna selengkap dan sesubstantif mungkin. Tulis HANYA jawaban yang sudah direvisi, tanpa komentar tambahan."""

NO_ISSUE_MARKERS = ("tidak ada masalah", "tidak ada pelanggaran", "sudah sesuai")


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_constitution(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    principles = data.get("principles", [])
    if not principles:
        raise ValueError(f"No principles found in {path}")
    return principles


def _has_no_issue(critique: str) -> bool:
    c = critique.strip().lower()
    return any(marker in c for marker in NO_ISSUE_MARKERS)


def critique_and_revise(
    prompt: str,
    draft_response: str,
    principles: list[dict],
    generate_fn: Callable[[str], str],
    max_principles: int | None = None,
) -> dict:
    """Run one critique-then-revise pass per principle, chaining revisions
    (each principle sees the output of the previous one)."""
    response = draft_response
    applied = []

    chosen = principles[:max_principles] if max_principles else principles
    for p in chosen:
        critique_prompt = CRITIQUE_TEMPLATE.format(prompt=prompt, response=response, principle=p["principle"])
        critique = generate_fn(critique_prompt).strip()

        if _has_no_issue(critique):
            continue

        revise_prompt = REVISION_TEMPLATE.format(
            prompt=prompt, response=response, critique=critique, principle=p["principle"]
        )
        revised = generate_fn(revise_prompt).strip()
        if revised:
            response = revised
            applied.append({"principle": p["name"], "critique": critique})

    return {
        "prompt": prompt,
        "original_response": draft_response,
        "revised_response": response,
        "principles_applied": applied,
        "was_revised": response != draft_response,
    }


# ══════════════════════════════════════════════════════════════════
#  Generation backends — pluggable, so this script isn't tied to any
#  one model/provider.
# ══════════════════════════════════════════════════════════════════

class Backend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class HFBackend(Backend):
    """Any local/HF causal LM — including an AksaraLLM checkpoint exported
    via aksarallm.hf_export. Reference implementation; swap in an API-backed
    Backend subclass if you'd rather use a stronger external judge model."""

    def __init__(self, model_id: str, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
#  I/O + CLI
# ══════════════════════════════════════════════════════════════════

def iter_drafts(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run(drafts: Iterable[dict], principles: list[dict], generate_fn, out_path: str, max_principles=None) -> None:
    n_revised = 0
    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in drafts:
            prompt = item.get("prompt") or item.get("instruction", "")
            response = item.get("response") or item.get("output", "")
            if not prompt or not response:
                continue

            result = critique_and_revise(prompt, response, principles, generate_fn, max_principles)
            total += 1
            if result["was_revised"]:
                n_revised += 1

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if total % 20 == 0:
                log(f"  {total} processed, {n_revised} revised so far")

    log(f"Done: {total} processed, {n_revised} revised ({n_revised / max(total, 1) * 100:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Constitutional AI-style self-critique & revision for SFT data")
    ap.add_argument("--input", required=True, help="JSONL with {prompt, response} drafts")
    ap.add_argument("--constitution", default="generators/constitutions/default.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["huggingface"], default="huggingface")
    ap.add_argument("--model", help="Model id/path for --backend huggingface")
    ap.add_argument("--max-principles", type=int, default=None, help="Limit principles applied per example (default: all)")
    args = ap.parse_args(argv)

    principles = load_constitution(args.constitution)
    log(f"Loaded {len(principles)} principles from {args.constitution}")

    if args.backend == "huggingface":
        if not args.model:
            ap.error("--model is required for --backend huggingface")
        backend = HFBackend(args.model)
    else:
        raise AssertionError("unreachable")

    run(iter_drafts(args.input), principles, backend.generate, args.out, args.max_principles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
