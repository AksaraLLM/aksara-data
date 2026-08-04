# 📚 aksara-data

Data curation pipeline for AksaraLLM — 100% transparent, 100% reproducible.

## Data Statistics

| Dataset | Samples | HuggingFace |
|---|---|---|
| SFT v5 | 500,149 | `AksaraLLM/aksara-mega-sft-v5` |
| DPO v4 | 200,000 | `AksaraLLM/aksara-dpo-id-v4` |
| Multi-turn + CoT | 1,657 | `AksaraLLM/aksara-v3-multiturn-cot` |

## Pipeline

```
generators/autopilot_v2.py            → Discover + extract + augment SFT data
generators/multiturn_cot.py           → Multi-turn conversations + chain-of-thought
generators/constitutional_revision.py → Self-critique & revision (Constitutional AI-style)
quality/auditor.py                    → Fuzzy dedup, quality scoring, rebalancing
quality/decontaminate.py              → Remove pretraining docs that overlap aksara-eval benchmarks
translate_v2.py / *_fast.py / *_pipeline.py / *_range.py
                                       → Anthropic/hh-rlhf → Indonesian (Helsinki-NLP/opus-mt-en-id)
```

## Quick Start
```bash
# Generate more multi-turn + CoT data
python3 generators/multiturn_cot.py

# Audit data quality
python3 quality/auditor.py

# Decontaminate a pretraining corpus against aksara-eval's benchmarks
# (13-gram overlap — same method GPT-3/PaLM/LLaMA report in their papers)
python3 quality/decontaminate.py --corpus /data/corpus_20b/cleaned/ --report-only

# Improve SFT drafts via self-critique + revision against a written principle
# set (see generators/constitutions/default.json — edit it to change what
# "better" means for this project; it's plain data, not hardcoded)
python3 generators/constitutional_revision.py \
    --input drafts.jsonl --model AksaraLLM/aksarallm-mini --out revised.jsonl
```

## License
Apache 2.0
