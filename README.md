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
generators/autopilot_v2.py  → Discover + extract + augment SFT data
generators/multiturn_cot.py → Multi-turn conversations + chain-of-thought
quality/auditor.py          → Fuzzy dedup, quality scoring, rebalancing
```

## Quick Start
```bash
# Generate more multi-turn + CoT data
python3 generators/multiturn_cot.py

# Audit data quality
python3 quality/auditor.py
```

## License
Apache 2.0
