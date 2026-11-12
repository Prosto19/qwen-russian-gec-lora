"""Audit recorded JSON outputs; does not load or run a model."""
import json
from pathlib import Path

def audit(root):
    stages = [("Base", "baseline_gec.json", "predicted", "exact_match"),
              ("LoRA", "lora_gec_results.json", "lora", "exact_match"),
              ("Merged", "merged_gec_results.json", "merged", "merged_exact_match"),
              ("F16", "f16_gec_results.json", "f16", "f16_exact_match"),
              ("Q4", "q4_gec_results.json", "q4", "q4_exact_match")]
    baseline = None
    counts = {}
    for label, filename, prediction, flag in stages:
        rows = json.loads((root / "results" / filename).read_text(encoding="utf-8"))
        ids = [r["test_index"] for r in rows]
        if len(rows) != 20 or len(set(ids)) != 20:
            raise ValueError(f"{label}: expected 20 unique samples")
        identity = [(r["test_index"], r["input"], r["expected"]) for r in rows]
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError(f"{label}: samples or references differ")
        matches = [r[prediction].strip() == r["expected"].strip() for r in rows]
        if any(type(r[flag]) is not bool or r[flag] != m for r, m in zip(rows, matches)):
            raise ValueError(f"{label}: stored flags disagree with strings")
        counts[label] = sum(matches)
    if list(counts.values()) != [0, 6, 6, 6, 3]:
        raise ValueError(f"README / figure needs updating: {counts}")
    return counts

if __name__ == "__main__":
    for name, count in audit(Path(__file__).resolve().parents[1]).items():
        print(f"{name}: {count}/20 ({count*5}%)")
    print("PASS: archived strings, IDs and flags agree. Not live inference.")
