"""Collects results/*.json into results/summary.md (a table for the README)."""
import glob, json
rows = [json.load(open(f)) for f in sorted(glob.glob("results/*_*.json"))]
lines = ["| Model | Augmentation | Parameters | Test accuracy | Train time (s) |", "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['model']} | {'yes' if r['augmentation'] else 'no'} | {r['parameters']:,} | "
                 f"{r['final_test_accuracy']:.2f}% | {r['train_seconds']} |")
open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
