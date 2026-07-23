import csv
from pathlib import Path
import matplotlib.pyplot as plt

path_a = Path("results/mano_fit_summary_20iter_clean.csv")
path_b = Path("results/mano_fit_summary_pathB_20iter.csv")

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

def load_counts(path):
    counts = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mudra = row["mudra"].strip()
            counts[mudra] = int(row["fit_files"])
    return counts

a = load_counts(path_a)
b = load_counts(path_b)

mudras = sorted(set(a) | set(b))
a_counts = [a.get(m, 0) for m in mudras]
b_counts = [b.get(m, 0) for m in mudras]
gains = [b.get(m, 0) - a.get(m, 0) for m in mudras]

# Chart 1: Path A vs Path B valid sample counts
x = range(len(mudras))
width = 0.4

plt.figure(figsize=(18, 8))
plt.bar([i - width / 2 for i in x], a_counts, width, label="Path A Strict")
plt.bar([i + width / 2 for i in x], b_counts, width, label="Path B Improved")
plt.xticks(list(x), mudras, rotation=90)
plt.ylabel("Valid MANO Fit Files")
plt.title("Path A vs Path B Valid MANO Fit Counts per Mudra")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "pathA_vs_pathB_valid_counts.png", dpi=200)
plt.close()

# Chart 2: Biggest Path B gains
gain_pairs = sorted(zip(mudras, gains), key=lambda x: x[1], reverse=True)
top = gain_pairs[:20]
top_mudras = [m for m, g in top]
top_gains = [g for m, g in top]

plt.figure(figsize=(14, 7))
plt.bar(top_mudras, top_gains)
plt.xticks(rotation=75)
plt.ylabel("Additional Usable Samples")
plt.title("Top 20 Mudra Classes Improved by Path B")
plt.tight_layout()
plt.savefig(out_dir / "top20_pathB_gains.png", dpi=200)
plt.close()

print("Saved figures:")
print(out_dir / "pathA_vs_pathB_valid_counts.png")
print(out_dir / "top20_pathB_gains.png")

print("\nOverall:")
print("Path A total:", sum(a_counts))
print("Path B total:", sum(b_counts))
print("Gain:", sum(b_counts) - sum(a_counts))
