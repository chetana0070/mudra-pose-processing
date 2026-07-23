import csv
from collections import defaultdict

manifest_path = "data/landmarks/manifest.csv"

counts = defaultdict(lambda: defaultdict(int))

with open(manifest_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mudra = row.get("mudra", "UNKNOWN")
        status = row.get("status", "UNKNOWN")
        counts[mudra][status] += 1

total_all = 0
ok_all = 0

print("\nPer-mudra landmark extraction status:\n")

for mudra, statuses in sorted(counts.items()):
    total = sum(statuses.values())
    ok = statuses.get("ok", 0)

    total_all += total
    ok_all += ok

    print(f"{mudra:20s} ok={ok:3d}/{total:<3d}  {dict(statuses)}")

print("\n" + "-" * 70)
print(f"TOTAL ok={ok_all}/{total_all}  success_rate={ok_all / total_all:.2%}")
