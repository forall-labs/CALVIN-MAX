"""Average chain length of a Base run from its traces (works mid-run): python scripts/probe/avg_len.py results/base/<run>"""
import glob, pickle, sys, statistics
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
L = []
for p in glob.glob(f"{sys.argv[1]}/traces/*.pkl"):
    s = pickle.load(open(p, "rb")).subtask_success
    n = 0
    for x in s:
        if not x: break
        n += 1
    L.append(n)
m = statistics.mean(L); se = statistics.stdev(L) / len(L) ** 0.5
print(f"n={len(L)} avg_len={m:.3f} se={se:.3f} sr@k={[round(sum(l >= k for l in L) / len(L), 3) for k in range(1, 6)]}")
