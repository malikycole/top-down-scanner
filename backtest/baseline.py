"""Random entry baseline: buy at the open on every 5th bar, hold exactly N bars, exit at the close.
Same stocks and years as bt_scanner.py. Shows what a setup has to beat once market drift and
the survivorship of today's stock list are counted in."""
import json, pickle
import numpy as np
from bt_scanner import PRICES, HERE

frames = pickle.loads(PRICES.read_bytes())
out = {}
for N in (2, 5, 10, 20, 40, 65, 120):
    r = []
    for df in frames.values():
        o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
        if len(c) < 300 or np.any(c <= 0):
            continue
        idx = np.arange(260, len(c) - N - 1, 5)
        r.append(c[idx + N] / o[idx + 1] - 1)
    r = np.concatenate(r)
    out[N] = {"n": int(len(r)), "win": round(float((r > 0).mean() * 100), 1), "avg": round(float(r.mean() * 100), 2),
              "per_day": round(float(r.mean() * 100 / N), 3)}
    print(N, out[N])
res = json.loads((HERE / "results.json").read_text())
res["baseline"] = out
(HERE / "results.json").write_text(json.dumps(res, indent=1))
