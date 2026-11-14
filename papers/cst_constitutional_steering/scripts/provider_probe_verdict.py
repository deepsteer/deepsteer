#!/usr/bin/env python3
"""Amendment 51 item A9's verdict line from a probe's pulled records (zero GPU, local).

    /opt/homebrew/bin/python3.13 provider_probe_verdict.py DIR PRICE_PER_HOUR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provider_probe as PP  # noqa: E402

if __name__ == "__main__":
    d, price = Path(sys.argv[1]), float(sys.argv[2])
    recs = {f.stem: json.loads(f.read_text()) for f in d.glob("*.json")}
    v = PP.verdict(recs, price)
    (d / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    print(json.dumps(v, indent=1))
