#!/usr/bin/env python3
"""Hyperstack's availability probe (CST Amendment 51 item B): read-only GET calls, zero GPU.

    HYPERSTACK_API_KEY=... /opt/homebrew/bin/python3.13 hyperstack_probe.py

The author runs it twice, a day apart; each run writes
``outputs/hyperstack_probe/<UTC stamp>.json`` (the raw responses less anything account-identifying)
and prints the summary: the regions; every single-A100-80GB flavor with its region and
``stock_available``; the stock endpoint's A100 80GB rows; the Ubuntu 22.04 CUDA images with their
driver; the pricebook's A100 rows; the account's credit. Endpoints and the ``api_key`` header from
Hyperstack's API reference (docs.hyperstack.cloud, read 2026-11-14). Only GET; the key is read from
the environment and is never written.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://infrahub-api.nexgencloud.com/v1"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "hyperstack_probe"
US_CA = ("US-1", "CANADA-1", "CANADA-2")


def get(path: str, key: str, **params) -> dict | list:
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    req = urllib.request.Request(BASE + path + q, headers={"api_key": key,
                                                           "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (a fixed https host)
        return json.loads(r.read().decode())


def a100_80(name: str) -> bool:
    return bool(re.search(r"A100", name or "")) and bool(re.search(r"80", name or ""))


def summarize(regions, flavors, stocks, images, pricebook, credit) -> dict:
    flav = []
    for grp in (flavors.get("data") or []):
        for f in grp.get("flavors", [grp]) if isinstance(grp, dict) else []:
            gpu = f.get("gpu") or ""
            if "A100" in gpu and f.get("gpu_count") == 1:
                flav.append({k: f.get(k) for k in ("name", "gpu", "gpu_count", "region_name",
                                                   "stock_available", "disk", "ephemeral")})
    st = [{"region": s.get("region"), **{k: m.get(k) for k in ("model", "available",
                                                                "configurations")}}
          for s in (stocks.get("stocks") or []) for m in (s.get("models") or [])
          if "A100" in (m.get("model") or "")]
    imgs = []
    for grp in (images.get("images") or []):
        for im in grp.get("images", [grp]) if isinstance(grp, dict) else []:
            n = im.get("name") or ""
            if "22.04" in n and "CUDA" in n.upper():
                imgs.append({"region": grp.get("region_name") or im.get("region_name"), "name": n,
                             "version": im.get("version")})
    pb = [p for p in (pricebook if isinstance(pricebook, list) else [])
          if "A100" in (p.get("name") or "")]
    # on-demand only (Amendment 51 item B3): a spot flavor can be preempted mid-pod
    us_ca_stock = [f for f in flav if f["region_name"] in US_CA and f.get("stock_available")
                   and "spot" not in f"{f.get('name')}{f.get('gpu')}".lower()]
    return {"regions": [{k: r.get(k) for k in ("name", "country")}
                        for r in (regions.get("regions") or [])],
            "single_a100_flavors": flav, "a100_stock_rows": st,
            "ubuntu_2204_cuda_images": imgs, "pricebook_a100": pb,
            "credit": {k: (credit.get("data") or {}).get(k)
                       for k in ("credit", "threshold", "can_create_instance")},
            "us_or_canada_a100_80gb_on_demand_in_stock": bool(us_ca_stock),
            "note": ("no US or Canada region shows on-demand A100 80GB stock on this run (B3: the "
                     "dedicated-GPU fallback is Verda or RunPod Secure)") if not us_ca_stock else
                    "a US or Canada on-demand single-A100 flavor shows stock on this run"}


def main() -> None:
    key = os.environ.get("HYPERSTACK_API_KEY")
    if not key:
        sys.exit("BAIL: HYPERSTACK_API_KEY is not set")
    raw = {"regions": get("/core/regions", key), "flavors": get("/core/flavors", key),
           "stocks": get("/core/stocks", key),
           "images": get("/core/images", key, include_public="true", search="Ubuntu"),
           "pricebook": get("/pricebook", key),
           "credit": get("/billing/user-credit/credit", key)}
    s = summarize(**raw)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (OUT / f"{stamp}.json").write_text(json.dumps({"at": stamp, "summary": s, "raw": {
        k: v for k, v in raw.items() if k != "credit"}}, indent=1) + "\n")
    print(json.dumps(s, indent=1))
    print(f">> written: {OUT / (stamp + '.json')}")


if __name__ == "__main__":
    main()
