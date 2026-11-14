"""CST Amendment 51's provider probes: papers/cst_constitutional_steering/scripts/provider_probe.py,
hyperstack_probe.py and runpod/remote_provider_probe.sh."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts")]

import hyperstack_probe as HP  # noqa: E402
import provider_probe as PP  # noqa: E402

RUNPOD = ROOT / "papers/cst_constitutional_steering/runpod"


def test_worst_window_excludes_the_warmup_and_catches_a_dip():
    # most probable failure (Amendment 51 item A6): the worst 5-minute window is taken over the
    # warm-up, or a mid-run throttle is averaged away by the mean
    t0 = 1000.0
    ev = [(t0 + 60 * m + 1, 600 * 60 if m not in (10, 11) else 100 * 60) for m in range(20)]
    ev[0] = (t0 + 1, 0)  # minute 0 is warm-up
    w = PP.windows(t0, ev, 20)
    assert abs(w["mean_tps"] - (16 * 600 + 2 * 100) / 18) < 1e-9
    assert abs(w["worst_5min_tps"] - (3 * 600 + 2 * 100) / 5) < 1e-9 and w["worst_over_mean"] < 0.8


def test_verdict_thresholds_are_the_amendments():
    # most probable failure (Amendment 51 item C): the break-even throughput is inverted, so a
    # provider at $1.09 reads as beating Hyperstack only above 124% of RunPod's throughput
    recs = {"stack": {"ok": True}, "gate_tests": {"ok": True},
            "smoke_serve": {"parity_passes": True},
            "hf_fixed": {"tv_base_vs_saved": 0.0},
            "tput_plain": {"relative_to_runpod": 0.9, "worst_over_mean": 0.95},
            "tput_doc": {"relative_to_runpod": 0.85, "worst_over_mean": 0.9}}
    v = PP.verdict(recs, 1.09)
    assert abs(v["thresholds"]["beats_hyperstack_above"] - 0.807) < 0.001
    assert abs(v["thresholds"]["beats_runpod_secure_above"] - 0.522) < 0.001
    assert abs(v["effective_price"] - 1.09 / 0.85) < 1e-9 and v["passes"]
    recs["tput_doc"]["worst_over_mean"] = 0.7
    assert not PP.verdict(recs, 1.09)["passes"]


def test_the_probe_builds_the_pods_venv_exactly_as_pinned():
    # most probable failure (Amendment 51 item A2): the probe's venv drifts from the pods' pins, so
    # a pass on the provider says nothing about the stack the registered reads run on
    pins = re.compile(r'"(torch==[^"]+|torchvision==[^"]+|torchaudio==[^"]+|transformers==[^"]+|'
                      r'peft==[^"]+)"|VLLM_WHEEL_(?:URL|SHA256)=\S+')
    probe = (RUNPOD / "remote_provider_probe.sh").read_text()
    pod = (RUNPOD / "remote_cst_teacher.sh").read_text()
    assert sorted(pins.findall(probe)) == sorted(pins.findall(pod)) and pins.findall(probe)
    assert 'STOP (A1): Python' in probe and 'STOP (A2)' in probe


def test_hyperstack_summary_flags_us_or_canada_stock():
    # most probable failure (Amendment 51 item B3): an EU-only, spot or out-of-stock A100 counts as
    # the dedicated-GPU provider's on-demand stock for the partner's document
    fl = {"data": [{"gpu": "A100-80G-PCIe", "region_name": "CANADA-1", "flavors": [
        {"name": "n3-A100x1", "gpu": "A100-80G-PCIe", "gpu_count": 1, "region_name": "CANADA-1",
         "stock_available": False}]},
        {"gpu": "A100-80G-PCIe", "region_name": "NORWAY-1", "flavors": [
            {"name": "n3-A100x1", "gpu": "A100-80G-PCIe", "gpu_count": 1,
             "region_name": "NORWAY-1", "stock_available": True}]}]}
    s = HP.summarize({"regions": []}, fl, {"stocks": []}, {"images": []}, [], {"data": {}})
    assert len(s["single_a100_flavors"]) == 2 and not s["us_or_canada_a100_80gb_on_demand_in_stock"]
    fl["data"][0]["flavors"].append({"name": "n3-A100x1-spot", "gpu": "A100-80G-PCIe-spot",
                                     "gpu_count": 1, "region_name": "CANADA-1",
                                     "stock_available": True})  # spot is not on-demand stock
    s = HP.summarize({"regions": []}, fl, {"stocks": []}, {"images": []}, [], {"data": {}})
    assert not s["us_or_canada_a100_80gb_on_demand_in_stock"]
    fl["data"][0]["flavors"][0]["stock_available"] = True
    s = HP.summarize({"regions": []}, fl, {"stocks": []}, {"images": []}, [], {"data": {}})
    assert s["us_or_canada_a100_80gb_on_demand_in_stock"]
