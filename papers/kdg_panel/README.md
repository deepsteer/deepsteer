# KDG panel (knowing–doing gap)

Spec of record: `../KDG_PANEL_SPEC.md` (v0.4). Library: `deepsteer/kdg/` (schema, harness,
breadth rubric, statistics, calibration). This directory holds the scenario data, the API
scripts, the pod driver, and the pod-boundary documents.

```
data/pilot_scenarios_<half>_<generator>.json   scenario set of record (both frames, twins, tags)
data/calibration_set_v1.json                   200-item harness calibration set (stage 1)
scripts/generate_scenarios.py                  §8.2 generation via API (structured outputs)
scripts/build_calibration_set.py               §8.3 calibration set + harness check
scripts/rate_with_judge.py                     non-generator external labels; rater 2; breadth
scripts/pod_kdg_pilot.py + kdg_pod_lib.py      pod driver (dry-run stub; per-rollout saves)
scripts/analyze_pilot.py                       zero-GPU screen / gate / ladder / base cell
runpod/remote_kdg_pilot.sh                     remote runner (VALIDATE=1 first)
models.yaml                                    pinned model ids, readout conventions, rollouts
LIT_PASS.md                                    §8.1 verified citations + novelty re-centering
PILOT_SESSION.md                               pod-boundary checklist, power table, review
```

Tests: `pytest tests/kdg tests/scripts/test_pod_kdg_pilot.py`.
