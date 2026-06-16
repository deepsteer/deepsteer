#!/usr/bin/env python3
"""Sprint 0.5: persona-feature probe + direction extraction on a base model.

Runs the persona minimal-pair probe across all layers and saves BOTH per-layer
accuracy (mirroring ``PersonaFeatureProbe`` / ``LayerProbingResult``) AND the
per-layer probe-weight + mean-diff persona directions as NPZ, so Sprint 0.6
(``persona_morality_angles.py``) and the pipeline study can reuse them.

Methodology matches the moral probe: mean-pooled hidden states, seeded
``nn.Linear`` BCE probe (50 epochs). In a base model the persona direction is
expected to be weak (accuracy near chance at most layers).

Usage:
    python papers/5_moral_alignment/scripts/persona_probe_base.py \
        --model allenai/Olmo-3-1025-7B \
        --output-dir papers/5_moral_alignment/outputs/olmo3_base \
        --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402

logger = logging.getLogger(__name__)
_ONSET = 0.6


def summarise(accs: dict[int, float], n_layers: int, onset: float = _ONSET) -> dict:
    """onset/peak/depth/breadth, matching GeneralLinearProbe._build_result."""
    onset_layer = next((L for L in sorted(accs) if accs[L] >= onset), None)
    peak_layer = max(accs, key=accs.get)
    n_above = sum(1 for v in accs.values() if v >= onset)
    return {
        "onset_layer": onset_layer,
        "onset_threshold": onset,
        "peak_layer": peak_layer,
        "peak_accuracy": round(accs[peak_layer], 4),
        "encoding_depth": round((onset_layer / n_layers) if onset_layer is not None else 1.0, 4),
        "encoding_breadth": round(n_above / n_layers, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Persona probe + direction extraction.")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output-dir", default="papers/5_moral_alignment/outputs/olmo3_base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--input-format", choices=["raw", "chat"], default="raw")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.persona_pairs import get_persona_dataset

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pairs, test_pairs = get_persona_dataset(
        test_fraction=args.test_fraction, seed=args.split_seed, stratified=True
    )
    print(f"Persona dataset: {len(train_pairs)} train, {len(test_pairs)} test pairs")

    t0 = time.time()
    model = WhiteBoxModel(
        args.model, device=args.device, access_tier=AccessTier.WEIGHTS,
        revision=args.revision,
    )
    n_layers = model.info.n_layers
    print(f"Loaded {args.model} in {time.time()-t0:.1f}s ({n_layers} layers)")

    directions, accs = du.extract_pair_directions(
        model, train_pairs, test_pairs=test_pairs, input_format=args.input_format,
    )
    model.release()

    summary = summarise(accs, n_layers)
    print(f"  peak persona accuracy: {summary['peak_accuracy']:.1%} @ layer "
          f"{summary['peak_layer']}  (onset {summary['onset_layer']}, "
          f"breadth {summary['encoding_breadth']})")

    # ---- save accuracy JSON ----
    result = {
        "benchmark": "persona_feature_probe",
        "model": args.model,
        "revision": args.revision,
        "input_format": args.input_format,
        "n_layers": n_layers,
        "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "per_layer_accuracy": {str(L): round(accs[L], 4) for L in sorted(accs)},
        **summary,
    }
    with open(out / "persona_probing.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- save directions NPZ (probe -> "persona", mean-diff -> "persona_meandiff") ----
    du.save_directions(
        out / "persona_directions.npz",
        {"persona": directions["probe"], "persona_meandiff": directions["mean_diff"]},
    )
    print(f"Wrote {out/'persona_probing.json'} and {out/'persona_directions.npz'}")


if __name__ == "__main__":
    main()
