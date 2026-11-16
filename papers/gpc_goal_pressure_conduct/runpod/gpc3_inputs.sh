#!/usr/bin/env bash
# Sourced by GPC-3's launch cards (from the repo root). CST's inputs are another paper's outputs:
# copied into GPC's own outputs (outputs/gpc3_in/cst/<path>) to sync by name, and the remote runner
# puts each back at its path on the pod. GPC-1's and GPC-2's outputs are this paper's and sync by name
# as they are. Sets SYNC (comma-separated) and the game bundle's ON_DEMAND_SYNC entry; bails on a
# missing input before any pod exists. G5's upstream code is cloned on the pod (never synced).
G=papers/gpc_goal_pressure_conduct
C=papers/cst_constitutional_steering
CST_INPUTS=(
  outputs/teacher_v1/teacher/base_good_behavior.jsonl          # the base and GOOD of record
  outputs/finalist_c2_rule_right_mach/mach_states_base.jsonl   # the state labels
  outputs/finalist_c2_rule_right_mach/mach/decisions.jsonl     # the replay
  outputs/finalist_c2_rule_right_mach/mach/episodes.jsonl      # the replay and the scale link
  outputs/_sync/eval_8gram_hashes_v1.npy                       # the eval 8-gram index
  outputs/_sync/C_anth.txt
  outputs/_sync/a0n_faraday_v1.txt
  outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors  # the fixed smoke adapter
  outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_config.json
  outputs/pilot_sdf/parity/hf_d_doc_1M.json
  outputs/pilot_sdf/parity/hf_base.json
  outputs/a52_v1/teacher/a49_pctx.jsonl                        # P-ctx's single decisions (G3)
)
GPC_INPUTS=(
  outputs/gpc1_v1/conduct/slots.json outputs/gpc1_v1/conduct/kept_c1.jsonl
  outputs/gpc1_v1/conduct/teacher_c1.jsonl outputs/gpc1_v1/conduct/base_gap.jsonl
  outputs/gpc1_v1/conduct/holdout_base.jsonl outputs/gpc1_v1/train_c1.json
  outputs/gpc1_v1/adapters/c1/612732/adapter_model.safetensors
  outputs/gpc1_v1/adapters/c1/612732/adapter_config.json
  outputs/gpc1_v1/teacher/screen_c1.jsonl
  outputs/gpc2_v1/adapters/c1_s200/1/adapter_model.safetensors  # C1 x 2.0 (G4)
  outputs/gpc2_v1/adapters/c1_s200/1/adapter_config.json
  outputs/gpc2_v1/x/b1_base.jsonl                               # B1's strata (G2, G4)
)
I="$G/outputs/gpc3_in"
rm -rf "$I" && mkdir -p "$I"
SYNC=""
stage_in() {  # <paper dir> <tag> <path...>
  local src="$1" tag="$2"; shift 2
  for f in "$@"; do
    [ -f "$src/$f" ] || { echo "BAIL: $src/$f is missing"; exit 1; }
    mkdir -p "$I/$tag/$(dirname "$f")" && cp -L "$src/$f" "$I/$tag/$f"
    SYNC="${SYNC:+$SYNC,}outputs/gpc3_in/$tag/$f"
  done
}
stage_in "$C" cst "${CST_INPUTS[@]}"
for f in "${GPC_INPUTS[@]}"; do
  [ -f "$G/$f" ] || { echo "BAIL: $G/$f is missing"; exit 1; }
  SYNC="$SYNC,$f"
done
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
GPC_BUNDLE="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
