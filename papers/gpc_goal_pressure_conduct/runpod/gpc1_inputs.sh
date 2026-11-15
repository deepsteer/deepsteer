#!/usr/bin/env bash
# Sourced by GPC-1's launch cards (from the repo root): copies the CST inputs GPC-1 reads into
# GPC's own outputs (outputs/gpc1_in/cst/<the CST path>), so they sync by name (SYNC_OUTPUTS
# carries only the launching paper's outputs, and named files pass the blob excludes), and the
# remote runner puts each back at its CST path on the pod. Sets SYNC (comma-separated) and the
# game bundle's ON_DEMAND_SYNC entry. Bails on a missing input before any pod exists.
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
)
I="$G/outputs/gpc1_in/cst"
rm -rf "$I" && mkdir -p "$I"
SYNC=""
for f in "${CST_INPUTS[@]}"; do
  [ -f "$C/$f" ] || { echo "BAIL: $C/$f is missing"; exit 1; }
  mkdir -p "$I/$(dirname "$f")" && cp -L "$C/$f" "$I/$f"
  SYNC="${SYNC:+$SYNC,}outputs/gpc1_in/cst/$f"
done
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
GPC_BUNDLE="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
