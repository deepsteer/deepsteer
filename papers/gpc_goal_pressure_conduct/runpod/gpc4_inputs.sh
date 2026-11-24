#!/usr/bin/env bash
# Sourced by GPC-4's pod card (from the repo root). CST's inputs are another paper's outputs: copied into
# GPC's own outputs (outputs/gpc4_in/cst/<path>) to sync by name, and the remote runner puts each back
# at its path on the pod. GPC-1's outputs and GPC-4's own (the v2 8-gram index, the VM's two public
# keys and the session, written by launch_gpc4.sh before the pod starts) sync by name as they are.
# Sets SYNC and the game bundle's ON_DEMAND_SYNC entry; bails on a missing input before any pod exists.
# Upstream code and data (ODCV-Bench, ImpossibleBench's data, agentic-misalignment) are fetched on the
# pod at their pinned commits, never synced or committed.
G=papers/gpc_goal_pressure_conduct
C=papers/cst_constitutional_steering
CST_INPUTS=(
  outputs/teacher_v1/teacher/base_good_behavior.jsonl          # the base and GOOD of record
  outputs/finalist_c2_rule_right_mach/mach_states_base.jsonl   # the state labels
  outputs/finalist_c2_rule_right_mach/mach/decisions.jsonl     # the replay
  outputs/finalist_c2_rule_right_mach/mach/episodes.jsonl
  outputs/_sync/eval_8gram_hashes_v1.npy
  outputs/_sync/C_anth.txt
  outputs/_sync/a0n_faraday_v1.txt
  outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors  # the fixed smoke adapter
  outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_config.json
  outputs/pilot_sdf/parity/hf_d_doc_1M.json
  outputs/pilot_sdf/parity/hf_base.json
  outputs/pressure/adapters/p_ctx/1.94053e+06/adapter_model.safetensors  # P-ctx (the summed adapter)
  outputs/pressure/adapters/p_ctx/1.94053e+06/adapter_config.json
)
GPC_INPUTS=(
  outputs/gpc1_v1/conduct/slots.json outputs/gpc1_v1/conduct/holdout_base.jsonl
  outputs/gpc1_v1/train_c1.json
  outputs/gpc1_v1/adapters/c1/612732/adapter_model.safetensors
  outputs/gpc1_v1/adapters/c1/612732/adapter_config.json
  outputs/_sync/eval_8gram_hashes_v2.npy                        # the bounds writer's 8-gram check
  outputs/gpc4_in/vm_keys/forward_key.pub outputs/gpc4_in/vm_keys/results_key.pub
  outputs/gpc4_in/session
  outputs/gpc4_in/l6_odcv_base.jsonl  # launch 6's gate runs for the final pass (z_gpc4_l6_transcripts.py)
  outputs/gpc4_v1/g5/blackmail.jsonl  # launch 6's stage 4 transcripts, judged in launch 7 (§11.23 ruling 2d)
  outputs/gpc4_in/l6_judged_first.jsonl  # launch 7's judge on launch 6's 80: the stock base at both thresholds (§12)
  outputs/gpc4_in/a7/states.jsonl  # §18's late block: A7's 48 states (gitignored; z_gpc4_a7_states.py)
)
I="$G/outputs/gpc4_in/cst"
rm -rf "$I" && mkdir -p "$I"
SYNC=""
for f in "${CST_INPUTS[@]}"; do
  [ -f "$C/$f" ] || { echo "BAIL: $C/$f is missing"; exit 1; }
  mkdir -p "$I/$(dirname "$f")" && cp -L "$C/$f" "$I/$f"
  SYNC="${SYNC:+$SYNC,}outputs/gpc4_in/cst/$f"
done
for f in "${GPC_INPUTS[@]}"; do
  [ -f "$G/$f" ] || { echo "BAIL: $G/$f is missing (the launcher writes the VM keys and the session)"; exit 1; }
  SYNC="$SYNC,$f"
done
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
GPC_BUNDLE="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
