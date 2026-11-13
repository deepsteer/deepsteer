# What a second GPU provider would take (CST Amendment 48 item 7)

2026-11-13. Zero GPU, nothing built. The author decides on a port before the partner's document
arrives.

**RunPod-specific code is confined to `session_lib.sh` and `run_session.sh`.** Every remote runner
(`remote_*.sh`), every pod script and the sync emulation are provider-neutral: they need a Linux
host with one A100 80GB, CUDA 12.9 drivers, `/workspace`, root over SSH and Python 3.11.

| piece | RunPod-specific? | what a port replaces |
|---|---|---|
| `gql`, `API`, `RUNPOD_API_KEY` | yes | the provider's API client and key |
| `rp_provision_pod` (`podFindAndDeployOnDemand`: cloud, GPU type, disk, image, `PUBLIC_KEY`) | yes | create an instance with one A100 80GB, a 300 GB disk and our SSH key |
| `rp_datacenter_field` | yes | the provider's region field |
| `rp_wait_for_ssh` (the pod query for its public port 22) | yes | poll the instance until SSH answers |
| `rp_log_rate` (`costPerHr`) | yes | the provider's price for the instance |
| `cleanup` (terminate on exit) | yes | delete the instance |
| `rp_sync_up`, `rp_output_filters`, `rp_check_sync_*`, `rp_download`, `rp_push_token`, `rp_on_demand_sync` | no (rsync and ssh) | nothing |
| the launch command and the `.session_done` loop in `run_session.sh` | no | nothing |

**Image and bootstrap.** RunPod starts `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with
sshd and our key injected through `PUBLIC_KEY`, and a `/workspace` volume. Every runner then builds
its own venv (torch 2.13.0+cu129, vLLM 0.31.0's pinned wheel, transformers 5.17.0, peft 0.21.2) and
checks the stack, so the image only has to supply Ubuntu 22.04, Python 3.11, `rsync`, bash
(`jq` runs on this machine, not the pod), a CUDA 12.9-capable driver (vLLM's cu129 wheel), and sshd with our key. A provider with a
plain CUDA base image needs an install step for sshd and the key, and a symlink or mount for
`/workspace`.

**The validation run before any registered read on the new provider** (about 3-4 A100-h):
1. The runner's stack check and its gate tests (every pod already runs these).
2. **The fixed smoke check** (Amendment 31): the saved D-doc 1M adapter served against its saved HF
   letters, within 3 floors, with the served-vs-base distance at least half the saved shift.
3. **A P-ctx reproduction against a same-session base:** the operator cell's 458 clear items at k 2,
   P-ctx's ordered share against the base in the same session, read against P-ctx's three draws on
   RunPod (−0.275, −0.286 and config 1's −0.272, each against its own session base). It passes when
   the contrast's CI contains the RunPod draws' range.

**Cost of the port:** about half a day of code (the six RunPod rows above behind a provider switch),
one validation pod, and the sync emulation unchanged.
