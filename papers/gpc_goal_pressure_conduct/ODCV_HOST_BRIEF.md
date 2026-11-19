# ODCV-Bench as a GPC-4 stage: the Docker host (`GPC_PLAN.md` §11's host page; accepted 2026-11-17)

**Where this stands (2026-11-17; accepted by the author with Amendment 3):** the one-box route (one Thunder instance as GPU and Docker host) does
not work as Thunder documents itself, so the route is a separate VM with no manual steps. **The
author's rulings:** **AWS EC2** (a new DigitalOcean account is capped at 2 vCPU), one on-demand x86
instance with 16 dedicated (non-burstable) vCPU, one provider only (the stage is latency-bound with a
30 s limit on each tool command, and shared vCPUs could turn contention into tool timeouts); ImpossibleBench runs on the same VM with upstream's Docker sandbox; the
VM is built and passes its gate tests before the pod starts; a rehearsal first. Nothing is built until
he rules on the draft. Upstream: `github.com/McGill-DMaS/ODCV-Bench`, MIT, pinned `6924daf`.

## 1. The check: can one Thunder instance be both?

- **Our cards' mode now:** the default. `tnr` 2.3.0's `create` has no mode flag (`--gpu`, `--num-gpus`,
  `--vcpus`, `--template`, `--disk`), and our provider passes none.
- **Production mode, in Thunder's current docs: not a selectable mode.** No docs or pricing page defines
  "prototyping" or "production". The create-instance API keeps a deprecated `mode` enum
  (prototyping, production): "Its value does not select execution mode; routing is derived from
  num_gpus. New integrations should omit mode." The modify-instance description says the same. No
  page states that a production instance is a standard VM. **Price:** one A100 80GB at $1.09 an hour
  with 8 vCPU (our 300 GB disk adds about $0.06, hence the $1.15 we log); no mode split, so no
  documented premium.
- **Docker on an instance** (`docs/guides/using-docker-on-thundercompute`): "Because Thunder instances
  are themselves containers, running Docker on Thunder is like running Docker inside of Docker";
  "Docker has experimental support"; "a modified version of `dockerd`"; **"Docker Compose does not
  work"**; "The container network is not isolated" (ports not published with `-p` are reachable).
  User-defined bridges, internal networks and `DOCKER-USER` rules are not addressed.
- **Upstream's harnesses need Compose:** ODCV-Bench's `run.py` runs `docker compose up --build
  --abort-on-container-exit` (an executor and a scenario container on a bridge), and its Inspect route,
  like ImpossibleBench's Docker sandbox, goes through Inspect's Docker sandbox, which shells out to
  `docker compose` (`inspect_ai/util/_sandbox/docker/compose.py`).
- **Verdict: the one-box route fails on the documented facts** (no Compose, a non-isolated container
  network, no selectable VM mode). If routing by GPU count put a 2-GPU instance on a VM (not
  documented), the second A100 adds $1.09 an hour, about $9.80 over 9 hours, over the $6 premium bar
  anyway. So: no one-box; the VM hosts both instruments (§2).

## 2. The separate VM, with no manual steps

**The instance:** **c6a.4xlarge** (16 vCPU, 32 GiB, AMD, not burstable) at **$0.612 an hour** on demand,
the best-priced 16 vCPU / 32 GiB x86 type (c7i.4xlarge is $0.714; c6i.4xlarge $0.680; c7a.4xlarge,
16 physical cores, $0.821; AWS's published price data, 2026-11-17, identical in all three regions);
a 160 GB gp3 root ($0.08 per GB-month, about $0.018 an hour) and one public IPv4 address ($0.005 an
hour): **about $0.635 an hour**, billed per second (60 s minimum). A current Ubuntu LTS image (the
launcher resolves Canonical's current AMI from AWS's public parameter at run time). No spot.
**Region: us-east-2 (Ohio).** Thunder names no datacenter ("Location: North America"); the two GPC
pods' addresses register to LightEdge Solutions (Des Moines) and Denvr Dataworks (Calgary), so an
instance's location varies, and Ohio is the most central of the three regions the author's quota
covers (32 vCPU, none in use, in each). The forward logs its measured round trip; at 11 model calls a
run, 60 ms adds about 1.3 s to a run.

One launch script (the GPC-4 card) runs everything:
- **Credentials:** read by the AWS client from the author's local profile or environment; never on a
  command line, never written to disk by us, never sent to the VM or the pod. **The account ID is read
  from the client at run time (`sts get-caller-identity`) and never written into the repo, a log or a
  commit message.** The minimal IAM policy for a dedicated identity is in §5.
- **First step: the quota.** The launcher reads the region's on-demand standard vCPU quota
  (`L-1216C47A`) and the vCPUs already running, and stops with a clear message unless 16 are free.
- **Create:** the instance tagged `gpc4-session=<stamp>` (the instance and its volume), a session key
  pair, and one security group, **inbound SSH from the launcher's own public address only** (a /32,
  looked up at launch), outbound open for setup (apt, GitHub, the image registries, PyPI) and the
  forward; **no instance profile**; **instance-initiated shutdown set to terminate**, the root volume
  **deleted on termination**. **The launcher refuses to start while any instance with a `gpc4-session`
  tag exists.**
- **The metadata endpoint:** Ubuntu's cloud-init needs the instance metadata service at first boot to
  install the session key and run the setup, so the instance launches with IMDSv2 required and a hop
  limit of 1 (no container can reach it), and **the launcher disables the endpoint as soon as setup
  finishes, before any container runs**; a gate test checks that 169.254.169.254 answers nothing, from
  the VM and from a container.
- **Setup (cloud-init):** Docker with Compose; ODCV-Bench at `6924daf` and ImpossibleBench at `d4f9a14`
  with Inspect 0.3.291 pinned; the 40 ODCV-Bench images built and ImpossibleBench's pulled; the
  firewall; `shutdown -h` at the session's hard limit, **which terminates the instance and so ends
  billing** (instance-initiated shutdown terminates). **The order:** the VM is created, built and passes
  every gate test before the pod starts; no image build or setup runs on GPU time.
- **The forward:** the pod's model server binds the pod's `127.0.0.1:8000` (the judge's `:8001`
  later). The launcher puts a second session key on the VM and its public half in the pod's
  `authorized_keys` as `restrict,port-forwarding,permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8001"`;
  the VM forwards those two ports to the Docker bridge's gateway only (`172.30.0.1:18000` and `:18001`).
  No credential beyond that restricted key reaches the VM.
- **Isolation (on the VM):** a user-defined bridge `odcv` (172.30.0.0/24); scenario containers on an
  `internal: true` network with their executor only; in `DOCKER-USER`, from 172.30.0.0/24 accept TCP
  to 172.30.0.1:18000 and :18001 and established replies, drop the rest (the internet, 169.254.169.254,
  the VM's own ports). Gate tests from a throwaway container before any run: the endpoint answers; the
  internet, the metadata address, the VM's SSH port and DNS do not.
- **Results never sit only on the VM:** the instance deletes its disk when it terminates, so each
  finished run's results go to the pod's results directory as the run completes, through a second
  restricted key in the pod's `authorized_keys` (`command="rrsync -wo <the pod's vm_results dir>",restrict`:
  write only, one directory, no shell, no forward); the pod's existing sync carries them to the
  laptop. **A rehearsal gate test:** the launcher is killed mid-run and the finished runs are found on
  the receiving side (in the rehearsal, a t3.micro stand-in for the pod, $0.0104 an hour, with the same
  restricted key; terminated with the rest). **The detached gate test (the author's ruling):** kill the
  launcher mid-run; the run in progress and the ones after it still complete, the results reach the
  stand-in, and the instance terminates itself.
- **Detached (the author's ruling):** after launch the pod and the VM run to completion with the
  laptop gone. The VM's stages run on its own driver (systemd, started by cloud-init); the pod decides
  and publishes a read-only control document on its :8001 (the next stage, each arm's served model name
  and n, or wait or stop); the VM reports by writing results and done-markers through the results key,
  and confirms delivery from the control document's received list. The judge runs on the pod. A sampling
  proxy on the VM applies the sampling of record to every request. No new credential reaches the VM.
  **The VM terminates itself** when its last stage's results are confirmed on the pod, or at the hard
  limit. **Reattach** (`launch_gpc4.sh reattach`) and **teardown** work from any address: they replace
  the security group's SSH rule with the new /32 first.
- **Egress lock (the author's ruling):** after setup, once the pod's address is known and before any
  container runs, the launcher revokes the security group's default outbound rule and allows outbound
  only to the pod's address on its SSH port. Gate tests from the VM host itself: the internet does not
  answer, the pod's port does, the forward and the results path still work. Reason: a container escape
  would otherwise land on a host with open outbound access.
- **Run, copy back, terminate:** the stage runs (§3), the final results also come back to the author's
  laptop by rsync over the session key, the instance is terminated, the key pair and security group deleted, and
  the launcher lists any `gpc4-session` instances still alive. **Teardown runs in an exit trap and as a
  standalone command** (`launch_gpc4.sh teardown`: every `gpc4-session` instance, printed, then
  terminated). The instance's own shutdown at the hard limit is the third stop.
- **ImpossibleBench on the VM (ruled):** Inspect runs on the VM as the client, through the same
  forward, with upstream's Docker sandbox (`network_mode: none`, 1 GB). The custom sandbox provider is
  dropped; rider 1's gate tests run against the container. **Capacity:** 16 vCPU covers both stages in
  sequence (ODCV-Bench at 8 concurrent runs, 16 containers; ImpossibleBench at 10 sandboxes); they
  overlap only if measured.
- **The rehearsal (zero GPU; the author runs it before the real launch):** create the VM, set it up,
  run every gate test, one ODCV-Bench scenario and one ImpossibleBench task per scaffold against a stub
  endpoint on the VM, copy the results back, tear down. Its record holds the commit, each gate test,
  the setup wall time and the cost (expected: setup 30 to 45 minutes, about 1 hour of VM time, about
  $0.65). **The real launch refuses to start without a passed rehearsal at the same commit.**
- **The failure branch:** if the VM fails its gate tests or dies mid-session, the pod runs what needs
  no VM (the bounds grid's writer, stage 3's fallback gate on the bounds pilot, stage 4) and stops.

## 3. The stage

The base on all 80 runs (40 scenarios, mandated and incentivized) is the gate, judged by gpt-oss-120b
through our runner with upstream's rubric and preamble unchanged, misaligned at upstream's threshold of
3; it counts at a base rate of 0.15 or more. Then `C_conduct` and `C_bounds` in context and C1, 80 runs
each, paired by scenario, mandated and incentivized apart, the acting rate beside every rate, and a
Codex agreement read of 30 transcripts (GPC-A11). **Tokens a run** (the repo's 80 saved Qwen3.6-27B
runs, Qwen3.8's tokenizer): generated median 3.4k, mean 6.2k, max 49k; cumulative input median 34k,
mean 70k. **About 0.7 A100-h an arm** (latency-bound at 8 to 10 concurrent runs), 0.25 h of judging.

**Every arm reports its share of runs with a tool timeout or a container error**, beside its violation
and acting rates.

**Price (both instruments on the VM; the VM at about $0.635 an hour):**

| case | A100-h on Thunder | the VM | total |
|---|---|---|---|
| floor (every instrument banks) | 4.4 ($5.20) | about 3.25 h, $2.05 | **about $7.25** |
| ceiling (the 9 h cap) | 9.0 ($10.50) | up to about 8.75 h, $5.55 | **about $16.05** |
| the failure branch | about 2.4 ($2.90) | whatever ran before the loss | about $3 to $5 |
| the rehearsal | none | about 1 h | **about $0.65** |

## 4. Rootless Podman on a default Thunder instance (a fallback only; not built on)

- **What the instance must allow:** unprivileged user namespaces inside its container
  (`/proc/sys/user/max_user_namespaces` above 0, and `unshare` not blocked by its seccomp profile);
  sub-UID and sub-GID ranges of at least 65,536 for the user (`/etc/subuid`, `/etc/subgid`, with
  `newuidmap` and `newgidmap` setuid); a storage driver (native overlay in a user namespace on a
  recent kernel, else `fuse-overlayfs`, which needs `/dev/fuse`); and `pasta` or `slirp4netns` for
  networking.
- **Against Podman's Docker-compatible socket** (`podman system service`, `DOCKER_HOST` at the user's
  socket): `docker compose` v2 and Inspect's Docker sandbox speak the Docker API and generally run;
  user-defined and `internal` networks exist (netavark). There is no `DOCKER-USER` chain: a rootless
  network lives in the rootless network namespace, so the isolation rule becomes "internal networks
  only, plus the one forwarded endpoint", and upstream's host-gateway address needs checking.
- **The ten-minute probe** (on a default instance, after `apt-get install -y podman uidmap
  fuse-overlayfs slirp4netns`): `cat /proc/sys/user/max_user_namespaces`; `unshare -Ur true`;
  `ls -l /dev/fuse`; `grep "$USER" /etc/subuid /etc/subgid`; `podman info` (storage and network
  backends); `podman run --rm alpine true`; `podman network create --internal probe`; start
  `podman system service`, point `DOCKER_HOST` at it, and run a two-service Compose file on an internal
  network with one container reaching a host port forwarded in; then Inspect's Docker sandbox on one
  hello-world sample. Any step failing settles it as no.
- Sysbox is out (it installs on the host).

## 5. The minimal IAM policy (for a dedicated identity; us-east-2 only)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ReadOnly", "Effect": "Allow",
     "Action": ["ec2:DescribeInstances", "ec2:DescribeInstanceStatus", "ec2:DescribeImages",
                "ec2:DescribeKeyPairs", "ec2:DescribeSecurityGroups", "ec2:DescribeVpcs",
                "ec2:DescribeSubnets", "ec2:DescribeVolumes", "servicequotas:GetServiceQuota"],
     "Resource": "*", "Condition": {"StringEquals": {"aws:RequestedRegion": "us-east-2"}}},
    {"Sid": "DecodeDenials", "Effect": "Allow", "Action": "sts:DecodeAuthorizationMessage",
     "Resource": "*"},
    {"Sid": "UbuntuImage", "Effect": "Allow", "Action": "ssm:GetParameter",
     "Resource": "arn:aws:ssm:us-east-2::parameter/aws/service/canonical/ubuntu/*"},
    {"Sid": "RunTaggedOnly", "Effect": "Allow", "Action": "ec2:RunInstances",
     "Resource": ["arn:aws:ec2:us-east-2:*:instance/*", "arn:aws:ec2:us-east-2:*:volume/*"],
     "Condition": {"StringLike": {"aws:RequestTag/gpc4-session": "*"},
                   "StringEquals": {"ec2:InstanceType": ["c6a.4xlarge", "t3.micro"]}}},
    {"Sid": "RunSupporting", "Effect": "Allow", "Action": "ec2:RunInstances",
     "Resource": ["arn:aws:ec2:us-east-2::image/*", "arn:aws:ec2:us-east-2::snapshot/*",
                  "arn:aws:ec2:us-east-2:*:subnet/*", "arn:aws:ec2:us-east-2:*:network-interface/*"]},
    {"Sid": "RunInSessionGroupWithSessionKey", "Effect": "Allow", "Action": "ec2:RunInstances",
     "Resource": ["arn:aws:ec2:us-east-2:*:security-group/*", "arn:aws:ec2:us-east-2:*:key-pair/*"],
     "Condition": {"StringLike": {"aws:ResourceTag/gpc4-session": "*"}}},
    {"Sid": "TagOnCreate", "Effect": "Allow", "Action": "ec2:CreateTags",
     "Resource": "arn:aws:ec2:us-east-2:*:*/*",
     "Condition": {"StringEquals": {"ec2:CreateAction": ["RunInstances", "CreateSecurityGroup",
                                                         "CreateKeyPair"]}}},
    {"Sid": "ActOnSessionResources", "Effect": "Allow",
     "Action": ["ec2:TerminateInstances", "ec2:ModifyInstanceMetadataOptions",
                "ec2:DeleteKeyPair", "ec2:DeleteSecurityGroup",
                "ec2:AuthorizeSecurityGroupIngress", "ec2:RevokeSecurityGroupIngress",
                "ec2:AuthorizeSecurityGroupEgress", "ec2:RevokeSecurityGroupEgress"],
     "Resource": "*",
     "Condition": {"StringLike": {"aws:ResourceTag/gpc4-session": "*"},
                   "StringEquals": {"aws:RequestedRegion": "us-east-2"}}},
    {"Sid": "CreateSessionKeyAndGroup", "Effect": "Allow",
     "Action": ["ec2:CreateKeyPair", "ec2:CreateSecurityGroup"], "Resource": "*",
     "Condition": {"StringLike": {"aws:RequestTag/gpc4-session": "*"},
                   "StringEquals": {"aws:RequestedRegion": "us-east-2"}}},
    {"Sid": "CreateGroupInAVpc", "Effect": "Allow", "Action": "ec2:CreateSecurityGroup",
     "Resource": "arn:aws:ec2:us-east-2:*:vpc/*"}
  ]
}
```

No IAM, no instance profile (`iam:PassRole` is absent, so a profile cannot be attached), no other
service; `sts:GetCallerIdentity` needs no grant. Every write is limited to resources carrying the
`gpc4-session` tag, two instance types (the host and the rehearsal's t3.micro stand-in) and one region,
and **an instance can start only in a `gpc4-session` security group with a `gpc4-session` key pair**
(the lead's tightening: the launcher cannot start one in another group or with another key). The
snapshot ARN is in RunSupporting because AWS's Service Authorization Reference lists `snapshot` among
RunInstances' resource types (unstarred: evaluated when it applies, as when launching from an
EBS-backed image), with the format `arn:${Partition}:ec2:${Region}::snapshot/${SnapshotId}` (no account,
as for images). **The rehearsal is the policy's real test:** on an access denial, the launcher decodes
the message (`sts decode-authorization-message`, which needs one more grant, `sts:DecodeAuthorizationMessage`,
added below) and reports the missing resource; the policy is never widened by guess. **The first rehearsal's denial
(2026-11-18):** `ec2:CreateSecurityGroup` on `arn:aws:ec2:us-east-2:<account>:vpc/vpc-...`: creating a group
is authorized against the VPC as well as the new group, and the request-tag condition cannot hold on the
VPC. Added: `CreateGroupInAVpc`, that one action on a us-east-2 VPC; the group itself still needs the
`gpc4-session` tag.
