---
type: Investigation
title: Generation slowness campaign (July 2026)
description: A suspected engine regression (3-9x slowdowns) was GPU contention from the host app's own UI render loop; the engine was innocent.
tags: [performance, contention, fluxforge-studio, root-cause]
timestamp: 2026-07-15T00:00:00Z
---

# Symptom

Host app (FluxForge Studio) reported generations 3-9x slower than usual on
identical configs (klein-9b bf16, 2048x1536, 4 steps: 85 s on July 9 vs
364 s on July 13), with the engine commit identical between fast and slow
runs. Initial hypothesis: cold-start / kernel-compilation cost.

# Method

1. Exact reproduction: the slow run's full config (prompt, seed, source
   image, chain parameters) extracted from the app's SwiftData store and
   replayed through `flux2 inpaint` with per-phase profiling (`--profile`,
   `--repeat-count`, added in PR #112).
2. Cross-suspension attribution: `kill -STOP <pid>` on suspects while
   sampling GPU utilization (`ioreg -r -c IOAccelerator`).
3. Live stack capture (`sample`) during in-app generations.

# Findings

- Healthy baseline: ~60-66 s wall for the incriminated config — see
  [baselines](/docs/knowledge/benchmarks/klein9b-baselines.md). Loading was
  exonerated immediately (17 GB transformer loads in 2.5 s warm).
- The slowdowns reproduced ONLY under a competing GPU consumer: suspending
  the host app mid-series turned a 577 s run into 57 s within the same
  process. The app, idle at 0.3% CPU, saturated the GPU at 96-99%
  (renderer/tiler = UI rendering, not compute).
- Mechanism in the app: a SwiftUI animation clock that never stopped
  (`repeatForever` indicators; layout storm re-rendering whatever view was
  frontmost, confirmed by sampled stacks). Fixed app-side.
- The cold-start hypothesis was **disproved**: cold first runs were the
  FAST ones (see [Metal full-JIT](/docs/knowledge/pitfalls/metal-full-jit.md)).
- A methodological trap encountered twice: benchmarks run while the machine
  had ambient GPU/CPU load (Xcode builds, active UI) produced 2-4x inflated
  and erratic step times. Hence the GPU-quiet gate in the
  [contention playbook](/docs/knowledge/playbooks/gpu-contention-diagnosis.md).

# Outcome

Framework PRs #112 (per-phase profiling + reproduction tooling), #113
(compile decision), #114 (loading hygiene), #115 (training TE off-main),
#116 (pre-quantized checkpoints) all grew out of this campaign.
