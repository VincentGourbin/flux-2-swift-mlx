---
type: Playbook
title: Diagnosing GPU contention
description: The 5-minute protocol to attribute a slow generation to its real cause instead of suspecting the engine.
tags: [diagnostics, gpu, contention, protocol]
timestamp: 2026-07-15T00:00:00Z
---

Use when a generation runs far above the
[baselines](/docs/knowledge/benchmarks/klein9b-baselines.md), or a machine
feels GPU-busy with nothing obviously running.

# Protocol

```bash
# 1. Is the GPU saturated, and by rendering or compute?
ioreg -r -c IOAccelerator -w0 | grep -E 'Utilization'
#    Renderer/Tiler high => UI rendering; Device high alone => compute.

# 2. Attribute by suspension (reversible, safe): suspend the suspect,
#    watch utilization. Repeat per suspect. SUSPEND THE BINARY, not its
#    zsh wrapper (pgrep -f matches the wrapper's command line too).
PID=$(pgrep -x "Fluxforge Studio"); kill -STOP $PID; sleep 6
ioreg -r -c IOAccelerator -w0 | grep 'Device Utilization'; kill -CONT $PID

# 3. The decisive artifact: sample the culprit DURING the episode —
#    hot stacks name the exact code (SwiftUI AnimationDriver, ViewGraph,
#    MLX eval, etc.).
sample "<process name>" 5 -file /tmp/culprit_sample.txt
```

# Benchmarking rule derived from this

Never trust step-time benchmarks from a machine with ambient GPU load
(Xcode builds, an animating UI, another generation). Gate every measured
run on a quiet window (e.g. device utilization < 15-25% sustained) and
record utilization alongside the run to tag contaminated measurements.
Correctness tests are fine under contention; timing tests are not.

# Gotchas learned the hard way

- `Renderer Utilization` does NOT discriminate UI vs compute while your own
  MLX process runs — only cross-suspension does.
- A per-step spread of 10x within one run (e.g. 16 s and 134 s steps) is
  the signature of contention, not thermal throttling (which is smooth).
- Per-process GPU attribution without sudo is limited; suspension +
  `sample` beats trying to read per-process GPU counters.
