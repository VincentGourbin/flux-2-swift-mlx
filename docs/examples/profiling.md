# Profiling Toolkit

Profile, benchmark, and compare Flux.2 inference performance on Apple Silicon with Chrome Trace visualization and MLX memory tracking.

Inspired by [HuggingFace diffusers profiling patterns](https://huggingface.co/docs/diffusers) but built natively for MLX and Apple Silicon unified memory.

## Overview

The profiling toolkit provides three levels of analysis:

| Command | Purpose | Output |
|---------|---------|--------|
| `flux2 profile run` | Single profiled generation | Chrome Trace JSON + console report |
| `flux2 profile benchmark` | Statistical benchmarking | Aggregate stats (mean, std, min, max) |
| `flux2 profile compare` | Cross-config comparison | Side-by-side timings + combined trace |

All commands produce Chrome Trace JSON files viewable in [Perfetto UI](https://ui.perfetto.dev/) with visual swim lanes for each pipeline stage, memory counters, and per-step denoising timelines.

---

## Quick Start

```bash
# Single profiled generation (default subcommand)
flux2 profile run "a beaver building a dam" --model klein-4b

# Benchmark with warm-up
flux2 profile benchmark "a beaver building a dam" --model klein-4b --warmup 1 --runs 3

# Compare quantization configs
flux2 profile compare "a beaver building a dam" \
  --configs "klein-4b:qint8,klein-4b:bf16"
```

After any run, open the exported `.json` trace file in [Perfetto UI](https://ui.perfetto.dev/) to get an interactive timeline view.

---

## CLI Commands

### `flux2 profile run`

Single profiled generation with Chrome Trace export. This is the default subcommand (`flux2 profile` is equivalent to `flux2 profile run`).

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `<prompt>` | (required) | Text prompt for image generation |
| `--model` | `klein-4b` | Model variant: `dev`, `klein-4b`, `klein-9b`, `klein-9b-kv` |
| `--steps` / `-s` | model default | Number of inference steps |
| `--guidance` / `-g` | model default | Guidance scale |
| `--width` / `-w` | `512` | Image width |
| `--height` / `-h` | `512` | Image height |
| `--seed` | random | Random seed for reproducibility |
| `--text-quant` | `8bit` | Text encoder quantization: `bf16`, `8bit`, `6bit`, `4bit` |
| `--transformer-quant` | `qint8` | Transformer quantization: `bf16`, `qint8`, `int4` |
| `--output-dir` | `./profile_results` | Output directory for trace files |
| `--per-step-memory` | false | Record memory at each denoising step |
| `--no-chrome-trace` | false | Disable Chrome Trace JSON export |
| `--no-image` | false | Skip saving the generated image |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for gated models |
| `--models-dir` | system default | Custom models directory |

#### Example

```bash
flux2 profile run "A red rose on a wooden table, soft morning light" \
  --model klein-4b \
  --width 512 --height 512 \
  --per-step-memory \
  --output-dir ./profile_results
```

Output files:

```
profile_results/
  klein-4b_qint8_trace.json    # Chrome Trace (open in Perfetto)
  profiled_output.png           # Generated image
```

A ready-to-use example trace and image are included in [`docs/examples/profiling_output/`](profiling_output/):
- [`klein-4b_qint8_trace.json`](profiling_output/klein-4b_qint8_trace.json) -- open in [Perfetto UI](https://ui.perfetto.dev/)
- [`profiled_output.png`](profiling_output/profiled_output.png) -- the generated image

Console output (real run on M2 Ultra 96GB, Klein 4B qint8, 512x512, 4 steps):

```
Profiling: Flux.2 Klein 4B 8bit/qint8
Image: 512x512, 4 steps, guidance 1.0

Step 1/4... Step 2/4... Step 3/4... Step 4/4

  FLUX.2 PROFILING REPORT

  Model: klein-4b  Quant: 8bit/qint8
  Image: 512x512  Steps: 4
  Device: applegpu_g15s  RAM: 96GB

  PHASE TIMINGS
  ──────────────────────────────────────────────────────
  1. Load Text Encoder            1.83s   18.1% ███
  2. Text Encoding               631.0ms   6.2% █
  3. Unload Text Encoder          31.0ms   0.3%
  4. Load Transformer             1.14s   11.3% ██
  5. Load VAE                     43.2ms   0.4%
  6. Denoising Loop               5.96s   58.9% ███████████
  7. VAE Decode                  470.9ms   4.7%
  8. Post-processing               6.9ms   0.1%
  ──────────────────────────────────────────────────────
  TOTAL                          10.12s  100.0%

  DENOISING STEP STATISTICS
  ──────────────────────────────────────────────────────
  Steps: 4
  Average:    1.49s   Std:     3.5ms
  Min:    1.49s   Max:    1.49s

  Estimated for different step counts:
    10 steps:   14.89s
    20 steps:   29.77s
    28 steps:   41.68s
    50 steps: 1m 14.4s

  MEMORY
  ──────────────────────────────────────────────────────
  Peak MLX Active: 4510.1 MB
  Peak Process: 9025.8 MB

  Memory Timeline:
    begin:1. Load Text Encoder          MLX:     0.0 MB
    end:1. Load Text Encoder            MLX:  4495.1 MB
    begin:2. Text Encoding              MLX:  4495.1 MB
    end:2. Text Encoding                MLX:  4510.1 MB
    begin:3. Unload Text Encoder        MLX:  4510.1 MB
    end:3. Unload Text Encoder          MLX:    15.0 MB
    begin:4. Load Transformer           MLX:    15.0 MB
    end:4. Load Transformer             MLX:  3942.0 MB
    begin:5. Load VAE                   MLX:  3942.0 MB
    end:5. Load VAE                     MLX:  4102.3 MB
    begin:6. Denoising Loop             MLX:  4102.8 MB
    end:6. Denoising Loop               MLX:  4104.9 MB
    begin:7. VAE Decode                 MLX:  4105.9 MB
    end:7. VAE Decode                   MLX:  4236.9 MB
    begin:8. Post-processing            MLX:  4236.9 MB
    end:8. Post-processing              MLX:  4108.9 MB
```

---

### `flux2 profile benchmark`

Statistical benchmarking with warm-up runs excluded from measurements. Uses a fixed seed (default 42) across all runs for reproducibility.

#### Options

All options from `profile run` are available, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--warmup` | `1` | Number of warm-up runs (excluded from stats) |
| `--runs` | `3` | Number of measured runs |
| `--output-dir` | `./benchmark_results` | Output directory |

#### Example

```bash
flux2 profile benchmark "a fox in a forest" \
  --model klein-4b \
  --warmup 2 --runs 5 \
  --width 512 --height 512
```

Console output:

```
Benchmarking: Flux.2 Klein 4B 8bit/qint8
Image: 512x512, 4 steps
Warm-up: 2, Measured runs: 5

Warm-up 1/2... 13.21s
Warm-up 2/2... 12.84s
Run 1/5... 12.72s
Run 2/5... 12.68s
Run 3/5... 12.71s
Run 4/5... 12.65s
Run 5/5... 12.69s

  FLUX.2 BENCHMARK REPORT

  Model: klein-4b  Quant: 8bit/qint8
  Image: 512x512  Steps: 4
  Warm-up: 2  Measured runs: 5

  PHASE TIMINGS (mean +/- std)
  ──────────────────────────────────────────────────────
  1. Load Text Encoder    2.31s +/- 0.03s  [2.28s - 2.35s]
  2. Text Encoding        0.85s +/- 0.01s  [0.84s - 0.87s]
  6. Denoising (4 steps)  6.82s +/- 0.05s  [6.75s - 6.89s]
  7. VAE Decode           0.18s +/- 0.00s  [0.18s - 0.19s]
  ──────────────────────────────────────────────────────
  TOTAL                  12.69s +/- 0.03s  [12.65s - 12.72s]

  DENOISING STEP
  ──────────────────────────────────────────────────────
  Average per step: 1705.3ms +/- 42.1ms
  Range: [1652.8ms - 1771.2ms]

  MEMORY
  ──────────────────────────────────────────────────────
  Peak MLX Active: 8234.2 MB
  Peak Process: 9812.7 MB
```

The benchmark also exports a combined Chrome Trace of all measured runs:

```
benchmark_results/
  benchmark_klein-4b_5runs.json   # All runs as separate processes in one trace
```

---

### `flux2 profile compare`

Compare performance across multiple model/quantization configurations in a single command.

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `<prompt>` | (required) | Text prompt for image generation |
| `--configs` | (required) | Comma-separated `model:quant` pairs |
| `--steps` / `-s` | model default | Number of inference steps |
| `--width` / `-w` | `512` | Image width |
| `--height` / `-h` | `512` | Image height |
| `--seed` | `42` | Random seed |
| `--runs` | `1` | Runs per configuration |
| `--text-quant` | `8bit` | Text encoder quantization (shared across configs) |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token |
| `--models-dir` | system default | Custom models directory |
| `--output-dir` | `./comparison_results` | Output directory |

The `--configs` value is a comma-separated list of `model:transformer_quant` pairs. Supported models: `dev`, `klein-4b`, `klein-9b`, `klein-9b-kv`. Supported transformer quantizations: `bf16`, `qint8`, `int4`.

#### Example

```bash
# Compare Klein 4B quantizations
flux2 profile compare "a beaver building a dam" \
  --configs "klein-4b:qint8,klein-4b:bf16,klein-4b:int4"

# Compare models at same quantization
flux2 profile compare "a detailed landscape" \
  --configs "klein-4b:qint8,klein-9b:qint8" \
  --width 1024 --height 1024

# Multiple runs per config for more stable results
flux2 profile compare "a portrait photograph" \
  --configs "klein-4b:qint8,klein-4b:bf16" \
  --runs 3
```

Console output:

```
Comparing 3 configurations
Image: 512x512, seed 42

Running: klein-4b qint8...
  klein-4b qint8: 12.72s
Running: klein-4b bf16...
  klein-4b bf16: 14.31s
Running: klein-4b int4...
  klein-4b int4: 11.98s

COMPARISON SUMMARY
------------------------------------------------------------
  klein-4b qint8                12.72s  Peak: 8234MB
  klein-4b bf16                 14.31s  Peak: 14102MB
  klein-4b int4                 11.98s  Peak: 6128MB

Comparison trace exported to ./comparison_results/comparison_trace.json
View in Perfetto: https://ui.perfetto.dev/
```

---

## Chrome Trace Visualization

All profiling commands export Chrome Trace JSON files compatible with [Perfetto UI](https://ui.perfetto.dev/) and `chrome://tracing`.

### Viewing a Trace

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. Click "Open trace file" (or drag and drop)
3. Select the `.json` file from your output directory

### Visual Lanes (Threads)

Events are organized into visual lanes by category:

| Thread ID | Lane | Events |
|-----------|------|--------|
| 1 | **Text Encoding** | Text encoder load, text encoding, text encoder unload, VLM interpretation |
| 2 | **Transformer** | Transformer load, denoising loop, individual denoising steps |
| 3 | **VAE** | VAE load, VAE decode |
| 4 | **Post-processing** | Image post-processing |
| 5 | **Memory** | Memory counter events (MLX Active, MLX Cache, MLX Peak, Process) |
| 6 | **eval() Syncs** | MLX evaluation synchronization points |

### Event Types

The trace uses standard Chrome Trace Event Format phases:

- **`B` / `E`** (Begin/End) -- Pipeline phases (text encoding, denoising loop, VAE decode)
- **`X`** (Complete) -- Individual denoising steps with known duration
- **`C`** (Counter) -- Memory timeline (plotted as a line chart in Perfetto)
- **`i`** (Instant) -- Session metadata (device, model, quantization)
- **`M`** (Metadata) -- Process and thread names

### Comparison Traces

When using `profile benchmark` or `profile compare`, multiple sessions are exported as separate **processes** within a single trace file. In Perfetto, each process appears as a separate collapsible row, making it easy to visually compare timelines side by side.

### Reading Memory Counters

In Perfetto, the Memory lane (thread 5) shows stacked counter tracks:

- **MLX Active (MB)** -- Currently allocated GPU memory managed by MLX
- **MLX Cache (MB)** -- MLX cached (reusable) memory
- **MLX Peak (MB)** -- High-water mark of MLX allocations
- **Process (MB)** -- Total process physical footprint (`phys_footprint` from `task_vm_info`)

Click on any event to see its args panel with exact memory values.

---

## Programmatic API

The profiling toolkit can be used directly from Swift code, independent of the CLI.

### ProfilingConfig

Configure what data to collect:

```swift
import Flux2Core

// Default single-run config (memory tracking on, Chrome Trace on)
let config = ProfilingConfig.singleRun

// Detailed profiling with per-step memory snapshots
let config = ProfilingConfig.detailed

// Benchmark config (3 runs, 1 warm-up, no Chrome Trace)
let config = ProfilingConfig.benchmark(runs: 3, warmup: 1)

// Fully custom
let config = ProfilingConfig(
    trackMemory: true,            // Record memory at phase boundaries
    trackPerStepMemory: true,     // Record memory at each denoising step
    benchmarkRuns: nil,           // nil = single run, N = benchmark
    warmupRuns: 1,                // Warm-up runs (benchmark only)
    outputDirectory: URL(fileURLWithPath: "./output"),
    exportChromeTrace: true,      // Export Chrome Trace JSON
    printSummary: true            // Print console report
)
```

### ProfilingSession

The central coordinator that collects events and memory snapshots:

```swift
let session = ProfilingSession(config: .singleRun)

// Set metadata (shown in reports and trace files)
session.modelVariant = "klein-4b"
session.quantization = "8bit/qint8"
session.imageSize = "512x512"
session.steps = 4

// Record phase boundaries
session.beginPhase("Text Encoding", category: .textEncoding)
// ... do work ...
session.endPhase("Text Encoding", category: .textEncoding)

// Record a complete event with known duration
session.recordComplete("Custom Op", category: .custom, durationUs: 1500)

// Record individual denoising steps
session.recordDenoisingStep(index: 1, total: 4, durationUs: 1_705_000)

// Record a memory snapshot at an arbitrary point
session.recordMemorySnapshot(context: "after_model_load")

// Access collected data
let events = session.getEvents()           // [ProfilingEvent]
let timeline = session.getMemoryTimeline() // [MemoryTimelineEntry]
let elapsed = session.elapsedSeconds       // TimeInterval

// Generate human-readable report
print(session.generateReport())
```

Session metadata is auto-populated from the system:

- `session.deviceArchitecture` -- GPU architecture string from `GPU.deviceInfo()`
- `session.systemRAMGB` -- Total system RAM in GB

### ProfilingCategory

Events are categorized for lane assignment in Chrome Trace:

```swift
public enum ProfilingCategory: String {
    case textEncoderLoad       // Thread 1 - Text Encoding lane
    case textEncoding          // Thread 1
    case textEncoderUnload     // Thread 1
    case vlmInterpretation     // Thread 1
    case transformerLoad       // Thread 2 - Transformer lane
    case denoisingLoop         // Thread 2
    case denoisingStep         // Thread 2
    case vaeLoad               // Thread 3 - VAE lane
    case vaeDecode             // Thread 3
    case postProcess           // Thread 4 - Post-processing lane
    case evalSync              // Thread 6 - eval() Syncs lane
    case memoryOp              // Thread 5 - Memory lane
    case custom                // Thread 7 - Other
}
```

### ChromeTraceExporter

Export sessions to Chrome Trace JSON:

```swift
// Single session
let traceData: Data = ChromeTraceExporter.export(session: session)
try traceData.write(to: URL(fileURLWithPath: "trace.json"))

// Multiple sessions for comparison (each becomes a separate process in the trace)
let labeledSessions: [(label: String, session: ProfilingSession)] = [
    ("Klein 4B qint8", session1),
    ("Klein 4B bf16", session2),
]
let comparisonData = ChromeTraceExporter.exportComparison(sessions: labeledSessions)
try comparisonData.write(to: URL(fileURLWithPath: "comparison.json"))
```

### BenchmarkAggregator

Aggregate multiple sessions into statistical results:

```swift
let result: BenchmarkResult = BenchmarkAggregator.aggregate(
    sessions: measuredSessions,  // [ProfilingSession] (warm-up already excluded)
    warmupCount: 1               // For display in report only
)

// Access phase-level statistics
for phase in result.phaseStats {
    print("\(phase.name): \(phase.meanMs)ms +/- \(phase.stdMs)ms")
    print("  Range: [\(phase.minMs)ms - \(phase.maxMs)ms]")
}

// Step-level statistics
if let step = result.stepStats {
    print("Per-step: \(step.meanMs)ms +/- \(step.stdMs)ms")
}

// Total statistics
print("Total: \(result.totalStats.meanMs)ms")

// Peak memory across all runs
print("Peak MLX: \(result.peakMLXActiveMB) MB")
print("Peak Process: \(result.peakProcessMB) MB")

// Formatted report
print(result.generateReport())
```

---

## Architecture

### Active Session Bridge Pattern

The profiling toolkit integrates with the existing `Flux2Profiler` singleton without modifying the pipeline code. The pipeline already calls `Flux2Profiler.shared.start()` / `.end()` / `.recordStep()` at key points. The profiling commands attach a `ProfilingSession` to the profiler, which receives richer data (memory snapshots, microsecond timestamps, categorized events):

```
Flux2Pipeline                     Flux2Profiler.shared
    |                                     |
    +-- start("1. Load Text Encoder")  -->+-- isEnabled? --> activeSession?.beginPhase(...)
    |                                     |                   (records timestamp + memory snapshot)
    +-- end("1. Load Text Encoder")    -->+-- isEnabled? --> activeSession?.endPhase(...)
    |                                     |
    +-- recordStep(duration:)          -->+-- isEnabled? --> activeSession?.recordDenoisingStep(...)
    |                                     |                   (records step + optional per-step memory)
    ...                                   ...
```

This means:

1. **No pipeline changes needed** -- `Flux2Profiler.start()` / `.end()` calls are already in place
2. **Backward compatible** -- Without an `activeSession`, the profiler behaves as before (simple timing collection)
3. **Rich when attached** -- With a session, each phase boundary captures MLX memory, process footprint, and microsecond-precision timestamps

The category of each event is inferred from the phase name via `ProfilingSession.inferCategory(_:)`, which maps strings like `"1. Load Text Encoder"` to `ProfilingCategory.textEncoderLoad` for correct lane assignment.

### Data Flow

```
ProfilingSession
    |
    +-- events: [ProfilingEvent]         --> ChromeTraceExporter.export()
    |     (name, category, phase,             --> trace.json
    |      timestampUs, durationUs,
    |      memory snapshots)
    |
    +-- memoryTimeline: [MemoryTimelineEntry] --> Counter events in trace
    |     (timestampUs, context,
    |      mlxActiveMB, mlxCacheMB,
    |      mlxPeakMB, processFootprintMB)
    |
    +-- generateReport()                 --> Console output
    |
    +-- BenchmarkAggregator.aggregate()  --> BenchmarkResult
          (multi-session statistics)          --> generateReport()
```

---

## Memory Timeline

### MLX Memory Metrics on Apple Silicon

Apple Silicon uses unified memory shared between CPU and GPU. MLX manages GPU allocations through its own allocator. The profiling toolkit tracks four memory metrics:

| Metric | Source | What It Measures |
|--------|--------|------------------|
| **MLX Active** | `Memory.activeMemory` | Currently allocated GPU tensors |
| **MLX Cache** | `Memory.cacheMemory` | Freed but not returned to OS (reusable pool) |
| **MLX Peak** | `Memory.peakMemory` | High-water mark since process start |
| **Process Footprint** | `task_vm_info.phys_footprint` | Total physical memory used by the process |

**Key observations:**

- **MLX Active** rises during model loading and drops when models are unloaded. This is the most useful metric for understanding GPU memory pressure.
- **MLX Cache** represents memory that MLX has freed but keeps in a pool for fast reallocation. It does not count toward memory pressure but can be reclaimed under memory pressure.
- **Process Footprint** includes MLX allocations plus CPU-side allocations (Swift heap, mapped files, etc.). It is always larger than MLX Active.
- On Apple Silicon, `vm_stat` uses **16K pages** (not 4K). Multiply page counts by 16384 for correct byte values.

### Memory Tracking Modes

| Config Flag | When Sampled | Use Case |
|-------------|-------------|----------|
| `trackMemory: true` | At each phase begin/end | See memory per pipeline stage |
| `trackPerStepMemory: true` | At each denoising step | Fine-grained step-level memory |

The `--per-step-memory` CLI flag enables `trackPerStepMemory`. It adds slight overhead since each denoising step queries the Metal allocator, but provides a detailed view of memory behavior during the denoising loop.

---

## Examples

### Profile Different Models

```bash
# Klein 4B (fastest, commercial-friendly)
flux2 profile run "a beaver building a dam" --model klein-4b --seed 42

# Klein 9B (better quality)
flux2 profile run "a beaver building a dam" --model klein-9b --seed 42

# Dev (maximum quality, much slower)
flux2 profile run "a beaver building a dam" --model dev --steps 28 --seed 42
```

### Profile Different Resolutions

```bash
flux2 profile run "a city skyline" --model klein-4b --width 512 --height 512
flux2 profile run "a city skyline" --model klein-4b --width 1024 --height 1024
flux2 profile run "a city skyline" --model klein-4b --width 2048 --height 2048
```

### Quantization Comparison

```bash
flux2 profile compare "a detailed portrait" \
  --configs "klein-4b:bf16,klein-4b:qint8,klein-4b:int4" \
  --width 1024 --height 1024 \
  --output-dir ./quant_comparison
```

### Stable Benchmarks for Regression Testing

```bash
flux2 profile benchmark "a red fox in the snow" \
  --model klein-4b \
  --warmup 3 --runs 10 \
  --width 512 --height 512 \
  --seed 42 \
  --output-dir ./regression_baseline
```

### Detailed Step-Level Memory Profiling

```bash
flux2 profile run "a complex scene with many elements" \
  --model klein-9b \
  --per-step-memory \
  --output-dir ./memory_analysis
```

Then open `memory_analysis/klein-9b_qint8_trace.json` in Perfetto to see memory counters plotted at every denoising step.

### Skip Image Output for Pure Timing

```bash
flux2 profile run "benchmark prompt" \
  --model klein-4b \
  --no-image \
  --output-dir ./timing_only
```

---

## Source Files

| File | Description |
|------|-------------|
| `Sources/Flux2CLI/ProfileCommand.swift` | CLI command definitions (`profile run`, `benchmark`, `compare`) |
| `Sources/Flux2Core/Utils/Profiling/ProfilingSession.swift` | Central session coordinator |
| `Sources/Flux2Core/Utils/Profiling/ProfilingConfig.swift` | Configuration presets and options |
| `Sources/Flux2Core/Utils/Profiling/ProfilingEvent.swift` | Event, category, and phase types |
| `Sources/Flux2Core/Utils/Profiling/ChromeTraceExporter.swift` | Chrome Trace JSON export |
| `Sources/Flux2Core/Utils/Profiling/BenchmarkRunner.swift` | `BenchmarkResult` and `BenchmarkAggregator` |
| `Sources/Flux2Core/Utils/Flux2Profiler.swift` | Existing profiler singleton (bridge target) |
