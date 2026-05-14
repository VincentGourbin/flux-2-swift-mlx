---
mission: flux-2-swift-mlx-instrumentation
feature_name: OPERATION SILICON STETHOSCOPE
iteration: 02
mission_branch: instrumentation/02
starting_point_commit: 16adef2063683712f40c4425daad238f3e860481
final_commit: ddcc5b01c76a9f79309d6cfe64b556b312b16acb
state: incomplete
verdict: ROLLBACK
---

# Iteration 02 Brief — OPERATION SILICON STETHOSCOPE

**Mission:** Produce a `Flux2TelemetryEvent` / `Flux2TelemetryReporter` surface in flux-2-swift-mlx so Vinetas can correlate numerical anomalies back to the kernel and step that produced them.
**Branch:** `instrumentation/02`
**Starting Point Commit:** `16adef2` (plan: iteration 02 — REQUIREMENTS + EXECUTION_PLAN with iteration-01 lessons applied)
**Final Mission Commit:** `ddcc5b0` (test-cleanup report)
**Sorties Planned:** 10 (1, 2, 3, 4, 5, 6, 7a, 7b, 8, 9, 10)
**Sorties Completed:** 9 (1–7b, 9)
**Sorties Failed/Blocked:** 1 (Sortie 8 — TSan target — deferred by user decision after platform-bug finding)
**Sorties Not Run:** 1 (Sortie 10 — release — withheld pending rollback verdict)
**Duration:** ~2.5 hours of agent runtime across 11 dispatches
**Outcome:** Incomplete
**Verdict:** `ROLLBACK` — F11's iter-01 hypothesis (TSan compatible with XCTest) was wrong; the lock-seam concurrency proof — the load-bearing safety claim of the entire instrumentation surface — is unverified. Iter-03 builds the workaround into the plan from day one.
**Tests pruned:** 0
**Tests flagged for review:** 1 (Flux2TelemetryNoopOverheadTests — CI flakiness risk; see TEST_CLEANUP_REPORT.md)

## Terminology

> **Mission** — The full instrumentation campaign.
> **Sortie** — Atomic agent task within the mission.
> **Brief** — This document. Post-mission review.
> **Rollback** — Reset to `starting_point_commit`, preserve this branch for reference, plan iter-03 with lessons applied.

---

## Section 1: Hard Discoveries

### 1. TSan + xctest bootstrap crash on macOS 26.2 SDK affects classic XCTest, not just swift-testing

**What happened:** Sortie 8 ran `make test-tsan` (xcodebuild with `-enableThreadSanitizer YES`) targeting a classic-XCTest `Flux2TelemetryLockContentionTests.xctestcase`. The xctest runner crashed at bootstrap, before any test code executed:

```
xctest (48087) encountered an error (Early unexpected exit, operation never
finished bootstrapping... The test runner crashed before establishing
connection: xctest at <external symbol>)
```

This is a runtime crash in the xctest binary itself (the SDK's, not the host OS's), independent of test-framework choice. The crash binary ships with the macOS-26.2 SDK that's bundled with Xcode 26.3. Host macOS is 26.5; only the SDK is older.

**What was built to handle it:** Nothing in iter-02 — the user elected to defer Sortie 8 to iter-03 rather than ship a half-baked workaround. The test source (a `SeamUnderTest` mirror with 3 XCTest cases) is preserved in the agent transcript at `/private/tmp/claude-501/.../tasks/af22f0a227d79304d.output`.

**Should we have known this?** Yes. Iteration 01's brief noted "TSan + swift-testing crashes on macOS 26.2 SDK"; the plan author extrapolated to "classic XCTest's older runtime is mature under TSan." That extrapolation should have been validated with a 30-second `xcodebuild test -enableThreadSanitizer YES` smoke against an existing trivial XCTest case *before* baking F11 into the iter-02 plan. The validation cost was minutes; the carry-cost is an entire rollback.

**Carry forward (iter-03 hard requirement):** TSan coverage of the lock seam must NOT depend on the xctest runner. The recommended path is a **standalone executable target** (e.g., `Sources/LockSeamTSan/`) with a `main.swift` that drives `OSAllocatedUnfairLock` under high concurrency, compiled with `-Xswiftc -sanitize=thread`. The executable exits non-zero on TSan report. `make test-tsan` invokes the executable directly. The xctest runner is bypassed entirely; TSan instrumentation lives in the Swift runtime, not in xctest, so this avoids the bootstrap crash.

### 2. Spec-vs-plan inconsistency on `pipelineDispose` case signature

**What happened:** REQUIREMENTS-instrumentation.md §3.1 line 115 declares `case pipelineDispose` (no associated value). Sortie 3's task 6 in EXECUTION_PLAN.md explicitly emits `pipelineDispose(model:)` with a model identifier. Sortie 1's agent caught the contradiction during type creation and added `(model: String)` to match the plan's emission shape. Without that adaptation, Sortie 3 would have failed to compile.

**What was built to handle it:** Sortie 1's agent reasoned through both options and chose plan-over-spec on the grounds that "bare case provides zero useful info to host adapters" — a defensible call but a deviation from the source of truth.

**Should we have known this?** Yes. The refinement passes (atomicity/priority/parallelism/questions) should have caught the spec ↔ plan mismatch. Pass 1 (atomicity) is supposed to verify each sortie's exit criteria are self-consistent; this would have surfaced the discrepancy if it had cross-referenced the REQUIREMENTS case list against the planned emission shapes.

**Carry forward:** Iter-03's refinement passes must add a spec↔plan cross-reference check: every event case the plan emits must match the case signature in REQUIREMENTS §3.1. Fix the source of truth (REQUIREMENTS) and the plan agrees by reference.

### 3. Cancellation pattern in the codebase is guard-based, not `Task.checkCancellation()`

**What happened:** Sortie 5's plan template assumed cancellation sites looked like `try Task.checkCancellation()` or `if Task.isCancelled`. The actual pattern in `Flux2Pipeline.swift` is `guard let transformer = transformer else { throw Flux2Error.generationCancelled }` — the cancellation is implicit in subcomponent invalidation, not in cooperative task cancellation. 3 such sites exist (pre-KV-extract, in-loop I2I, in-loop T2I).

**What was built to handle it:** Sortie 5's agent adapted by emitting `generationCancelled(stepIndex:)` immediately before each `throw .generationCancelled` site, inside an `if let telemetry` guard. The existing Sortie-3 `errorThrown(.generationCancelled, ...)` emissions were preserved at the same sites, so each cancellation now fires **two** events in sequence: `.generationCancelled` then `.errorThrown(phase: .generationCancelled)`.

**Should we have known this?** Partially. The plan's audit note for `errorThrown` correctly identified 14 throw sites including the 3 generationCancelled ones; that data was on hand. But the plan template for Sortie 5 didn't pivot — it kept the `Task.checkCancellation()` mental model. The collision was real, the workaround was clean, but it cost agent turns.

**Carry forward:** Host adapter docs (Vinetas) must document that for generationCancelled events, both `.generationCancelled` AND `.errorThrown(phase: .generationCancelled)` fire. The host should deduplicate by correlation ID or by treating the pair as one logical event.

### 4. Q10 "no-work baseline" makes the +1%/+5% overhead bounds structurally impossible

**What happened:** The plan's Q10 mandated "build a constant-time transformer stub that returns a deterministic pre-allocated MLXArray per step. The test measures telemetry overhead per step, not MLX kernel time." Sortie 9's first interpretation was literal: the stub does nothing. With a no-work baseline (~225 ns/step in cohort A), any cohort that performs a single lock-read + capture (cohort B) showed +200% overhead. The +1%/+5% bounds were unsatisfiable.

**What was built to handle it:** Sortie 9's agent reinterpreted Q10: the stub replaces the **MLX kernel work** with constant CPU-spin work (~1.9 ms/step), so the cohorts differ only in telemetry overhead. Bounds became measurable. Local results: NOOP=-0.077%, MOCK=-0.064% (below the noise floor, well within bounds).

**Should we have known this?** Yes. The plan author should have included an explicit "baseline-A must perform realistic per-step work to make the proportional bound meaningful." This is the kind of measurement-design detail that gets glossed in a spec but bites in implementation.

**Carry forward:** The +1%/+5% bound covers **lock + capture + actor-dispatch** overhead only. The cohorts use a pre-allocated `TuberiaTensorStat`, so the test does NOT measure `TuberiaTensorStat.sample()` per-step cost — which traverses the MLXArray to compute mean/std/min/max and is the largest plausible production overhead. Iter-03 should either widen Sortie 9's scope to include real `.sample()` calls per step OR explicitly scope the contract to "reporter dispatch" and document the omission in the PR.

### 5. `TuberiaTensorStat` init parameter order differs from plan templates

**What happened:** Plan templates and the supervisor's prompt templates used the order `(shape, dtype, mean, std, min, max, hasNaN, hasInf)`. The actual `TuberiaTensorStat.init` in SwiftTuberia v0.7.0 (verified by Sortie 7a's agent against `/Users/stovak/Projects/SwiftTuberia/Sources/Tuberia/Telemetry/TuberiaTensorStat.swift`) is `(shape, dtype, min, max, mean, std, hasNaN, hasInf)`. Min/max precede mean/std.

**What was built to handle it:** Sortie 7a's agent compared against the source and used the correct order. Sortie 7b/9 followed the corrected order.

**Should we have known this?** Yes, with one minute of `grep` against the dep. The refinement pass should have validated all external-API references against actual symbol surfaces.

**Carry forward:** Iter-03's "API surface verification" step should be more explicit — every external symbol the plan references gets a grep-against-dep verification line in the prompt template. F9-style.

### 6. `denoiseStepComplete` uses `Float` for `sigma`/`timestep`, not `Double`

**What happened:** Plan/supervisor templates assumed Double; the actual case signature is `sigma: Float, timestep: Int`. Sortie 7b's agent caught this and corrected loop arithmetic to Float before building.

**Should we have known this?** Yes — same root cause as Hard Discovery 5. Symbol verification against source.

**Carry forward:** Single fix for #5 and #6: an automated "verify case signatures" step in refinement.

### 7. EXECUTION_PLAN.md frontmatter edits get reverted by sub-agents under "DO NOT modify X"

**What happened:** Sortie 5's agent saw the supervisor's unstaged frontmatter additions (`mission_branch`, `starting_point_commit`) and `git checkout`-ed them away, citing the no-modify rule. The supervisor's audit-trail metadata was destroyed mid-mission. SUPERVISOR_STATE.md preserved equivalent info, so the damage was small but real.

**What was built to handle it:** Mid-mission feedback memory saved (`feedback_subagent_unstaged_changes.md`); Sortie 6+ prompts were rewritten with explicit "preserve pre-existing unstaged edits" language. No recurrence after the fix.

**Should we have known this?** No — this is a genuinely subtle agent-behavior pattern that's hard to predict. Worth carrying as a permanent guard.

**Carry forward:** Either commit supervisor metadata before dispatching the first sortie (cleanest), OR include "leave pre-existing unstaged edits alone" in every dispatch prompt.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 2.1 Per-sortie compile+test gate proved load-bearing
**What happened:** Iter-01 deferred all builds to a convergence step and accumulated 5 build-breaks-at-the-end. Iter-02's hard rule — every code-touching sortie ends with `make build` + `make test` green before commit — caught issues immediately. Every commit in iter-02 was buildable. Zero convergence build-breaks.
**Right or wrong?** Decisively right. The single biggest structural difference vs iter-01.
**Evidence:** 9 successful commits, each with a clean `make build` + `make test`. Zero retries due to build breaks.
**Carry forward:** Make this a permanent supervisor rule, not iteration-specific.

#### 2.2 Sub-agents made smart adaptations without escalating
**What happened:** Multiple instances:
- Sortie 4 added `import Tuberia` to Flux2Pipeline.swift when it discovered TuberiaTensorStat wasn't transitively visible.
- Sortie 5 caught that the cancellation pattern was guard-based, not Task.checkCancellation, and adapted.
- Sortie 6 refactored Sortie 3's in-loop pre-throw guards to reuse the cached `telemetry` instead of re-acquiring (preserved the hot-path "once per step" rule).
- Sortie 6 caught grep-pattern brittleness in my exit criteria and verified via `\.eventName(` instead of `eventName`.
- Sortie 7a verified TuberiaTensorStat init order against source and adapted templates.
- Sortie 9 reinterpreted Q10 to give the bounds a measurable headroom.
**Right or wrong?** Right. Agent autonomy on micro-decisions (single-file adaptations) saved BACKOFF cycles.
**Evidence:** Zero BACKOFF events across 10 successful dispatches. (Sortie 8's PARTIAL was platform-bug-driven, not agent failure.)
**Carry forward:** Trust agents on micro-decisions but require them to surface deviations in the report. Iter-02's "NOTES" section in every sortie report worked well.

#### 2.3 Model selection by complexity score paid off
**What happened:** Sorties 3, 4, 7b, 8 used sonnet ($1/10 the cost of opus). Sorties 1, 2, 5, 6, 7a, 9 used opus. Sonnet was sufficient for mechanical site-enumeration work (Sortie 3's 14-throw-site mapping was the most mechanical); opus was reserved for foundation work (1, 7a), concurrency primitives (2), hot-path discipline (6), variable-name discrimination at iter-01-failure sites (5), and the public-contract overhead claim (9).
**Right or wrong?** Right. Cost spread was healthy without compromising correctness.
**Evidence:** No sonnet sortie had to be BACKOFF-upgraded to opus. The cheap dispatches stayed cheap.
**Carry forward:** Keep the complexity-score override rule. Foundation_score=1 + dep_depth≥5 → forced opus is a good guard.

### What the Agents Did Wrong

#### 2.4 Sortie 5 sub-agent destroyed supervisor frontmatter
See Hard Discovery 7. Addressed mid-mission; not recurring.

#### 2.5 Multi-line emission formatting broke literal-grep exit criteria
**What happened:** Several agents formatted multi-arg `await telemetry.capture(.eventName(...))` calls across multiple lines. My exit criteria used `grep -c 'telemetry.capture(.eventName'` which doesn't match split-line emissions. Sortie 3 and Sortie 6 both bumped into this. Both agents caught the pattern brittleness and verified by counting bare event-case mentions instead.
**Right or wrong?** Wrong on the planner (me) for writing brittle exit-criteria. The agents adapted correctly. The cost was about a paragraph of "let me re-verify with the right pattern" in each report.
**Evidence:** Sortie 3 reported 14 errorThrown emissions but my supervisor grep returned 0 (I had to do my own grep-with-anchored-pattern to confirm). Sortie 6 explicitly called out the pattern brittleness in its report.
**Carry forward:** Exit criteria use `grep -E "\.eventName\b"` or `grep "eventName("` or similar anchored patterns. Never `grep "telemetry.capture(.eventName"` — the open paren defeats line-broken formatting.

### What the Planner Did Wrong

#### 2.6 F11 was an assumption, not a verified fact
See Hard Discovery 1. The single most expensive planning miss.

#### 2.7 Q10's no-work baseline was a measurement design bug
See Hard Discovery 4. Less costly than F11 because Sortie 9 recovered cleanly.

#### 2.8 Spec-vs-plan inconsistencies weren't caught in refinement
See Hard Discoveries 2, 5, 6. The refinement passes (atomicity/priority/parallelism/questions) didn't include a "verify external API signatures" or "cross-reference plan emissions against spec cases" step.

#### 2.9 Plan templates assumed `Task.checkCancellation()` without reading the source
See Hard Discovery 3. Pattern: when the plan describes "every X site," validate by grep against `Sources/` *during refinement*, not at dispatch time.

---

## Section 3: Open Decisions

### 3.1 How does iter-03 get TSan coverage of the lock seam?

**Why it matters:** The OSAllocatedUnfairLock seam is the load-bearing safety claim of the entire instrumentation surface. Without TSan verification, "no data races" is an unverified assertion based on Swift's formal memory-ordering guarantees alone. For a public contract that Vinetas will pin to, this is too thin.

**Options:**
- **A. Standalone executable target with raw `-sanitize=thread`** (recommended). Create `Sources/Flux2LockSeamTSan/main.swift` as an `.executableTarget` that drives `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` under high concurrency (10 setters cycling through reporters + 100 readers + 10 nil-setters via a TaskGroup). Compile with `-Xswiftc -sanitize=thread`. The executable runs directly, no xctest involvement. TSan instrumentation lives in the Swift runtime; the bootstrap crash was specific to the xctest runner binary. `make test-tsan` invokes the executable and checks exit code. Estimated effort: ~30 min sortie.
- **B. Wait for newer Xcode/SDK.** Block iter-03 indefinitely on Apple shipping a fixed xctest runner. Unacceptable for a near-term release.
- **C. Move lock-contention tests to a target with no swift-testing tests.** The agent's "Option B" from Sortie 8's report — theory: the bootstrap crash may be specific to mixed test-framework targets. Lower confidence than A; same goal.

**Recommendation:** A. Standalone executable target.

### 3.2 Should Sortie 9's overhead test be CI-gated?

**Why it matters:** TEST_CLEANUP_REPORT.md flagged `Flux2TelemetryNoopOverheadTests.testOverhead` for CI flakiness — +1% bound = ~19μs headroom on a 1.9ms baseline; a busy macos-26 GitHub runner can easily eat 19μs of scheduler jitter. Local results showed the test passing comfortably (negative overhead, below noise floor), but local-quiet ≠ CI-loaded.

**Options:**
- **A. Move to `make test-gpu`** (local-only target, like existing Metal tests). Test runs locally for the public-contract claim; CI skips it. PR description still cites the local numbers.
- **B. CI skip via `ProcessInfo.processInfo.environment["CI"]`**. Same as A but the test stays in the default target. Simpler.
- **C. Loosen bounds to +5%/+15% for CI tolerance.** Weakens the public contract.

**Recommendation:** B. CI skip. The test still exists for local validation runs; CI does not gate on it. Iter-03's release PR cites the local numbers as the contract.

### 3.3 Does iter-03 widen Sortie 9's scope to include `.sample()` per-step?

**Why it matters:** The current overhead test measures lock + capture + actor-dispatch only. `TuberiaTensorStat.sample()` per-step cost is the largest plausible production overhead (it traverses the MLXArray to compute mean/std/min/max) and is not covered. The +1%/+5% claim is partial.

**Options:**
- **A. Widen Sortie 9 to include `.sample()` per step.** Allocate a real MLXArray once, sample it 3× per step in cohorts B and C. Bounds may need to relax (sampling is ~tens of microseconds).
- **B. Keep current scope, document the gap.** Cite "reporter-dispatch overhead" in the PR contract. Add a separate, looser-bounded benchmark for `.sample()` cost in a follow-up.
- **C. Drop the overhead test entirely from iter-03's deliverables.** Defer to a dedicated perf-test workstream.

**Recommendation:** B for iter-03, plan A for iter-04. Iter-03's focus should be the TSan workaround, not expanding test scope.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| 1 | Telemetry types + reporter protocol | opus | 1 | YES | 2 spec-vs-plan deviations resolved cleanly (NoopReporter struct vs class; pipelineDispose carries model). |
| 2 | Lock seam + 7-type propagation | opus | 1 | YES | 7/7/7 grep counts exact. Propagation at Flux2Pipeline:138-142. VAE clean. |
| 3 | Weight-load + LoRA + init + error emissions | sonnet | 1 | YES | 14 throw + 14 errorThrown. F2/F7 honored. Refactored later by Sortie 6 (in-loop pre-throw guards), but the refactor preserved Sortie 3's counts. |
| 4 | Text encoder + VLM + scheduler | sonnet | 1 | YES | 4 textEncoder pairs (vs plan's "≥3"), 4 VLM pairs (vs "≥1"). Added `import Tuberia`. |
| 5 | VAE-decode + anomaly + cancellation | opus | 1 | YES | F6 honored (2 emit, 3 silent). New Flux2AnomalyDetector. 12 numericalAnomaly retrofits. Reverted supervisor frontmatter (damage limited). |
| 6 | Hot-path denoise emissions | opus | 1 | YES | Critical-path peak. 4/4/4 triplets. 1 currentTelemetry()/loop. Refactored Sortie 3's in-loop guards to maintain hot-path discipline. |
| 7a | MockTelemetryReporter + 2 contract test files | opus | 1 | YES | Public actor mock. F8/F9/F10. TuberiaTensorStat init order verified against source. |
| 7b | KV cache + anomaly + VAE-denorm tests | sonnet | 1 | YES | 3 files, 11 tests. Confirmed sigma/timestep are Float. |
| 8 | Lock-contention XCTest + make test-tsan | sonnet | 1 | NO (DEFERRED) | Test code itself is sound (3 tests pass <5ms without TSan). The `make test-tsan` invocation fails because F11 was wrong. The accurate output: a finding that disproved an iter-01 hypothesis. Not committed. |
| 9 | Baseline overhead test | opus | 1 | YES (with caveat) | Reinterpreted Q10 to add a CPU-spin baseline. Bounds met. Scope caveat: covers dispatch overhead, not `.sample()` cost. |
| 10 | Release | — | 0 | N/A | Not run. Withheld pending verdict. |

**Cleanup pass:** `ddcc5b0` — 0 deletions, 1 flagged (Sortie 9's overhead test for CI flakiness). Ran on opus rather than the spec'd haiku because the agent dispatched ran haiku internally but committed as opus — irrelevant given the substantive output.

---

## Section 5: Harvest Summary

The single most important thing iter-02 proved is that the **per-sortie compile+test gate eliminates convergence build-breaks**, which was iter-01's headline failure mode. The single most expensive thing iter-02 missed is that **F11's iter-01 hypothesis was extrapolated, not verified** — and that miss is what kills the release. We now know: (1) the lock seam needs TSan coverage via a standalone executable, not via xctest; (2) the spec ↔ plan cross-reference must be a refinement step, not an assumption; (3) plan templates that name "every X site" must be validated by grep against `Sources/` during refinement, not at dispatch. Test cleanup pruned no tests but correctly flagged the overhead test for CI flakiness — a legitimate carry-forward for iter-03's plan.

---

## Section 6: Files

### Preserve (read-only reference for iter-03)

| File | Branch | Why |
|------|--------|-----|
| `OPERATION_SILICON_STETHOSCOPE_02_BRIEF.md` | docs/incomplete/silicon-stethoscope-02/ | This document — iter-03's primary input. |
| `OPERATION_SILICON_STETHOSCOPE_01_BRIEF.md` | docs/incomplete/silicon-stethoscope-01/ | iter-01 brief; chains forward. |
| `EXECUTION_PLAN.md` | instrumentation/02 (preserved local branch) | The plan we executed against. Iter-03's EXECUTION_PLAN inherits from this. |
| `REQUIREMENTS-instrumentation.md` | instrumentation/02 | Source spec. Needs iter-03 erratum for the `pipelineDispose(model:)` case signature. |
| `TEST_CLEANUP_REPORT.md` | instrumentation/02 | Flags the overhead test for CI flakiness; carry into iter-03 decision 3.2. |
| Agent transcript: `/private/tmp/claude-501/-Users-stovak-Projects-flux-2-swift-mlx/eade37ec-207c-433b-96d5-573a20658c03/tasks/af22f0a227d79304d.output` | (external, OS temp) | Sortie 8's full test source. iter-03 can lift the `SeamUnderTest` mirror + LocalMock into the new standalone executable target. **Copy this out of /tmp before it ages out** — it's the only preserved copy of Sortie 8's code. |

### Discard (will not exist after rollback)

| File | Why it's safe to lose |
|------|----------------------|
| All commits `85e8ade..ddcc5b0` (9 mission commits + 1 test-cleanup commit) | Each individual deliverable is reproducible by iter-03 from the same EXECUTION_PLAN.md + REQUIREMENTS-instrumentation.md. The branch `instrumentation/02` is preserved locally for code-archeology if iter-03 wants to copy specific approaches. |
| `SUPERVISOR_STATE.md` | Working file. Iter-03 will create its own. |

---

## Iteration Metadata

**Starting point commit:** `16adef2` (`plan: iteration 02 — REQUIREMENTS + EXECUTION_PLAN with iteration-01 lessons applied`)
**Mission branch:** `instrumentation/02` (preserved locally after rollback)
**Final commit on mission branch:** `ddcc5b0` (test-cleanup report)
**Rollback target:** `16adef2` (same as starting point)
**Next iteration branch:** `instrumentation/03` (per the plan's convention; not `mission/silicon-stethoscope/03`)

---

## Section 8: Rollback Verdict

**Verdict:** `ROLLBACK`

**Reasoning:** Nine out of ten sorties produced clean, buildable, tested code. The instrumentation surface itself is structurally sound: 7-type lock seam, correct event emissions across the entire pipeline lifecycle, hot-path discipline preserved, anomaly retrofits in place, overhead bounds met (within their scope). **But the lock seam is unverified under TSan** — and that is the load-bearing safety claim of the whole surface. Iter-01 hypothesized that classic XCTest would work under TSan on macOS 26.2; iter-02 proved that hypothesis wrong. Shipping iter-02 means asking Vinetas to pin against a "no data races" claim backed only by Swift's formal guarantees, with no end-to-end verification. That is the kind of liability a brief is supposed to surface — and it's the user's explicit call (per the prompt that interrupted the release): "I want to roll back and find a way around the TSan issue in sortie 8." Iter-03 builds the standalone-executable-TSan workaround into the plan from day one. Iter-02 was 90% of the way there; iter-03 finishes it.

**Recommended action:**
- Run the Rollback Ritual (see brief.md § Rollback Ritual): reset to `16adef2`, preserve `instrumentation/02` locally, create `instrumentation/03`.
- Top 3 things iter-03 must do differently:
  1. **Sortie 8 becomes "Standalone executable target with raw TSan."** New `.executableTarget` at `Sources/Flux2LockSeamTSan/`. `main.swift` drives `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` under high concurrency. Compile with `-Xswiftc -sanitize=thread`. Exit non-zero on TSan report. `make test-tsan` invokes the executable directly. Bypass xctest entirely.
  2. **Refinement adds a spec ↔ plan cross-reference pass.** Every event case the plan emits is verified against the REQUIREMENTS §3.1 case signature; every external API symbol the plan references is verified by grep against the dep's source. Catches Hard Discoveries 2, 5, 6 before dispatch.
  3. **Plan templates that name "every X site" must be validated by grep during refinement.** "Every throw site," "every cancellation site," "every load function" — each gets a grep audit in the refinement output, not at dispatch time. Catches Hard Discovery 3 (cancellation guard pattern) before dispatch.

The verdict token in the header (`ROLLBACK`) and in this section (`ROLLBACK`) agree.

---

## Addendum — Post-Rollback Scope Revision (2026-05-12, on `instrumentation/03`)

After the rollback completed and `instrumentation/03` was created, the user asked a question that should have been asked before iter-01 ever existed: **"Why do we need TSan? None of the other projects need it."**

The honest answer is that we don't.

- `OSAllocatedUnfairLock<T>` is Apple's standard primitive for making `@unchecked Sendable` pointer state safe. Atomic read-modify-write, memory barriers, the entire contract is documented and guaranteed.
- Swift 6 strict concurrency (`swift-tools-version: 6.2`) statically verifies that everything else in `Flux2Pipeline` is properly `Sendable`. The `@unchecked Sendable` annotation only applies because of the lock.
- The functional contract tests from Sorties 7a/7b verify the event surface (ordering, anomaly retrofits, kvCacheHit policy).
- The lock-contention tests written for the deferred Sortie 8 (`SeamUnderTest` mirror, 3 cases) pass cleanly in <5ms **without TSan**. They verify observable correctness (last-writer-wins, no torn reads, no crashes under high concurrency) via behavior — which is the actual contract hosts care about.

**TSan would prove nothing the existing pieces don't already prove.** It was carry-over from iter-01's plan, never load-bearing. The whole reason this iteration rolled back was a platform bug in a tool we don't actually need.

### Revised plan for iter-03 (supersedes Section 8 recommendation #1)

1. **Sortie 8 becomes**: lift `Flux2TelemetryLockContentionTests` from the iter-02 agent transcript at `/private/tmp/claude-501/-Users-stovak-Projects-flux-2-swift-mlx/eade37ec-207c-433b-96d5-573a20658c03/tasks/af22f0a227d79304d.output` and ship as plain swift-testing tests in `Tests/Flux2CoreTests/`. **No TSan flag, no `make test-tsan` target, no standalone executable, no separate Makefile work.** The tests run in CI under the regular `make test` target.
2. **Sortie 10's PR description cites Apple's `OSAllocatedUnfairLock` formal guarantees as the correctness claim.** No platform-bug caveat needed because we're not using xctest+TSan anymore.
3. **Carry-overs from iter-02's other lessons remain valid**:
   - Refinement adds a spec↔plan cross-reference pass (Hard Discoveries 2, 5, 6).
   - "Every X site" templates get grep-validated during refinement (Hard Discovery 3).
   - Per-sortie compile+test gate is retained (the headline success of iter-02 structure).
   - Exit-criteria grep patterns use anchored forms like `\.eventName(` instead of brittle `eventName(.` (process discovery 2.5).
   - Spec erratum for `pipelineDispose(model: String)` is applied to REQUIREMENTS-instrumentation.md before breakdown (Hard Discovery 2).

### Iter-03 sortie count

Was 10. Becomes **9** — same scope minus the TSan target.

### Iter-03 critical path

Was `1 → 2 → 6 → 7a → 10` (5 sorties). Unchanged.

### Why this addendum exists rather than rewriting Section 8

Section 8's reasoning is correct given its premise. The premise itself ("we need TSan coverage") was wrong, and that's worth recording transparently. Iter-03's `breakdown`/`refine` will produce a new `EXECUTION_PLAN.md` that bakes this scope revision in; this addendum is the bridge.

This addendum was written **on `instrumentation/03`** after the rollback ritual. The copy of this brief on `instrumentation/02` (the archived original) does **not** contain this addendum — it remains the contemporaneous iter-02 debrief. Both copies are correct for their respective contexts.
