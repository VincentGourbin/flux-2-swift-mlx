---
state: completed
---

# Iteration 03 Brief — OPERATION TWIN LIGHTHOUSE

**Mission:** Wire matching boundary-only telemetry into `flux-2-swift-mlx`, align pixart-swift-mlx's stale REQUIREMENTS doc with its live implementation, and lock in the shared §11 chokepoint convention across both libraries.
**Branch:** `instrumentation/03`
**Starting Point Commit:** `fba1583` (post-plan-refinement, pre-mission-init)
**Sorties Planned:** 17 (A1 + B1–B16)
**Sorties Completed:** 17
**Sorties Failed/Blocked:** 0 (B9 PARTIAL → amended same iteration; B14 attempt 1 failed compile → retried successfully on sonnet)
**Duration:** ~5 hours wall clock on 2026-05-13; ~18 sortie commits + 2 pre-start commits + 1 test-cleanup commit + 1 supervisor-metadata commit
**Outcome:** Complete
**Verdict:** `KEEP` — full emission surface wired, 215 tests passing, hard discoveries were refinements within the architecture (not invalidations of it), the single test-cleanup removal was a measurement-methodology issue rather than a code issue.
**Tests pruned:** 1 (`Flux2TelemetryNoopOverheadTests`)
**Tests flagged for review:** 0

---

## 1. Hard Discoveries

### 1. TrainingTextEncoder is a protocol, not a class

**What happened:** Plan B3 listed `TrainingTextEncoder.swift` as one of 7 classes needing the `OSAllocatedUnfairLock`-backed `setTelemetry` seam. B3's agent grepped for the class declaration and found `public protocol TrainingTextEncoder: AnyObject, Sendable` instead. Protocols can't hold stored locks.
**What was built to handle it:** Agent skipped the protocol; concrete conformers (Klein, Dev) already got their own seams. Documented in a doc-comment on the protocol. `WeightComponent.textEncoderTraining` was kept as a live but unreferenced enum case.
**Should we have known this?** Yes. A 30-second grep `^public protocol TrainingTextEncoder` in plan-time would have caught it. The plan refinement pass produced filename → class-name pairs without verifying the file's contents.
**Carry forward:** Plans referencing "the class in `<file>.swift`" must verify with `grep "^public (final )?(class|struct|protocol|actor)"` at refinement time, not at sortie time.

### 2. WeightLoader is a static-only utility class

**What happened:** Plan B3 expected an instance-level seam on `Flux2WeightLoader`. The class has every method declared `static` with no instance state.
**What was built to handle it:** B3 agent invented a static-seam variant: `private static let _telemetryLock` + `public static func setTelemetry` + `public static func currentTelemetry`. B5's static load sites then call `Flux2WeightLoader.currentTelemetry()` to emit. The pattern works and preserves the cross-library §11 seam convention.
**Should we have known this?** Yes — same fix as #1, grep the class declaration during refinement.
**Carry forward:** Document the static-seam variant as the canonical pattern for utility/namespace classes. Encode it explicitly in AGENTS.md §11 so future libraries don't reinvent it.

### 3. `Flux2WeightLoader.loadWeights` is component-agnostic

**What happened:** Plan B5 expected `weightLoadComplete` to be emitted at the load function's exit, with `component:` derived inside the loader. Reality: there's a single generic `loadWeights(from:)` that returns `[String: MLXArray]`; the caller knows whether it's loading transformer weights, VAE weights, LoRA weights, etc. The loader does not.
**What was built to handle it:** B5 emitted `.transformer` only at `loadQuantizedTransformer` (where the component is unambiguous), then deferred the `.vae` emit to the call site in `Flux2Pipeline.swift`. B6 absorbed the VAE emit at Pipeline.swift:507 alongside its primary scope.
**Should we have known this?** Yes. A grep `loadWeights\b` at plan-time would have found 5 call sites in Pipeline.swift, immediately surfacing that the loader is called with different intent per site.
**Carry forward:** For events with a "component" or "phase" argument, the planner must identify whether the discriminator lives at the *callee* or *caller* before assigning the emit. Default assumption: caller, unless the callee's name encodes the discriminator (e.g., `loadVAEWeights`, `loadQuantizedTransformer`).

### 4. VAE checkpoint preview decodes exist inside the denoise loop

**What happened:** Plan B9 expected 1 `vae.decode` call site. B9 (haiku) found 5: 2 final-output decodes (one per T2I/I2I path) and 3 checkpoint-preview decodes nested inside the denoise loops. Haiku wired all 5; user-confirmed amendment removed the 3 in-loop emits as boundary-rule violations.
**What was built to handle it:** Amendment commit `30cd25e` removed the 3 checkpoint emits, kept the 2 final emits. TELEMETRY_AUDIT.md documents the rationale: `.vaeDecodeComplete` is a *boundary* event meaning "pipeline output ready"; checkpoint decodes are intra-boundary and need their own event if observed at all.
**Should we have known this?** Possibly. A grep `vae!.decode\|vae?.decode` at plan-time would have shown 5 sites; the planner would have had to reason about which boundary each represented.
**Carry forward:** For pipeline-style libraries, all `<heavy_component>.<method>` call sites should be enumerated at plan time, with each site classified as boundary-public or boundary-internal *before* assigning an event. Internal checkpoint observability is a separate event family, not an overload of boundary events.

### 5. MLXNN module instances are private inside `FluxTextEncoders.shared`

**What happened:** Plan B5 expected to compute `paramCount` for text-encoder load emits via `.numParameters()` or `parameters.flattenedValues().reduce`. Reality: the `MistralForCausalLM` and `Qwen3ForCausalLM` MLXNN modules are stored as `private` members of `FluxTextEncoders.shared`. No public accessor exists.
**What was built to handle it:** B5 hardcoded architectural constants — Qwen3-4B = 3,953,178,624; Qwen3-8B = 8,194,392,064; Mistral Small 3.2 = 24,000,000,000 — with code comments documenting the choice.
**Should we have known this?** Yes. A grep `class .*ForCausalLM\|class .*Module` in the FluxTextEncoders target at plan-time would have surfaced the access modifier.
**Carry forward:** Future iteration should add a public `numParameters: Int` accessor on `KleinTextEncoder` / `DevTextEncoder` / `Flux2TextEncoder` and replace the three hardcoded constants. Track as a follow-up ticket; not blocking for iter-03.

### 6. Zero cancellation-check sites exist in `Sources/Flux2Core/`

**What happened:** Plan B11 documented a Q1 contingency for this case. Confirmed by B11 grep: `Task.isCancelled|CancellationError|checkCancellation|cancellationCheck` returns zero hits.
**What was built to handle it:** B11 emitted no `.generationCancelled` events; added a deferral comment near each of the 4 denoise loop start sites: `// generationCancelled emission deferred: no cancellation-check sites in pipeline as of 2026-05-13`.
**Should we have known this?** Yes — Q1 was explicitly known going in. The plan handled it correctly.
**Carry forward:** Cancellation infrastructure for Flux2 is its own scope. When it lands, the deferral comments are the anchor for adding `.generationCancelled` emits at the right sites.

### 7. Pixart's live event surface is leaner than its stale REQUIREMENTS doc

**What happened:** A1 sortie discovered pixart's `PixArtTelemetryEvent.swift` has 6 cases: `weightLoadComplete (component: .dit)`, `weightUnloadComplete`, `recipeValidated`, `recipeValidationFailed`, `numericalAnomaly`, `errorThrown`. The stale REQUIREMENTS doc described per-block / per-attention-head / kernel-detail events (patchEmbedComplete, captionProjectionComplete, siluWorkaroundExecuted) that were never implemented.
**What was built to handle it:** A1 rewrote the doc to reflect reality (9,213 bytes vs 21,678 original). Added explicit "Out of scope" section listing the per-step events the stale doc had described but never built.
**Should we have known this?** Yes, before iter-03 started. Pixart's doc being stale was actually known going in (per the plan's mission scope: "stale 18 KB verbose draft predating the minimal-boundary decision").
**Carry forward:** When a sibling library is the canonical pattern source, audit *the live implementation*, not the doc. Add a check: `wc -c REQUIREMENTS-instrumentation.md` vs `wc -l Sources/.../TelemetryEvent.swift` — disproportionate ratios signal staleness.

### 8. `MockReporter` is implemented as an `actor`, not an `OSAllocatedUnfairLock`-guarded class

**What happened:** Plan B12 described pixart's `MockReporter.swift` as a "thread-safe array of captured events behind a lock". Supervisor's prompt suggested `OSAllocatedUnfairLock<[Event]>` as the pattern. B12 agent read pixart's actual file and found `public actor MockReporter` — a Swift actor, not a lock-guarded class.
**What was built to handle it:** B12 agent matched pixart's actor pattern. `MockFlux2TelemetryReporter` is an actor; tests use `await reporter.snapshot()` to read captured events.
**Should we have known this?** Yes. The supervisor's prompt was wrong; the agent's P5 verification caught it.
**Carry forward:** Supervisor prompts about pattern shape must say "match pixart's actual file, not my description of it" — and explicitly call out the property the agent should verify (`actor` vs `class`, `lock-guarded array` vs `actor.events`).

### 9. Noop-overhead ratio measurement breaks down at clock-floor scale

**What happened:** B16 wrote `Flux2TelemetryNoopOverheadTests` with median-of-20 over 10 emits per iteration, asserting noop/nil ratio within ±10%. B16's first run showed delta=0.0 (both medians at the ~2µs clock floor). test-cleanup's re-run showed delta=0.36 — the nil-branch median can clock at 1.49µs while the noop-branch median clocks at 2.03µs, putting the ratio at 1.36. Test-cleanup removed the test.
**What was built to handle it:** Test removed at commit `fb2a266`. The architecture remains correct; the benchmarking methodology was wrong.
**Should we have known this?** Yes — the plan's Q4 carry-over flagged ±2% as tight, but neither the plan nor the supervisor flagged that 10 emits/iteration at 1-2µs total would put the medians inside the clock resolution.
**Carry forward:** A meaningful noop-overhead benchmark needs ~1000 emits per iteration so each median clears the noise floor by 2-3 orders of magnitude. Pair with a dedicated `Flux2PerformanceTests` target that's excluded from CI's gating suite (or has its own loose tolerance). Track as a follow-up.

### 10. Plan's "exactly N" emit-site counts are systematically wrong

**What happened:** Multiple plan exit criteria specified "exactly N emits":
- B7: "exactly 1 schedulerConfigured" → actual 2 sites (T2I + I2I).
- B8: "exactly 4 denoiseLoopStart / 4 denoiseLoopEnd" → correct.
- B9: "exactly 1 vaeDecodeComplete" → actual 5 sites in code, 2 post-amendment.
- B10: "exactly 3 numericalAnomaly" → actual 7 sites in code (1 textEncode + 4 denoiseLoopEnd + 2 vaeDecode).
- B11: "20 throws across 4 files (Pipeline 14, ...)" → Pipeline had 15, total still 20.
**What was built to handle it:** Each sortie absorbed the drift at runtime. Supervisor decisions logged the count adjustments.
**Should we have known this?** Yes — exit criteria should grep the codebase at plan time, not hardcode counts.
**Carry forward:** **This is the single most important process lesson.** Future plans should express emit-site exit criteria as `≥ N AND all enum cases referenced AND grep counts equal between throws and emits` — never `exactly N`. The "exactly" wording invites a refusal-to-investigate that bites at sortie time.

---

## 2. Process Discoveries

### What the Agents Did Right

#### 1. B6's shared-variable emit pattern

**What happened:** B6 (textEncodeComplete) found 4 different encoder call sites (Dev with/without upsample × Klein with/without upsample). Rather than 4 separate emits, the agent introduced a `textEncodeEncoderName` variable set per branch and emitted once at the join point. Net: 1 emit covers all 4 branches.
**Right or wrong?** Right. Reduces code surface, preserves grep-ability, keeps the emit DRY.
**Evidence:** B6 commit `b700d82`, line 1027 single emit + branch-local encoderName.
**Carry forward:** Use this pattern wherever the plan calls for "N emits per N branches that all converge". Preserve it in future Pipeline.swift work.

#### 2. P5 verification in foundation sorties

**What happened:** B2 verified `TuberiaTensorStat` symbol names by reading SwiftTuberia checkout before writing the import. B10 verified `defaultOutOfRangeThreshold` and the property list (`min/max/mean/std/hasNaN/hasInf`, NOT `stddev`). B12 verified pixart MockReporter is an actor (overriding supervisor's wrong prompt suggestion). B14 retry verified `Flux2Pipeline.init` defaults, `Flux2Model` cases, and `Flux2QuantizationConfig` presets.
**Right or wrong?** Right. Every P5 check prevented downstream compile failures.
**Evidence:** Zero `cannot find symbol` errors in the entire mission.
**Carry forward:** Keep P5 in the dispatch prompt as a top-level boundary, not an afterthought.

#### 3. Sub-agent + supervisor-coordinated build for Group 2 and Group 5

**What happened:** B4+B5 (Group 2) and B13+B14+B15 (Group 5) ran as parallel sub-agents that wrote code only; supervisor committed each separately and ran builds. This preserved per-sortie attribution and avoided build-queue contention.
**Right or wrong?** Right. When B14 (haiku) compiled broken, attribution was immediate (only B14's file was the issue; B13 and B15 were fine).
**Evidence:** B14 failure was diagnosed in one `make test-core` run; rollback was trivial (just don't commit the broken file).
**Carry forward:** This is the right pattern for parallel-write windows. Make it the default for any plan-identified parallel group.

### What the Agents Did Wrong

#### 1. B9 haiku over-wired vaeDecodeComplete at 5 sites

**What happened:** Haiku found 5 `vae.decode` call sites and wired emits at all of them. 3 of those sites are checkpoint preview decodes *inside* the denoise loop body — violating the boundary-only contract.
**Right or wrong?** Wrong. Haiku met the exit-criterion letter ("≥ 1 emit") but missed the spirit. User caught it in supervisor verification; amendment commit `30cd25e` removed the 3 in-loop emits.
**Evidence:** PARTIAL state, amendment commit, ~5 minutes of supervisor + user attention to diagnose and direct the amendment.
**Carry forward:** Haiku is reliable for exit-criterion-letter compliance but unreliable for spirit-of-the-plan compliance, especially when the plan has implicit boundaries (e.g., "boundary-only events" applies to emit site location, not just event name).

#### 2. B14 haiku attempt 1 compiled broken

**What happened:** Haiku wrote a CGImage construction call assuming `CGImage(...)` returns `CGImage`. It returns `Optional<CGImage>`. Compile error in Test 2 (which was also redundant with Test 1). File discarded; sonnet retry sidestepped CGImage entirely with `[CGImage]()`.
**Right or wrong?** Wrong. Haiku didn't grep-verify the Swift API signature. Failure pattern matches B9: literal compliance, no API awareness.
**Evidence:** Compile error visible in `make test-core` output; commit log shows the broken `feat(b14)` commit was never made — only the retry `788314b` commits.
**Carry forward:** **Haiku is NOT appropriate for code that calls Swift system APIs**. Restrict haiku to: pure-function unit tests with hardcoded inputs, mechanical pattern-matching tasks (e.g., test-cleanup), small markdown edits. Anything calling Cocoa/Foundation/CoreGraphics/CoreImage APIs needs sonnet minimum.

### What the Planner Did Wrong

#### 1. Plan assumed `TrainingTextEncoder` was a class

**Carry forward:** See Hard Discovery #1.

#### 2. Plan assumed `WeightLoader` had instance state

**Carry forward:** See Hard Discovery #2.

#### 3. Plan undercounted Pipeline.swift throws (14 expected, 15 actual)

**What happened:** `encodeReferenceImages` had 2 throws the plan didn't enumerate.
**Right or wrong?** Wrong. Plan's grep at refinement time used a stale snapshot or didn't recount after B-series edits.
**Evidence:** B11 grep showed `15 sites` in Pipeline.swift; plan said 14.
**Carry forward:** Exit criteria expressed as `grep -rc "throw Flux2Error\." == grep -rc "capture(.errorThrown"` (an equality of two grep counts) is robust to drift. The plan used this form for B11's exit criterion — that part was right; the per-file estimate was just an estimate.

#### 4. Plan didn't account for VAE checkpoint preview decodes

**Carry forward:** See Hard Discovery #4.

#### 5. Plan's noop-overhead bound was unsatisfiable at the emit rate it specified

**Carry forward:** See Hard Discovery #9.

#### 6. Plan's "exactly N" emit-count exit criteria were systematically optimistic

**Carry forward:** See Hard Discovery #10 — the single most important process lesson.

---

## 3. Open Decisions

### 1. Should `WeightComponent.textEncoderTraining` be wired or removed?

**Why it matters:** Currently a live enum case with no emit site. Dead enum cases drift over time — either they get accidentally invoked from new code (silent telemetry gap) or they confuse readers about what's instrumented.
**Options:**
- A. Remove from `Flux2TelemetryEvent.WeightComponent` enum (one-line change).
- B. Add a training-time entry point (new feature; out of iter-03 scope).
- C. Leave it; document the deferral in TELEMETRY_AUDIT.md (already done as of this brief).
**Recommendation:** C for iter-03. Re-evaluate when training-time instrumentation is in scope.

### 2. Should checkpoint preview decodes get their own event family?

**Why it matters:** Currently zero observability for preview decodes (3 sites in the denoise loop). If preview latency starts hurting users, no telemetry to diagnose.
**Options:**
- A. Add `.checkpointPreviewEmitted(stepIndex:, durationSeconds:)` event in a future iteration.
- B. Don't observe; let preview latency be an open question.
**Recommendation:** A, but in its own iteration with proper "preview pipeline" scope.

### 3. Should the noop-overhead test be rewritten with higher iteration count?

**Why it matters:** Currently zero coverage of "telemetry adds negligible overhead when reporter is noop". The test was removed because methodology was broken, not because the question is unimportant.
**Options:**
- A. Rewrite with ~1000 emits/iteration in a `Flux2PerformanceTests` target excluded from CI gating.
- B. Skip; trust the actor-snapshot pattern by construction.
**Recommendation:** A, in a follow-up iteration. The question is real (production code paths should not pay a measurable cost when telemetry is off).

### 4. Should pixart close its `errorThrown` coverage (6/13 → 13/13)?

**Why it matters:** TELEMETRY_AUDIT.md flagged pixart's partial throw coverage. Flux is 20/20; pixart is 6/13 (only recipe-validation throws are wired).
**Options:**
- A. Close in a small pixart-side iteration (estimated 1-2 sorties).
- B. Leave; pixart's other throws are weight-load/forward-internal paths that don't need the same observability.
**Recommendation:** A, but low priority — pixart's audit notes the remaining ErrorPhase cases are reserved by design.

### 5. Should `Vinetas` (the cross-library telemetry adapter) be implemented next?

**Why it matters:** TELEMETRY_AUDIT.md says "Vinetas not implemented in this iteration." The whole point of standardizing flux + pixart on §11 was to feed Vinetas. With both libraries wired and audited, Vinetas adapter is the natural next mission.
**Recommendation:** Yes — Vinetas adapter implementation should be iteration 04's scope (separate mission, new repo `intrusive-memory/Vinetas`).

---

## 4. Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| A1 | Pixart REQUIREMENTS rewrite | sonnet | 1 | ✓ Yes | Reduced doc from 21,678 → 9,213 bytes; all event cases grep-verified. |
| B1 | Add SwiftTuberia dep | sonnet | 1 | ✓ Yes | Mechanical; resolved 0.7.0 cleanly. |
| B2 | Define Flux2TelemetryEvent + Reporter | opus | 1 | ✓ Yes | 41 cases (11 + 30 nested); P5-verified TuberiaTensorStat. Caught supervisor's textEncoderMistral over-spec by following §3.1 strictly. |
| B3 | Wire setTelemetry seam | opus | 1 | ✓ Yes | Found 2 plan-vs-code surprises (TrainingTextEncoder protocol, WeightLoader static) and resolved both elegantly. |
| B4 | pipelineInit/Dispose | sonnet | 1 | ✓ Yes | Clean Q3 resolution using real stored properties. |
| B5 | weightLoadComplete (5 sites) | sonnet | 1 | ⚠️ Partial | Deferred .vae to B6 because loadWeights is generic. Plan-vs-code drift, correctly absorbed. Hardcoded text-encoder paramCount due to private MLXNN modules — known limitation. |
| B6 | textEncodeComplete + VAE carry-over | sonnet | 1 | ✓ Yes | Elegant shared-variable pattern across 4 encoder branches. |
| B7 | schedulerConfigured | sonnet | 1 | ✓ Yes | 2 sites wired (T2I + I2I), mu recomputed via existing public free function. |
| B8 | denoiseLoopStart/End (4 variants) | sonnet | 1 | ✓ Yes | Clean variant mapping; no breaks in loops so completedSteps=totalSteps is safe. |
| B9 | vaeDecodeComplete | haiku → sonnet (amend) | 2 effective | ✗ Inaccurate | Haiku over-wired 5 sites; supervisor + user caught the boundary violation; sonnet amendment removed 3 in-loop emits. Net cost: 1 haiku + 1 sonnet vs. just sonnet would have been cheaper. |
| B10 | Anomaly helper + 7 side-channels | sonnet | 1 | ✓ Yes | Helper in new file (AnomalyCheck.swift); 4 denoiseLoopEnd sites refactored from inline await to `if let telemetry` pattern. P5-verified TuberiaTensorStat property names. |
| B11 | errorThrown (20 sites) | sonnet | 1 | ✓ Yes | Plan estimated 14 Pipeline.swift throws; actual 15. Drift absorbed via grep equality. Cancellation correctly deferred per Q1. |
| B12 | MockReporter + boundary tests | sonnet | 1 | ✓ Yes | Sonnet override (over algorithmic opus) paid off. Caught supervisor's wrong-pattern suggestion (lock-class vs actor). |
| B13 | Anomaly tests | sonnet | 1 | ✓ Yes | 10 @Tests; pure-function unit tests + 2 capture tests. Strategy C deferred (no fixtures for live denoise NaN). |
| B14 | Error path tests | haiku → sonnet (retry) | 2 | ✗ Inaccurate | Haiku compiled broken (CGImage Optional + redundant Test 2). Sonnet retry was clean. |
| B15 | Lock contention XCTest | sonnet | 1 | ✓ Yes | 4 XCTest methods; F10 compliant; XCTest per F11 carry-over. |
| B16 | Noop overhead + audit | opus | 1 | ⚠️ Partial | Audit (TELEMETRY_AUDIT.md) is solid. Noop overhead test was correct in intent but broken in methodology — test-cleanup removed it. |
| _(cleanup)_ | test-cleanup pass | haiku | 1 | ✓ Yes | Correctly identified B16's noop overhead test as flaky (delta=0.36 at clock floor), removed it. |

**Inaccurate sorties:** 2 of 17 (B9 amend, B14 retry). Both haiku. Both produced compile-clean or grep-pass output that violated the spirit of the task.

**Model accuracy:**
- haiku (3 dispatches: B9, B14 attempt 1, test-cleanup): 1 ✓ (test-cleanup), 2 ✗ (B9 over-wire, B14 broken compile). Hit rate: 1/3 = 33%.
- sonnet (12 dispatches incl. retries): 12/12 = 100%.
- opus (3 dispatches: B2, B3, B16): 3/3 = 100%.

---

## 5. Harvest Summary

What I now know that I didn't going in: the flux codebase has 5 VAE decode sites (not 1); checkpoint preview decodes are a thing that crosses boundary lines; pixart's live surface is much smaller than its stale doc claimed; haiku's exit-criterion-letter compliance is reliable but its plan-spirit compliance is not (B9 over-wired, B14 compiled broken — both due to literal-pattern-matching without thinking about implication); `TrainingTextEncoder` is a protocol not a class; `Flux2WeightLoader` is static-only; `loadWeights` is component-agnostic; and noop-overhead measurement at single-emit per iteration is fundamentally inside clock resolution noise.

The single most important thing that changes about the next iteration: **exit criteria should never express emit-site counts as "exactly N"**. Use `≥ N AND all enum cases referenced AND grep equality between paired patterns`. This one wording change would have absorbed every plan-vs-code count drift this iteration produced without supervisor intervention.

Test-cleanup pruned 1 of 5 mission tests (20% of mission tests). The single removal was a measurement-methodology issue (clock-floor variance), not a systemic flakiness pattern — agents did not repeatedly write sleep-based timing tests. Conservative test-writing held throughout the mission.

---

## 6. Files

### Preserve (read-only reference for next iteration)

| File | Branch | Why |
|------|--------|-----|
| `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` | `instrumentation/03` | Live event surface; all 11 events + 5 nested enums |
| `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` | `instrumentation/03` | Public protocol + NoopReporter |
| `Sources/Flux2Core/Telemetry/AnomalyCheck.swift` | `instrumentation/03` | `classify(_:)` helper used at 7 stat-carrying sites |
| `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` | `instrumentation/03` | All boundary emit sites + setTelemetry seam + propagation |
| `Sources/Flux2Core/Loading/*.swift` (4 files) | `instrumentation/03` | Encoder seams + weightLoadComplete emits |
| `Sources/Flux2Core/Loading/WeightLoader.swift` | `instrumentation/03` | Static-seam pattern (canonical for utility classes) |
| `Sources/Flux2Core/Scheduler/FlowMatchEulerScheduler.swift` | `instrumentation/03` | Scheduler seam |
| `Sources/Flux2Core/Transformer/Flux2Transformer.swift` | `instrumentation/03` | Transformer seam |
| `Sources/Flux2Core/LoRA/LoRAAdapter.swift` | `instrumentation/03` | LoRAManager seam + .lora load emit |
| `Tests/TestHelpers/MockFlux2TelemetryReporter.swift` | `instrumentation/03` | Actor-based mock; reused by 4 test files |
| `Tests/Flux2CoreTests/Flux2TelemetryBoundaryEventsTests.swift` | `instrumentation/03` | B12 boundary tests |
| `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` | `instrumentation/03` | B13 anomaly tests (10 @Tests) |
| `Tests/Flux2CoreTests/Flux2TelemetryErrorPathTests.swift` | `instrumentation/03` | B14 error path test |
| `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` | `instrumentation/03` | B15 XCTest lock contention tests |
| `REQUIREMENTS-instrumentation.md` | `instrumentation/03` | Slim iter-03 spec |
| `AGENTS.md` §11 | `instrumentation/03` | Cross-library convention |
| `TELEMETRY_AUDIT.md` | `instrumentation/03` | B16 cross-library audit |
| `TEST_CLEANUP_REPORT.md` | `instrumentation/03` | Test-cleanup pass record |
| `EXECUTION_PLAN.md` | `instrumentation/03` | Reference plan for future iterations |
| `SUPERVISOR_STATE.md` | `instrumentation/03` | Full decision log + sortie cadence |
| `OPERATION_TWIN_LIGHTHOUSE_03_BRIEF.md` | `instrumentation/03` | This file |
| `pixart-swift-mlx:REQUIREMENTS-instrumentation.md` | `development` (pixart repo) | A1's rewrite — live surface match |

### Discard (already gone)

| File | Why it's safe to lose |
|------|----------------------|
| `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` | Removed by test-cleanup; methodology broken at clock-floor scale. Rewrite in a follow-up iteration with proper benchmarking infrastructure. |

---

## 7. Iteration Metadata

**Starting point commit:** `fba1583` (deps: pin swift-tokenizers to .upToNextMinor(0.5.0))
**Mission branch:** `instrumentation/03` (user-named; skill default would have been `mission/twin-lighthouse/03`)
**Final commit on mission branch:** `fb2a266` (test-cleanup: prune 1 non-CI-safe test added during OPERATION TWIN LIGHTHOUSE)
**Rollback target:** `fba1583` (same as starting point commit)
**Next iteration branch (recommended):** `instrumentation/04` (preserves user's naming convention) OR a fresh `vinetas-adapter/01` if the next mission is the Vinetas implementation in a different repo.

---

## 8. Rollback Verdict

**Verdict:** `KEEP`

**Reasoning:** All 17 sorties completed. The full 11-event boundary surface is wired with 43+ emit sites across `Flux2Pipeline.swift` and the Loading/+LoRA/+Scheduler/+Transformer modules. Tests pass (215 in 35 suites). The two inaccurate sorties (B9, B14) were both haiku misses that were diagnosed and recovered within the same iteration without invalidating prior or subsequent work. The 10 hard discoveries are refinements within the architecture (protocol-vs-class seam handling, static-utility seam pattern, generic-loadWeights call-site emit placement) — none invalidate the §11 convention or the chosen event-surface shape. The single test-cleanup removal (noop overhead) was a measurement-methodology issue rather than an architectural one. Cross-library audit confirms flux and pixart share zero naming drift on the event cases that exist on both sides. Pixart's stale REQUIREMENTS doc is now aligned with its live surface (A1).

**Recommended action:**
- **Merge** `instrumentation/03` into `development`, then into `main` per the project's branching flow.
- **Open follow-up tickets:**
  1. Public `numParameters: Int` accessor on `KleinTextEncoder` / `DevTextEncoder` / `Flux2TextEncoder` — replaces 3 hardcoded constants in `weightLoadComplete` emits (Hard Discovery #5).
  2. Cancellation infrastructure (Task.isCancelled checks in denoise loops) — anchor sites already documented via deferral comments at 4 denoise-loop starts (Hard Discovery #6).
  3. Rewrite of noop-overhead benchmark in a `Flux2PerformanceTests` target with ~1000 emits/iter (Hard Discovery #9).
  4. Pixart `errorThrown` coverage closure (6/13 → 13/13) — small pixart-side iteration (Open Decision #4).
  5. `Vinetas` adapter implementation — likely iteration 04 in a separate repo (Open Decision #5).
- **Bake into next breakdown/refine:** Hard Discoveries #1, #2, #3, #10 are planning-time checks (grep class declarations, identify caller-vs-callee discriminator, enumerate `.method()` call sites). Encode them into the plan-refinement checklist before iteration 04 starts.
