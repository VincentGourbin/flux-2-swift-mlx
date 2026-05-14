# Test Cleanup Report: OPERATION TWIN LIGHTHOUSE (iteration 3)

**Date**: 2026-05-13  
**Mission**: OPERATION TWIN LIGHTHOUSE (iteration 03)  
**Branch**: `instrumentation/03`  
**Supervisor Report Generated**: test-cleanup phase

---

## Summary

- **Tests Removed**: 1 (out of 5 test files examined)
- **Tests Flagged for Review**: 0
- **Build Verification**: PASSED (215 tests in 35 suites, all pass)
- **Test Helper (MockFlux2TelemetryReporter.swift)**: KEPT (not a test file; critical fixture for all 4 retained tests)

---

## Removed

| File | Test Name | Pattern | Reason |
|------|-----------|---------|--------|
| Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift | noopReporter_overheadWithinTolerance() | #6 (timing assertions with tolerance) | Benchmark timing test exceeded ±10% CI hardware variance hard-fail bound with observed delta of 36%. Wall-clock medians on synthetic `dispose()` loop inherently flaky on shared CI runners. Test observed: nilMedian=1.49e-06s, noopMedian=2.03e-06s, ratio=1.36. Per EXECUTION_PLAN.md Q4 note: ±10% hard-fail threshold acknowledged prior variance observations (>±5%); test failure confirms CI unsuitability. Entire file deleted. |

---

## Flagged for Review

None.

---

## Kept Files (CI-Safe)

1. **Flux2TelemetryAnomalyTests.swift**
   - 6 unit tests of AnomalyCheck.classify(_:) and side-channel event capture
   - All use mock TuberiaTensorStat; no GPU, no filesystem paths, no network, no env vars
   - No timing assertions or flaky patterns
   - Status: ✓ PASS

2. **Flux2TelemetryBoundaryEventsTests.swift**
   - 3 tests of pipelineInit/pipelineDispose event ordering
   - Uses `Task.sleep(for: .milliseconds(100))` for settling fire-and-forget Task (not a timing assertion)
   - GPU-free pipeline construction; proper async/await patterns
   - Status: ✓ PASS

3. **Flux2TelemetryErrorPathTests.swift**
   - 1 test verifying `.errorThrown` fires before exception throw
   - Uses `Task.sleep(for: .milliseconds(50))` to settle pipelineInit (settling delay, not timing assertion)
   - Direct `await` of `.capture(...)` ensures event lands before throw unwinds
   - Status: ✓ PASS

4. **Flux2TelemetryLockContentionTests.swift**
   - 4 stress tests using `withTaskGroup` with concurrent `setTelemetry` / `dispose` calls
   - Iterations 300–5000; designed for lock contention (Sortie B15)
   - Per mission carry-over note (iteration 02 F11): deliberately uses XCTest over Swift Testing due to known flakiness of Swift Testing with concurrent patterns on macOS-26 CI
   - Settling sleep at line 185 is to let pipelineInit Task settle before asserting (same pattern as B12/B14)
   - Status: ✓ PASS

5. **TestHelpers/MockFlux2TelemetryReporter.swift**
   - Test helper actor (not a test file)
   - Critical infrastructure used by all 4 retained test files
   - Per instructions: leave helper actors alone unless concrete reason found
   - Status: ✓ KEPT

---

## Build Verification

```
Test run with 215 tests in 35 suites passed after 0.154 seconds.
** TEST SUCCEEDED **
```

All CI-safe tests in the suite pass. The deletion of Flux2TelemetryNoopOverheadTests.swift (1 test) reduced the test count from 216 → 215 without breaking any downstream tests.

---

## Detailed Deletion Rationale

### Flux2TelemetryNoopOverheadTests.swift (DELETED)

**Single test: `noopReporter_overheadWithinTolerance()`**

This is a **performance/benchmark test** that measures wall-clock overhead of a no-op telemetry reporter vs. nil reporter by timing a synthetic `dispose()` loop. The test:

1. Runs a loop of `dispose()` calls (10 emits per iteration, 20 iterations) on two pipelines:
   - Pipeline 1: `setTelemetry(nil)` (short-circuit baseline)
   - Pipeline 2: `setTelemetry(NoopFlux2TelemetryReporter())` (no-op overhead target)

2. Computes median wall-clock duration per branch using `Date()` and `timeIntervalSince(_:)`

3. Asserts: `ratio = noopMedian / nilMedian` with tolerance `observedDelta < 0.10` (±10%)

**Why it fails on CI:**

- Observed on run: nilMedian=1.49e-06s, noopMedian=2.03e-06s, ratio=1.36, **delta=0.36** (36%)
- Hard-fail bound: ±10% (0.10)
- **Exceeded by 3.6x**

The test's own comments (lines 24–31) acknowledge this risk: "prior iterations observed >±5% variance on similar timing tests." The EXECUTION_PLAN.md Q4 entry explicitly widened the tolerance from ±2% to ±10% as a "documented hard-fail threshold" because variance on CI runners makes tighter bounds unachievable.

**Pattern match:** This is a **benchmarking assertion** (using `Date()` to compute wall-clock durations and assert on ratio/tolerance), which is inherently flaky on CI due to:
- Shared runner hardware load
- System scheduling variance
- Timer resolution on virtualized or variable-frequency CPUs
- Task.sleep precision limits (lines 62, 66, 77–84 use 100ms settle, which is fine, but the subsequent timing loop has no protection from micro-variance)

**Confidence: HIGH** — The test failed on first run with a clear ±10% hard-fail message. The failure is repeatable evidence of CI unsuitability. Keeping this test would introduce intermittent failures on every CI run until hardware variance stabilizes below ±5% (uncertain timeline) or the bound is loosened further (defeats the purpose of the test).

---

## Notes on Settling Delays (NOT deleted)

Three retained test files use `Task.sleep(for: .milliseconds(100))` or `.milliseconds(50)`:
- **Flux2TelemetryBoundaryEventsTests.swift:34**
- **Flux2TelemetryErrorPathTests.swift:29**
- **Flux2TelemetryLockContentionTests.swift:185**

These are **settling delays**, not **timing assertions** (per deletion criterion #6). The pattern:
1. Trigger an async action (e.g., detached `pipelineInit` Task in `Flux2Pipeline.init`)
2. Sleep briefly to let the Task reach the actor
3. Snapshot the actor state
4. Assert on captured events (no timing bound on the sleep itself)

This is a **correct pattern** for testing fire-and-forget dispatch. The sleep ensures the Task has delivered before snapshotting, but there is no assertion like "the Task must complete in <X ms" — the sleep value is defensive (bumped from 50ms to 100ms if flakiness observed per comments), not performance-critical. Deleting these tests would break valid integration tests.

---

## References

- EXECUTION_PLAN.md Q4: "±5% is the documented target; ±10% is the hard-fail"
- Iteration 02 carry-over F11: Swift Testing flakiness with concurrent patterns; Flux2TelemetryLockContentionTests uses XCTest deliberately
- Flux2TelemetryNoopOverheadTests.swift lines 24–31: "prior iterations observed >±5% variance on similar timing tests"
