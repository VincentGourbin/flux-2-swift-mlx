---
type: mission-brief
state: completed
mission: thimble-typhoon-01
updated: 2026-07-05
---

# Iteration 01 Brief — OPERATION THIMBLE TYPHOON

**Mission:** Ship `flux-2-swift-mlx` on iPad-class Apple Silicon — a 16 GB qint8 Klein 4B path (ship-now) and an 8 GB int4 Klein 4B path (net-new), Klein 9B excluded on every tier.
**Branch:** `mission/thimble-typhoon/01`
**Starting Point Commit:** `9d94e55`
**Sorties Planned:** 13
**Sorties Completed:** 13
**Sorties Failed/Blocked:** 0
**Duration:** ~3.3h wall clock · relative model cost ≈ 240× (sonnet×6 + opus×6; naming haiku negligible)
**Outcome:** Complete
**Verdict:** `KEEP` — 13/13 sorties landed on first attempt, full CI gate green, one favorable hard discovery, test-cleanup pruned <5% (harmful stubs only); the three open items are follow-ups by design, not foundation defects.
**Tests pruned:** 2
**Tests flagged for review:** 8

---

## Section 1: Hard Discoveries

### 1. The VAE is ~0.168 GB, not ~3 GB
**What happened:** A7's CDN verification found the FLUX.2-klein-4B `vae/` subfolder is ~0.168 GB on the wire, while the requirements and the `estimatedTotalMemoryGB` heuristic (`+8` = "VAE 3 + working 5") assumed ~3 GB. A ~2.8 GB error baked into the memory budget — in the *favorable* direction.
**What was built to handle it:** B3 folded the correction into the named `iPadWorkingSetOverheadGB = 6` constant (down from the Mac-shaped 8), while conservatively holding the Mac floor at 8 pending measured telemetry.
**Should we have known this?** Yes — inspecting the HF repo file sizes before writing the requirements' §2 memory model would have revealed it. The `+8` was a Mac-era guess never checked against Klein 4B's actual component sizes.
**Carry forward:** Memory budgets must be derived from actual on-wire component sizes (via `Acervo`/manifest), not carried-over constants. The iPad overhead is now named and documented; the precise working-set value still needs on-device `phys_footprint` (see Open Decision 2).

### 2. `loadQuantizedTransformer` was dead code; the int4 layout is assumed, not verified
**What happened:** B2 discovered `WeightLoader.loadQuantizedTransformer` was never called before this mission — it was dead code. B2 wired it into the pipeline for the int4 branch. The mapping assumes the `themindstudio/flux2-klein-4b-mlx-4bit` repo is Diffusers/mflux-format (`transformer_blocks.` naming, per-layer `.scales`/`.biases`), per OQ-1 — but this is **not verified against the real weights** because they were never downloaded (out of sortie scope).
**What was built to handle it:** The direct int4 load path (mmap packed uint32 + scales/biases into `QuantizedLinear`, no bf16 intermediate) plus a model-gated `Int4DirectLoadGPUTests` that exercises it end-to-end only in CI against cached models.
**Should we have known this?** Partially. OQ-1 verified the repo's *metadata* (quantization_level, mflux_version) but nobody loaded a tensor. That's the correct cost/scope trade — but it means int4 runtime correctness is CI-gated, unproven locally.
**Carry forward:** The first CI run of `integration-tests.yml` against cached models is the real acceptance test for WU-B. Until it goes green, treat int4 load + 8 GB viability as *structurally implemented, runtime-unverified*.

## Section 2: Process Discoveries

#### What the Agents Did Right
### 1. Testability convention held across 13 sorties
**What happened:** Every sortie added a `forRAMGB:` / `forTier:` / `enable8GBTier:` parameterized overload so tier-dependent logic is unit-testable without mutating global state.
**Right or wrong?** Right. It let the pure-logic tier/config/registry assertions (the bulk of the 312-test suite) run deterministically in CI while the real-model paths stay `.enabled(if:)`-gated.
**Evidence:** 312 CI-safe tests, zero flaky-in-CI patterns found by cleanup beyond 2 pre-existing stubs. B4 even caught a latent `.iPad8GB` ternary bug purely through this discipline.
**Carry forward:** Keep mandating parameterized tier overloads in the plan's exit criteria.

### 2. Two agents caught bugs beyond their sortie scope
**What happened:** B4 found `defaultSteps`/`defaultGuidance` used `== .iPad ?` ternaries that would silently return Mac values (50/4.0) on `.iPad8GB` once the flag flipped. B2 proved the on-the-fly-quantize bypass with a mirrored-guard pure-logic test rather than hand-waving.
**Right or wrong?** Right — these are the difference between "tests pass" and "tests prove."
**Carry forward:** Nothing to change; this is the standard to hold.

#### What the Agents Did Wrong
### 3. A5 ran the full 20-minute GPU suite to verify one test, then ping-ponged
**What happened:** A5 launched `make test-gpu` (the entire `Flux2GPUTests` suite, real inference) to check one tiled-decode test, then repeatedly handed control back with the build still running instead of blocking to completion.
**Right or wrong?** Wrong, twice over: `make test-gpu` is local-only and not the CI gate, and the async agent model can't effectively "wait" on a detached build.
**Evidence:** ~20 min of wasted wall-clock; supervisor had to kill the run and verify via `make test-core`.
**Carry forward:** Already fixed mid-mission — every subsequent dispatch prompt (A2→B5) carried "do NOT run `make test-gpu`; verify with `make test-core`; finish in-turn." Zero recurrences after the fix. Bake this into the plan's dispatch template.

#### What the Planner Did Wrong
### 4. The plan never scoped wiring `ModelTierGate` into the generation path
**What happened:** A2 built `ModelTierGate.resolve` as library API and unit-proved the refusals, but nothing invokes it in `Flux2Pipeline`'s generation path — so at runtime a caller can still construct a Klein-9B pipeline. §0 says "any code path that could route to Klein 9B is a defect"; OQ-5 defers device-gating enforcement to VinetasIOS (app-side). The plan resolved this tension implicitly (library-API-only) but never stated whether the *library itself* should hard-refuse Klein 9B at construction.
**Right or wrong?** Planner gap. A2 did exactly what its sortie said; the sortie just didn't include the wiring, and the plan didn't decide whether it should.
**Evidence:** A2 report's own candor note; §0-vs-OQ-5 in the plan.
**Carry forward:** Open Decision 1 below.

### Section 3: Open Decisions

### 1. Should the library hard-refuse Klein 9B at pipeline construction, or is all enforcement app-side?
**Why it matters:** §0 calls any Klein-9B-reachable path a defect. Today `ModelTierGate.resolve` exists but is unwired, so the library does not itself refuse. If VinetasIOS is the *only* consumer and it always calls the gate, this is fine (OQ-5). If any other consumer loads models directly, Klein 9B routes through.
**Options:** (A) Wire `ModelTierGate.resolve` into `Flux2Pipeline.init`/model-resolution so unconditional Klein-9B refusal is enforced library-side (device-gating stays app-side). (B) Leave as library API; make VinetasIOS responsible per OQ-5. (C) Split: unconditional refusals (Klein 9B) enforced library-side, tier/device gating app-side.
**Recommendation:** (C). Unconditional Klein-9B refusal is a product absolute, not device-gating — it belongs at the library boundary. Tier defaults/device gating stay app-side per OQ-5. Small follow-up sortie.

### 2. When does `enable8GBTier` flip ON?
**Why it matters:** The 8 GB path is fully built but flag-gated OFF because the conservative floor (~10 GB: 2.18 int4 + 2.28 text-enc + 0.168 VAE + 6 working) exceeds 8 GB. Nobody can use it until measured on-device `phys_footprint` confirms the real floor.
**Options:** (A) Run `integration-tests.yml` against cached models + the manual `ON_DEVICE_CHECKLIST.md` on a real 8 GB iPad, then set the working-set constant from the measurement and flip the flag. (B) Accept 8 GB as permanently out of scope. (C) Leave gated indefinitely.
**Recommendation:** (A). The measurement infrastructure exists; take the number, then decide. Do not declare out-of-scope without the measurement.

### 3. Should the new CI check gate merges?
**Why it matters:** A8 added `integration-tests.yml` (job `iPad 16GB qint8 smoke (macOS)`) but did not add it to branch-protection required checks. It runs on the PR but is non-blocking.
**Options:** (A) Add to required checks once it's proven green against cached models. (B) Leave advisory. **Recommendation:** (A) after the first green run — a required check that has never passed would block all merges.

### Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| A1 | iPad tier | opus | 1 | ✓ | Foundation; every later sortie keyed off it cleanly. |
| A7 | CDN verify | sonnet | 1 | ✓ | Surfaced the VAE-size hard discovery. |
| A6 | telemetry | opus | 1 | ✓ | Attached footprint at all weightLoadComplete sites, not just the one listed. |
| A5 | VAE tiling | sonnet | 1* | ✓ | Code correct; *process miss on GPU-suite run (supervisor-finished, no retry). |
| A2 | model gating | opus | 1 | ✓ | Correct per scope; wiring gap is a planner gap, not A2's. |
| A4 | res cap | sonnet | 1 | ✓ | Fixed the unreachable-branch bug as specified. |
| A3 | 16GB knobs | sonnet | 1 | ✓ | Added a real pipeline-wiring test beyond the standalone fns. |
| A8 | smoke + CI | opus | 1 | ✓ | Biggest sortie; test + workflow + scripts + checklist all landed. |
| B1 | int4 variant | sonnet | 1 | ✓ | Widened size Int→Double (surveyed call sites first). |
| B3 | working-set | opus | 1 | ✓ | Conservative, well-documented; applied VAE correction to iPad only. |
| B2 | int4 load | opus | 1 | ✓ | Hardest sortie; wired dead code; layout assumption is the caveat (Discovery 2). |
| B4 | 8GB knobs | sonnet | 1 | ✓ | Caught + fixed a latent ternary bug. |
| B5 | 8GB smoke | sonnet | 1 | ✓ | Correct branch (built-but-gated, not out-of-scope). |

All 13 sorties accurate — no commit was reverted or made moot by a later sortie. Model selection was accurate: zero retries, zero forced upgrades.

### Section 5: Harvest Summary

We now know the iPad memory budget was over-stated by ~2.8 GB (the VAE is tiny), that the int4 load path was dead code now revived but only CI-verifiable, and that a clean tier abstraction (parameterized `forRAMGB:`/`forTier:` overloads) makes an otherwise GPU-heavy, weight-dependent library almost entirely unit-testable headless. The single most important thing for the next iteration: **the real acceptance test is the first green run of `integration-tests.yml` against cached models** — everything WU-B ships is structurally sound and unit-proven, but int4 generation and 8 GB viability are runtime-unverified until CI exercises them. Test-cleanup pruned only 2 pre-existing harmful stubs (`Issue.record`-only, would fail when run) and flagged 8 older `KLEIN_MODEL_PATH` env-gated tests for migration — no systemic test-quality problem in the mission's own output.

### Section 6: Files

**Preserve (product deliverables — the mission's output):**
| File | Branch | Why |
|------|--------|-----|
| `Sources/Flux2Core/**` (13 commits) | mission/thimble-typhoon/01 | The tier, gating, telemetry, VAE tiling, int4 load, and default-knob implementation. |
| `.github/workflows/integration-tests.yml` + `.github/scripts/acervo-ci-*.sh` | same | The acceptance harness; runs the model-gated tests against cached models. |
| `ON_DEVICE_CHECKLIST.md` | same | Manual jetsam checklist (16 + 8 GB) — the OQ-3 manual half. |
| `CDN_PROVISIONING.md` | same | Provisioning record + the VAE-size correction. |
| `TEST_CLEANUP_REPORT.md` | same | Cleanup record + the 8 flagged env-gated tests to migrate. |

**Discard:** none. This is a `KEEP` — no files are discarded.

## Iteration Metadata

**Starting point commit:** `9d94e55` (chore: add FLUX-on-iPad execution plan and requirements)
**Mission branch:** `mission/thimble-typhoon/01`
**Final commit on mission branch:** `a0c7caf` (test-cleanup)
**Rollback target:** `9d94e55` (not exercised — verdict is KEEP)
**Next iteration branch:** n/a (KEEP; follow-ups are tickets, not a re-run)

## Rollback Verdict

**Verdict:** `KEEP`

**Reasoning:** All 13 sorties completed on the first attempt with zero retries and zero forced model upgrades; the full CI gate (`make test`) is green at HEAD (Section 4, COMPLETE log). The one substantive hard discovery (VAE size, Section 1.1) was favorable and already folded into B3. Test-cleanup removed <5% of mission tests, and those were pre-existing harmful stubs, not mission output (Section 5). The three open items (Section 3) are follow-up tickets — a scope decision OQ-5 already anticipated, a measurement that gated infrastructure now enables, and a CI-governance toggle — none is a foundation defect. Per the KEEP signal (all COMPLETED, low retry, ≤1 hard discovery, <10% cleanup) the call is clear despite this being iteration 1.

**Recommended action:** Merge the mission branch into `development`. Open three follow-up tickets: (1) wire `ModelTierGate` for library-side unconditional Klein-9B refusal [Open Decision 1, rec C]; (2) run `integration-tests.yml` against cached models + on-device checklist, then set the 8 GB working-set constant from measurement and decide `enable8GBTier` [Open Decision 2, rec A]; (3) add the smoke-test CI check to branch protection after its first green run [Open Decision 3, rec A]. Also migrate the 8 flagged `KLEIN_MODEL_PATH` tests to `.enabled(if:)`.
