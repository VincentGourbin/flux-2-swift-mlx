# Circus Tart dev-environment API (handoff)

**Status:** v1 interface frozen 2026-06-21 (see "v1 contract"). **Profiles
implemented 2026-06-22** — file-authored, read-only in Circus, picked at launch.
The `circus` CLI verbs (`ensure-ready` / `put` / `exec` / `wait` / `get`) are
**not built yet**; they are the next phase. Until then the testable loop is:
author a profile → launch Circus → pick it → start the VM → Circus applies RAM +
`--dir` mounts and verifies them.
**Audience:** the Circus (`utility-be-circus`) agent, plus any genAI app repo that
smoke-tests inside the Tart guest (image, audio, DAW plugins, standalone apps).
**Origin:** drafted from `flux-2-swift-mix` VM smoke work; generalized because the
same plumbing helps every in-VM app.

---

## The shift this proposes

Circus today is **monitor-first**: it polls status, reports readiness, and
deliberately refuses to *trigger app work* (the ComfyUI section "must not queue
image-generation work… check `/queue` and refuse if busy"; manual refresh
buttons were removed in favour of polling).

It already does some **environment control** — Start All / Stop All, start/stop
the ComfyUI bridge, `tart run` / `tart stop`, `tart set --memory`, manage
virtio shares. What it does *not* do is run an app's own test entrypoint.

This API makes the role explicit: **a control plane that also monitors.** The
line it crosses is "observe apps, control only the environment" →
"control the environment **and** invoke apps' bounded test entrypoints." See
**Guardrails** for the discipline that keeps that safe.

Yes, this is real new building. Most of it is *exposing primitives Circus
already has* over a transport, not new VM plumbing (see
**Already exists vs new**).

---

## Runtime assumption

All genAI apps under development run **inside the Tart guest** — Flux2App /
Flux2CLI, DAWs (Logic, Reaper, …), AU/VST plugins, standalone audio apps. There
is no host-vs-VM split to model: one runtime, many apps. Circus orchestrates the
guest; each app repo owns what "success" means.

---

## The five primitives

| Primitive | Purpose | Bounds |
| --- | --- | --- |
| `ensure-ready` | Start VM if stopped → wait guest macOS + SSH → verify every mount in the active profile | bounded timeout, structured result |
| `restart` | Graceful shutdown → relaunch with current saved prefs (shares are launch-only `--dir`; RAM via `tart set --memory`) | only when launch-time config changed |
| `exec` | Run one shell command in the guest with an env map | caller-supplied timeout; no retry loop |
| `put` / `get` | Copy host ↔ guest (build products, fixtures, golden files) | single transfer |
| `wait` | Poll a guest path (exists / non-empty), process name, or guest localhost port | bounded timeout |

`ensure-ready` returns structured state so callers never guess:

```json
{
  "ready": true,
  "vmStatus": "running",
  "sshHost": "tart-virtual-mac",
  "mounts": [
    { "name": "flux2-model-cache", "guestPath": "/Volumes/My Shared Files/flux2-model-cache", "ok": true }
  ]
}
```

On failure, a clear reason (frozen enum in the contract below) so the caller
surfaces it instead of looping.

---

## v1 contract (frozen 2026-06-21)

This section is **authoritative**. Everything else in this doc is rationale; if
prose disagrees with the contract, the contract wins. Circus and every consumer
repo build to exactly this shape. Within v1, changes are **additive only** — any
breaking change bumps to v2.

### Stable surface = the `circus` CLI

The contract is a shell-callable `circus` command (a thin CLI Circus ships that
talks to the running app over its local transport). **The wire transport
beneath it — localhost HTTP, unix socket, whatever — is Circus's choice and is
*not* part of the contract.** Consumers depend only on the CLI, which is also
what lets a mock `circus` shim stand in before the real server exists.

- `circus --api-version` → prints `1`. Callers may assert on it.
- Global flags: `--profile <name>`, `--timeout <seconds>` (integer seconds), `--json`.
- Exit codes (all verbs except `exec`): `0` success / condition met; `1` failure
  (reason on stderr, and in `error` under `--json`); `124` Circus-side timeout.
- Human-readable logs go to stderr; machine output to stdout (JSON only with `--json`).
- Bracketed numbers below are **advisory defaults**; the frozen part is each flag
  and its seconds unit, not the default value.

### Verbs

| Verb | Signature | Exit |
| --- | --- | --- |
| ensure-ready | `circus ensure-ready --profile <name> [--timeout 180] [--json]` | `0` ready; `1` not ready |
| restart | `circus restart --profile <name> [--timeout 240] [--json]` | `0` ready after restart; `1` else |
| exec | `circus exec [--env K=V]… [--timeout <s>] -- <cmd>…` | guest command's exit code; `124` if Circus timeout fired |
| put | `circus put --host <path> --guest <abs-guest-path>` | `0` / `1` |
| get | `circus get --guest <abs-guest-path> --host <path>` | `0` / `1` |
| wait | `circus wait (--guest <path> [--min-bytes <n>] \| --process <name> \| --port <n>) [--timeout 60]` | `0` met; `124` timeout |

Frozen behaviour:

- **`exec`** runs one command in the active profile's guest. `--env` repeats and
  is injected into the guest process environment; everything after `--` is the
  command, run via `sh -lc`; stdout/stderr stream through. Circus never retries.
- `exec` is fire-and-return. To background a guest process, end the command with
  `&` and use `wait` for readiness — Circus does not track guest process
  lifecycle beyond `wait --process`.
- **`put`/`get`** are single transfers; `--guest` is always an absolute guest path.
- **`wait`** polls exactly one of `--guest` / `--process` / `--port`.

### `ensure-ready` / `restart` JSON (`--json`)

```json
{
  "ready": false,
  "apiVersion": 1,
  "vmStatus": "running",
  "sshHost": "tart-virtual-mac",
  "error": "mount_missing:flux2-model-cache",
  "mounts": [
    { "name": "flux2-model-cache", "guestPath": "/Volumes/My Shared Files/flux2-model-cache", "ok": false }
  ]
}
```

- `ready` (bool) · `apiVersion` (int `1`) · `vmStatus` (`running|stopped|starting|error`)
  · `sshHost` (string) · `mounts` (array of `{name, guestPath, ok}`).
- `error` is present only when `ready` is `false`, and is one of the reason
  strings below.
- `guestPath` is always `/Volumes/My Shared Files/<name>`.

### Reason strings (frozen enum)

`vm_off` · `vm_start_failed` · `ssh_timeout` · `guest_not_operating` ·
`mount_missing:<name>` · `profile_unknown:<name>`

New reasons may be **added** within v1; callers must treat an unrecognised
reason as a generic failure rather than crashing.

### Profile schema (frozen fields)

```text
name:   string                 # the --profile value
ram:    32 | 64                # GB; applied via `tart set --memory`
mounts:                        # each → tart run --dir=<name>:<hostPath>:<mode>
  - name:     string           # also the guest dir under /Volumes/My Shared Files/
    hostPath: string           # ~ allowed
    mode:     ro | rw
```

Profiles are **file-authored JSON**, one file per profile at
`~/Documents/Circus Configurations/<name>.json`, read-only in Circus and **picked
by the human at launch** (editing requires quit → relaunch → re-pick). Presence
of a mount entry = active; there is **no `enabled` field**. A consumer repo
hard-codes only the profile **name** and the mount **names** it reads — never
host paths. Extra JSON keys are ignored; `ram` snaps to `32` or `64`.

---

## Profiles

Profiles name **which mounts and how much RAM**, not which app. App identity
stays in the app repo's smoke script. Each profile is a JSON file at
`~/Documents/Circus Configurations/<name>.json`, picked at launch (see the
contract for the exact schema). Conceptual shape:

```text
profile: tart-av-dev
  ram: 64                      # tart set --memory before start
  mounts:                      # tart run --dir=<name>:<hostPath>:<ro|rw>
    flux2-model-cache  -> ~/Library/Caches/models     (ro)
    audio-fixtures     -> ~/Audio/TestFixtures         (ro)
    plugin-builds      -> ~/GitHub/<plugin>/.build     (rw)
    sample-library     -> ~/Audio/Samples              (ro)
```

Guest path convention is fixed by Tart: `/Volumes/My Shared Files/<name>`.
Flux smoke needs `flux2-model-cache`; a reverb plugin smoke needs
`plugin-builds` + `audio-fixtures`; both call the same `ensure-ready`.

---

## Transport (greenfield — build this)

> Per the **v1 contract**, the stable surface is the `circus` CLI; the wire
> transport described here sits *behind* it and may change without breaking
> callers.

There is **no control API today.** A `control.sock` file sits in the Circus repo
root but is untracked and has no server behind it — do not assume it is wired.
The closest existing pattern is the ComfyUI Photoshop bridge (a local HTTP
listener on a fixed port).

Requirements:

- **Local-only** — Circus runs on the dev Mac; bind to localhost or a unix
  socket. A control plane that can `exec` arbitrary guest commands must not be
  remotely reachable.
- Callable from a shell (`curl` / small client) so any repo's `bin/*.sh` can use it.
- Idempotent, bounded timeouts, structured errors, **no built-in retry loops**
  (callers stay in charge; matches the VM "stop, don't thrash" rule).

Either a localhost HTTP server or a real unix socket works; pick the smaller one.

---

## What stays in each app repo

| Concern | Owner |
| --- | --- |
| Success criteria (PNG bytes, WAV SNR/diff, MIDI events, plugin load, UI marker) | app repo |
| Env vars / CLI flags / fixtures / project files | app repo |
| Model weights, sample libraries, presets (may *probe* a mount Circus verified) | app repo |
| DAW-specific validation (`auval`, `pluginval`, render scripts, ReaScript/AppleScript) | app repo |
| "exit 2 = skipped" vs hard fail | app repo |

Circus never learns AU vs VST vs MLX, never picks generation params, never
downloads weights.

---

## Already exists vs new (lean on what's there)

| Need | Circus already has | New work |
| --- | --- | --- |
| Start/stop VM, wait IP/SSH | `CircusController+TartVM.swift` (`waitForVMIP`, `waitForSSH`, …) | wrap into `ensure-ready` |
| Run guest command | `runner.run(tartPath, ["exec", vmName, "sh", "-lc", …])` (prefers SSH, falls back to `tart exec`) | expose as `exec` |
| Verify mounts | `verifyComfyUIMounts` pattern | generalize to profile mounts by name |
| RAM 32/64 GB | `TartVMRAMAllocation`, `CircusController+TartVMRAM.swift` (`tart set --memory`) | read into profile |
| Shares | Shared Directories grid → `tart run --dir=…` | read into profile |
| Orchestration precedent | `startComfyUITartVMIfNeeded` → `waitForComfyUIGuest` → `verifyComfyUIMounts` | same shape, app-agnostic |

The genuinely new infrastructure is small: the transport/server, a profiles
config, thin `exec` / `put` / `get` / `wait` verbs. Not a rewrite.

---

## Guardrails (keep the monitor-era discipline)

Crossing into "control" does **not** drop the safety posture:

- **Preserve "don't trigger production work."** The ComfyUI guardrail stays — a
  smoke render is a *deliberate, bounded, agent-invoked test*, distinct from
  background production queueing. `exec` is for test entrypoints, not for
  auto-running real workloads.
- **Bounded, not reconciling.** No background reconcilers, mutation observers,
  or retry loops behind these verbs (per `simplest-design-first`). One command,
  one bounded result.
- **Caller-driven.** Circus reports and executes on request; it doesn't decide
  to start app work on its own.
- **Local-only + structured errors**, as above.

---

## Examples

### Flux image smoke

```bash
circus ensure-ready --profile tart-av-dev
circus put --host .build/.../Flux2App --guest /tmp/flux2-smoke/Flux2App
circus exec --env F2SM_PROJECT=/tmp/.../project.json \
            --env F2SM_SMOKE_MARKER=/tmp/flux2-smoke-ready -- \
            "cd /tmp/flux2-smoke && ./Flux2App &"
circus wait --guest /tmp/flux2-smoke-ready --timeout 30
# app repo: read marker (ok/error), screencapture, scp back
```

### DAW plugin smoke (same four steps)

```bash
circus ensure-ready --profile tart-av-dev
circus put --host .build/MyPlugin.vst3 --guest /tmp/MyPlugin.vst3
circus put --host Tests/Fixtures/dry.wav --guest /tmp/input.wav
circus exec --env SMOKE_MARKER=/tmp/smoke-ok -- \
            "run-plugin-smoke.sh MyPlugin.vst3 input.wav /tmp/wet.wav"
circus wait --guest /tmp/wet.wav --min-bytes 1000 --timeout 120
circus get --guest /tmp/wet.wav --host /tmp/smoke-wet.wav
# app repo: compare to golden WAV
```

Identical shape — **ensure → deploy → exec → wait/get** — different success criteria.

---

## Non-goals

- App-specific weight catalogs, model names, DAW project formats in Circus.
- Generation/queue orchestration (stays in each app; ComfyUI queue stays ComfyUI).
- Test assertions (WAV diff, PNG compare) — app repos own these.
- Remote access. Automatic retries. Background reconciliation.

---

*This doc is app-agnostic; it lives in `flux-2-swift-mix` because that's the
first consumer. It can be lifted into the Circus repo or a shared location
without change.*
