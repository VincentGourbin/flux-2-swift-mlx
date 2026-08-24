# Directory Update Log

## 2026-08-24

* **Added**: [VLM enrichment behind a provider protocol](/docs/knowledge/decisions/vlm-provider-abstraction.md)
  — the image-aware services (prompt rewriting, FLUX.2 description, LoRA
  scene/style scoring, training captions) now run on `FluxVLM.active`; Gemma 4
  E2B joins the bundled Qwen3.5 as a selectable provider, with the
  marker/batch/thinking constraints of the Gemma path recorded.

## 2026-07-15

* **Creation**: Bootstrapped the knowledge bundle from the July 2026
  performance campaign (PRs #112-#116). Initial concepts: baselines, loading
  costs, quantization verdicts, the compile-step decision, seven verified
  pitfalls, the [generation slowness investigation](/docs/knowledge/investigations/generation-slowness-2026-07.md)
  and the [GPU contention playbook](/docs/knowledge/playbooks/gpu-contention-diagnosis.md).
