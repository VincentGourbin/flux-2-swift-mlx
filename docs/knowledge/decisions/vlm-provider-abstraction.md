---
type: Decision
title: "VLM enrichment runs behind a provider protocol; Qwen3.5 stays the default"
description: Why the image-aware services were abstracted behind FluxVLMProvider, why Gemma 4 lives in its own target, and the constraints each provider path must respect.
tags: [vlm, qwen35, gemma4, architecture, prompt-enrichment]
timestamp: 2026-08-24T00:00:00Z
---

Decision (2026-08-24, PR: `feature/gemma4-vlm-provider`): the framework's
image-aware services — BFL prompt rewriting for the inpainting/outpainting
chains, `describeImageForFlux`, the 0-100 scene/style scoring used by the LoRA
evaluator and by VLM-guided checkpoint selection, the training captioner — run
on **`FluxVLM.active`**, a `FluxVLMProvider`. Two providers ship: the bundled
Qwen3.5 4B (**default, unchanged**) and Gemma 4 E2B-it (opt-in).

# Why abstract at all

Before this, every call site named Qwen3.5 directly. That was a real trap, not
just duplication: Qwen3.5 was *also* the image-guided prompt rewriter of the
inpainting/outpainting chains, so any migration that replaced the captioning
loops with another VLM and dropped Qwen would have degraded prompt enrichment
**silently** — the chains fall back to the verbatim prompt with only a
`FluxDebug` warning. One seat, one protocol, and the fallback stays visible.

# Shape of the abstraction

One primitive: `generateText(images: [CGImage], prompt:, systemPrompt:,
enableThinking:, maxTokens:, temperature:)`. `images` is `0` (text-only), `1`
(analysis) or `2` (comparison) — those are the only shapes the framework uses.

Everything else — the FLUX.2 description rubric, the comparison rubric, the JSON
score parsing, the five intent-specific BFL system prompts — stays in
`FluxTextEncoders` as protocol extensions. A provider knows how to run a
forward; it never re-derives prompt engineering. Adding a third VLM is four
members.

# Why Gemma 4 lives in its own target (`FluxGemma4VLM`)

`Gemma4Swift` depends on `mlx-swift-lm` (MLXLLM/MLXLMCommon). Putting it in
`FluxTextEncoders` would have pushed that graph into `Flux2Core`,
`Flux2Chains`, the CLI and the app — for a feature most processes never use.
The provider protocol therefore lives in `FluxTextEncoders` (no new
dependency); the Gemma implementation lives in an opt-in library target that
consumers link deliberately. Version compatibility checked: gemma-4-swift-mlx
1.5.0 accepts mlx-swift `0.31.4..<0.32`, so the framework's `exact: "0.31.6"`
pin holds; resolution brings mlx-swift-lm 3.31.4.

# Why E2B 6-bit is the Gemma default

~4.2 GB, the same memory class as the 4-bit Qwen3.5 the enrichment paths have
always downloaded, and E2B is the smallest Gemma 4 family carrying a vision
tower. 4-bit (~3.6 GB) and 8-bit (~5.2 GB) and bf16 (~10 GB, what the ltx
enhancer uses for parity with the reference service) are selectable.

# Constraints the Gemma path must respect

* **One `<|image|>` marker per image.** `Gemma4Processor.multimodalChatIds`
  prepends exactly one marker; images 2..N need their own in the user prompt.
  `chatStreamMultimodal` rejects a marker/batch mismatch — which is the good
  outcome, because `maskedScatter` indexes its source modulo size and would
  otherwise scatter unrelated embeddings with no error at all.
* **Batch padding is required.** `Gemma4ImageProcessor` sizes each image to its
  own aspect-ratio box, so a 1024² reference and a 512×768 render come out with
  different shapes; they are zero-padded top-left to the batch maximum before
  `concatenated(axis: 0)`.
* **Thinking off, temperature 0.** The enrichment paths parse the answer (a
  one-line prompt, a JSON object). Reasoning eats the token budget and then has
  to be stripped; `cleanGeneratedText` keeps only what follows the last
  `<channel|>` for the cases where it is deliberately enabled.
* **A `nil` system prompt on the text-only path** would make Gemma's
  `ChatSession` inject its own French default instructions — the provider
  substitutes a neutral English one instead.
* `Gemma4Pipeline` is `@MainActor`, unlike this framework's off-main loading
  policy. The weight loading and the generation still run off the main actor
  (nonisolated `loadModelContainer`, `ModelContainer` actor), but the pipeline
  handle itself must be created and called with `await`.

# What did not change

* Default provider, default weights, and the byte-for-byte Qwen3.5 forward
  (`generateMultiImage`).
* The load-bearing contract that the chains **never auto-load** a VLM and fall
  back to the caller's prompt when none is resident.
* The existing Qwen-specific public API (`loadQwen35VLM`,
  `analyzeImageWithQwen35`, `describeImageForFlux`, `compareImagesForFlux`, …),
  so Fluxforge Studio keeps compiling untouched.
