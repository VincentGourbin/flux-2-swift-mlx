#!/bin/bash
# Train LoRA on all models (Klein 4B, Klein 9B, Dev)
# Each uses 250 steps quick test configuration

set -e  # Exit on error

cd /Users/vincent/Developpements/flux-2-swift-mlx
FLUX=".build-xcode/Build/Products/Release/Flux2CLI"

# HuggingFace token for gated models
HF_TOKEN="${HF_TOKEN:-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}"

echo "=============================================="
echo "  LORA TRAINING - ALL MODELS (Quick Test)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Function to download base model if needed
download_base_model() {
    local model=$1
    local name=$2

    echo "Checking $name base model..."
    if ! $FLUX download --model "$model" --base --hf-token "$HF_TOKEN" 2>&1; then
        echo ""
        echo "⚠️  Failed to download $name base model."
        echo "   Please visit the HuggingFace page and accept the license:"
        echo "   https://huggingface.co/black-forest-labs/FLUX.2-klein-base-${model#klein-}"
        echo ""
        return 1
    fi
    return 0
}

# Function to run training with timing
run_training() {
    local config=$1
    local name=$2

    echo "=============================================="
    echo "  Training: $name"
    echo "  Config: $config"
    echo "  Started: $(date)"
    echo "=============================================="

    local start_time=$(date +%s)

    $FLUX train-lora --config "$config"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo ""
    echo "✅ $name completed in ${minutes}m ${seconds}s"
    echo ""
}

# 1. Klein 4B (base model already downloaded)
echo ""
echo "=== KLEIN 4B ==="
run_training "examples/tarot-style/tarot_quick_test.yaml" "Klein 4B"

# 2. Klein 9B
echo ""
echo "=== KLEIN 9B ==="
if download_base_model "klein-9b" "Klein 9B"; then
    run_training "examples/tarot-style/tarot_klein9b.yaml" "Klein 9B"
else
    echo "⏭️  Skipping Klein 9B training (base model not available)"
fi

# 3. Dev (32B)
echo ""
echo "=== DEV (32B) ==="
if download_base_model "dev" "Dev"; then
    run_training "examples/tarot-style/tarot_dev.yaml" "Dev (32B)"
else
    echo "⏭️  Skipping Dev training (base model not available)"
fi

echo "=============================================="
echo "  ALL TRAININGS COMPLETED"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Klein 4B: ./output/tarot-quick-test/"
echo "  - Klein 9B: ./output/tarot-klein9b/"
echo "  - Dev:      ./output/tarot-dev/"
echo ""
echo "Check learning curves and validation images in each output folder."
