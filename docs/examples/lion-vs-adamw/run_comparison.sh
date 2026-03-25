#!/bin/bash
# Lion vs AdamW comparison on Cat Toy dataset (Klein 4B)
# Run from project root: bash docs/examples/lion-vs-adamw/run_comparison.sh

set -e

CLI="$(find ~/Library/Developer/Xcode/DerivedData/flux-2-swift-mlx-*/Build/Products/Release/Flux2CLI -type f 2>/dev/null | head -1)"
if [ -z "$CLI" ]; then
    echo "Error: Flux2CLI not found. Build with: xcodebuild -scheme Flux2CLI -configuration Release -destination 'platform=macOS' build"
    exit 1
fi
echo "Using CLI: $CLI"

# Clean output
rm -rf output/lion-vs-adamw
mkdir -p output/lion-vs-adamw

echo ""
echo "=========================================="
echo "  Step 1: Train with AdamW (500 steps)"
echo "=========================================="
echo ""
time $CLI train-lora --config docs/examples/lion-vs-adamw/cat_toy_adamw.yaml

echo ""
echo "=========================================="
echo "  Step 2: Train with Lion (500 steps)"
echo "=========================================="
echo ""
time $CLI train-lora --config docs/examples/lion-vs-adamw/cat_toy_lion.yaml

echo ""
echo "=========================================="
echo "  Step 3: Evaluate checkpoints with VLM"
echo "=========================================="
echo ""

# Reference image for comparison
REF_IMAGE="examples/cat-toy/train/6.jpeg"

# Evaluate each checkpoint's validation images
for OPTIMIZER in adamw lion; do
    echo "--- Evaluating $OPTIMIZER checkpoints ---"
    OUTPUT_DIR="output/lion-vs-adamw/$OPTIMIZER"

    for STEP_DIR in "$OUTPUT_DIR"/checkpoint_*; do
        if [ -d "$STEP_DIR" ]; then
            STEP=$(basename "$STEP_DIR" | sed 's/checkpoint_//')
            echo ""
            echo "[$OPTIMIZER step $STEP]"

            # Find the first validation image with trigger word
            VAL_IMAGE=$(find "$STEP_DIR" -name "*statue_cat_toy*512*.png" | head -1)
            if [ -z "$VAL_IMAGE" ]; then
                VAL_IMAGE=$(find "$STEP_DIR" -name "*val_*.png" -o -name "*512*.png" | head -1)
            fi

            if [ -n "$VAL_IMAGE" ]; then
                echo "  Comparing: $VAL_IMAGE"
                $CLI test-qwen35 "Compare" \
                    --image "$REF_IMAGE" \
                    --image2 "$VAL_IMAGE" \
                    --compare \
                    --variant 4bit \
                    2>&1 | grep -E "Scene:|Style:"
            else
                echo "  No validation image found in $STEP_DIR"
            fi
        fi
    done
done

echo ""
echo "=========================================="
echo "  Done! Results in output/lion-vs-adamw/"
echo "=========================================="
