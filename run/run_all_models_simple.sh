#!/bin/bash

# Simple script to run all models sequentially using run_single_model.py

echo "============================================================"
echo "TPP Experiment Runner (Simple Version)"
echo "============================================================"
echo "Starting time: $(date)"
echo ""

# Default parameters
CONFIG_FILE="my_dataset_config.yaml"
OUTPUT_DIR="results/commerce"
OUTPUT_FILE="$OUTPUT_DIR/model_test_results_simple.csv"

# Create results directory
mkdir -p $OUTPUT_DIR

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config FILE    Configuration file (default: my_dataset_config.yaml)"
            echo "  --output FILE    Output CSV file (default: model_test_results_simple.csv)"
            echo "  --gpu ID         GPU ID to use for all models (default: 0)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default GPU
if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

echo "Configuration: $CONFIG_FILE"
echo "Output file: $OUTPUT_FILE"
echo "GPU ID: $GPU_ID"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create CSV header
echo "model,timestamp,status,test_loglike,test_rmse,test_acc,test_mae,test_type_error_rate,error_message,gpu_id" > $OUTPUT_FILE

# List of models to run
MODELS=(
    "RMTPP_train"
    "NHP_train"
    "THP_train"
    "SAHP_train"
    "IntensityFree_train"
    "ODETPP_train"
    "AttNHP_train"
    "S2P2_train"
)

# Counter
TOTAL=${#MODELS[@]}
CURRENT=0

# Run each model
for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] Running: $model on GPU $GPU_ID"
    echo "============================================================"

    python run_single_model.py --config "$CONFIG_FILE" --model "$model" --gpu "$GPU_ID" --output "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ $model completed successfully"
    else
        echo "✗ $model failed"
    fi
done

# Final summary
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""

# Show results
if [ -f "$OUTPUT_FILE" ]; then
    echo "Results summary:"
    python3 << EOF
import pandas as pd

# Read results
df = pd.read_csv('$OUTPUT_FILE')
df_models = df[df['model'] != 'AVERAGE']

completed = df_models[df_models['status'] == 'Completed']
trained_only = df_models[df_models['status'] == 'TrainedOnly']
failed = df_models[df_models['status'] == 'Failed']

print(f"\nTotal models: {len(df_models)}")
print(f"  - Fully completed (train+eval): {len(completed)}")
print(f"  - Trained only: {len(trained_only)}")
print(f"  - Failed: {len(failed)}")

# Show results for completed models
if len(completed) > 0:
    print("\nTest Results (sorted by Log-Likelihood):")
    print("-" * 80)
    completed_sorted = completed.sort_values('test_loglike', ascending=False, na_last=True)

    for idx, (_, row) in enumerate(completed_sorted.iterrows(), 1):
        ll = row['test_loglike']
        rmse = row['test_rmse']
        acc = row['test_acc']
        type_error_rate = row['test_type_error_rate']
        print(f"{idx}. {row['model']:<20} | LL={ll:.4f} | RMSE={rmse:.4f} | Acc={acc:.4f} | TypeErr={type_error_rate:.4f}")

    # Best model
    best = completed.loc[completed['test_loglike'].idxmax()]
    print(f"\n🏆 Best performing model: {best['model']}")
    print(f"   - Log-Likelihood: {best['test_loglike']:.4f}")
    print(f"   - RMSE: {best['test_rmse']:.4f}")
    print(f"   - Accuracy: {best['test_acc']:.4f}")
    print(f"   - Type Error Rate: {best['test_type_error_rate']:.4f}")

print(f"\nDetailed results saved to: $OUTPUT_FILE")
EOF
fi

echo ""
echo "============================================================"