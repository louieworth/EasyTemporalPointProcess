#!/bin/bash

# Run all TPP models using the original repository scripts
# This script iterates through all models and runs them individually

echo "============================================================"
echo "TPP Experiment Runner (using original scripts)"
echo "============================================================"
echo "Starting time: $(date)"
echo ""

# Default parameters
CONFIG_FILE="my_dataset_config.yaml"
OUTPUT_DIR="results/commerce"
OUTPUT_FILE="$OUTPUT_DIR/model_test_results.csv"

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
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config FILE    Configuration file (default: my_dataset_config.yaml)"
            echo "  --output FILE    Output CSV file (default: model_test_results.csv)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration: $CONFIG_FILE"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create CSV header
echo "model,timestamp,status,test_loglike,test_rmse,test_acc,test_mae,test_type_error_rate,error_message" > $OUTPUT_FILE

# List of models to run (from the config file)
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

# Function to run a single model
run_model() {
    local model=$1
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] Running: $model"
    echo "============================================================"

    # Create a temporary config for this model
    TEMP_CONFIG="temp_${model}.yaml"

    # Extract just this model's config
    python3 << EOF
import yaml

# Read the full config
with open('$CONFIG_FILE', 'r') as f:
    full_config = yaml.safe_load(f)

# Create a new config with just this model
model_config = {
    'pipeline_config_id': full_config['pipeline_config_id'],
    'data': full_config['data'],
    '$model': full_config['$model']
}

# Write the temp config
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(model_config, f)
EOF

    # Run the model using the original example script structure
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Create and run a Python script similar to examples/train_experiment/run_retweet.py
    python3 << EOF > "$OUTPUT_DIR/output_${model}.log" 2>&1
import sys
sys.path.insert(0, '.')

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

try:
    # Load config for this model
    config = Config.build_from_yaml_file('$TEMP_CONFIG', experiment_id='$model')
    print(f"Configuration loaded for $model")

    # Build and run the model
    model_runner = Runner.build_from_config(config)
    print("Model runner built successfully")

    # Run training
    model_runner.run()
    print("Training completed successfully")

    # After training, evaluate on test set if possible
    try:
        # Create a new runner for evaluation (since data_loader might be cleaned up)
        eval_config = Config.build_from_yaml_file('$TEMP_CONFIG', experiment_id='${model}')
        eval_config.base_config.stage = 'eval'
        eval_config.base_config.pretrained_model_dir = model_runner.get_model_dir()

        eval_runner = Runner.build_from_config(eval_config)

        # Load test data
        test_loader = eval_runner._data_loader.test_loader()
        eval_metrics = eval_runner._evaluate_model(test_loader)

        # Extract metrics
        ll = eval_metrics.get('loglike', 'nan')
        rmse = eval_metrics.get('rmse', 'nan')
        acc = eval_metrics.get('acc', 'nan')
        mae = eval_metrics.get('mae', 'nan')
        type_error_rate = eval_metrics.get('type_error_rate', 'nan')

        print(f"Evaluation results: LL={ll}, RMSE={rmse}, ACC={acc}, MAE={mae}, TypeErr={type_error_rate}")

        # Write to CSV
        with open('$OUTPUT_FILE', 'a') as f:
            f.write(f"$model,$TIMESTAMP,Completed,{ll},{rmse},{acc},{mae},{type_error_rate},\n")

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        # Still write training completion
        with open('$OUTPUT_FILE', 'a') as f:
            f.write(f"$model,$TIMESTAMP,TrainedOnly,nan,nan,nan,nan,nan,Evaluation failed: {str(e)[:100]}\n")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

    # Write failure to CSV
    with open('$OUTPUT_FILE', 'a') as f:
        f.write(f"$model,$TIMESTAMP,Failed,nan,nan,nan,nan,nan,{str(e)[:100]}\n")

    sys.exit(1)
EOF

    # Check result
    if [ $? -eq 0 ]; then
        echo "✓ $model completed successfully"
    else
        echo "✗ $model failed. Check $OUTPUT_DIR/output_${model}.log for details"
    fi

    # Clean up
    rm -f $TEMP_CONFIG

    # Show progress
    echo "Progress: $CURRENT/$TOTAL models completed"
}

# Run all models
for model in "${MODELS[@]}"; do
    run_model $model
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
import numpy as np

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

    # Add average row
    avg_row = {
        'model': 'AVERAGE',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'Summary',
        'test_loglike': completed['test_loglike'].mean(),
        'test_rmse': completed['test_rmse'].mean(),
        'test_acc': completed['test_acc'].mean(),
        'test_mae': completed['test_mae'].mean(),
        'test_type_error_rate': completed['test_type_error_rate'].mean(),
        'error_message': ''
    }
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df_with_avg.to_csv('$OUTPUT_FILE', index=False)

print(f"\nDetailed results saved to: $OUTPUT_FILE")
EOF
fi

# Clean up log files
read -p "Remove log files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f $OUTPUT_DIR/output_*.log
    echo "Log files removed."
fi

echo ""
echo "============================================================"