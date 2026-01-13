#!/bin/bash

# Run TPP models in parallel using different GPUs
# This script creates separate shell scripts for each model and runs them in parallel

echo "============================================================"
echo "TPP Parallel Experiment Runner"
echo "============================================================"
echo "Starting time: $(date)"
echo ""

# Default parameters
CONFIG_FILE="my_dataset_config.yaml"
OUTPUT_DIR="results/commerce"
OUTPUT_FILE="$OUTPUT_DIR/model_test_results_parallel.csv"

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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config FILE    Configuration file (default: my_dataset_config.yaml)"
            echo "  --output FILE    Output CSV file (default: model_test_results_parallel.csv)"
            echo "  --gpus LIST      Comma-separated list of GPU IDs (default: 0,1)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default GPU list
if [ -z "$GPUS" ]; then
    GPUS="0,1"
fi

# Convert comma-separated GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
GPU_COUNT=${#GPU_ARRAY[@]}

echo "Configuration: $CONFIG_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Available GPUs: ${GPU_ARRAY[@]}"
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

# Function to run a single model
run_model() {
    local model=$1
    local gpu_id=$2
    local temp_script="run_${model}_gpu${gpu_id}.sh"

    echo "Creating script for $model on GPU $gpu_id..."

    # Create a temporary script for this model
    cat > "$temp_script" << EOF
#!/bin/bash
echo "Starting $model on GPU $gpu_id at \$(date)"

# Create a temporary config for this model
TEMP_CONFIG="temp_${model}_gpu${gpu_id}.yaml"

# Extract just this model's config
python3 << PYEOF
import yaml

# Read the full config
with open('$CONFIG_FILE', 'r') as f:
    full_config = yaml.safe_load(f)

# Create a new config with just this model
model_config = {
    'pipeline_config_id': full_config['pipeline_config_id'],
    'data': full_config['data'],
}

# Add only this model's configuration
model_config['${model}'] = full_config['${model}']

# Update GPU setting
model_config['${model}']['trainer_config']['gpu'] = ${gpu_id}

# Write the temp config
with open('${TEMP_CONFIG}', 'w') as f:
    yaml.dump(model_config, f)
PYEOF

# Run the model
python3 << PYEOF > "$OUTPUT_DIR/output_${model}_gpu${gpu_id}.log" 2>&1
import sys
sys.path.insert(0, '.')

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

try:
    # Load config for this model
    config = Config.build_from_yaml_file('${TEMP_CONFIG}', experiment_id='${model}')
    print(f"Configuration loaded for ${model} on GPU ${gpu_id}")

    # Build and run the model
    model_runner = Runner.build_from_config(config)
    print("Model runner built successfully")

    # Run training
    model_runner.run()
    print("Training completed successfully")

    # After training, evaluate on test set
    try:
        # Create a new runner for evaluation
        eval_config = Config.build_from_yaml_file('${TEMP_CONFIG}', experiment_id='${model}')
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

        # Format metrics to 4 decimal places
        if isinstance(ll, float) and ll == ll:
            ll = f"{ll:.4f}"
        if isinstance(rmse, float) and rmse == rmse:
            rmse = f"{rmse:.4f}"
        if isinstance(acc, float) and acc == acc:
            acc = f"{acc:.4f}"
        if isinstance(mae, float) and mae == mae:
            mae = f"{mae:.4f}"
        if isinstance(type_error_rate, float) and type_error_rate == type_error_rate:
            type_error_rate = f"{type_error_rate:.4f}"

        print(f"Evaluation results: LL={ll}, RMSE={rmse}, ACC={acc}, MAE={mae}, TypeErr={type_error_rate}")

        # Write to CSV
        with open('${OUTPUT_FILE}', 'a') as f:
            f.write(f"${model},$(date '+%Y-%m-%d %H:%M:%S'),Completed,{ll},{rmse},{acc},{mae},{type_error_rate},,${gpu_id}\n")

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        with open('${OUTPUT_FILE}', 'a') as f:
            f.write(f"${model},$(date '+%Y-%m-%d %H:%M:%S'),TrainedOnly,nan,nan,nan,nan,nan,Evaluation failed: {str(e)[:100]},${gpu_id}\n")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

    # Write failure to CSV
    with open('${OUTPUT_FILE}', 'a') as f:
        f.write(f"${model},$(date '+%Y-%m-%d %H:%M:%S'),Failed,nan,nan,nan,nan,nan,{str(e)[:100]},${gpu_id}\n")

    sys.exit(1)
PYEOF

# Check result
if [ $? -eq 0 ]; then
    echo "✓ $model on GPU $gpu_id completed successfully"
else
    echo "✗ $model on GPU $gpu_id failed. Check $OUTPUT_DIR/output_${model}_gpu${gpu_id}.log for details"
fi

# Clean up
rm -f ${TEMP_CONFIG}
echo "Finished $model on GPU $gpu_id at \$(date)"
EOF

    chmod +x "$temp_script"
}

# Create scripts for all models
echo "Creating run scripts..."
model_idx=0
for model in "${MODELS[@]}"; do
    gpu_id=${GPU_ARRAY[$((model_idx % GPU_COUNT))]}
    run_model $model $gpu_id
    ((model_idx++))
done

# Run models in parallel
echo ""
echo "============================================================"
echo "Running models in parallel..."
echo "============================================================"

# Run all scripts in background and wait for them
pids=()
for script in run_*_gpu*.sh; do
    echo "Starting $script..."
    ./$script &
    pids+=($!)
done

# Wait for all processes to complete
echo ""
echo "Waiting for all models to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

# Clean up temporary scripts
echo ""
echo "Cleaning up temporary scripts..."
rm -f run_*_gpu*.sh

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

# Show GPU utilization
print("\nGPU Utilization:")
gpu_usage = df_models.groupby('gpu_id').size()
for gpu_id in sorted(gpu_usage.index):
    count = gpu_usage[gpu_id]
    print(f"  GPU {gpu_id}: {count} model(s)")

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
        gpu_id = row['gpu_id']
        print(f"{idx}. {row['model']:<20} | LL={ll:.4f} | RMSE={rmse:.4f} | Acc={acc:.4f} | TypeErr={type_error_rate:.4f} | GPU={gpu_id}")

    # Best model
    best = completed.loc[completed['test_loglike'].idxmax()]
    print(f"\n🏆 Best performing model: {best['model']}")
    print(f"   - Log-Likelihood: {best['test_loglike']:.4f}")
    print(f"   - RMSE: {best['test_rmse']:.4f}")
    print(f"   - Accuracy: {best['test_acc']:.4f}")
    print(f"   - Type Error Rate: {best['test_type_error_rate']:.4f}")
    print(f"   - GPU Used: {best['gpu_id']}")

print(f"\nDetailed results saved to: $OUTPUT_FILE")
EOF
fi

echo ""
echo "============================================================"