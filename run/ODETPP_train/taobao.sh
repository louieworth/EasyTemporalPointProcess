#!/bin/bash
# Runs ODETPP_train on taobao dataset.
# This script is generated automatically.
set -e

# Navigate to project root to ensure paths are correct
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT" || exit 1

# --- Configuration ---
# Experiment-specific variables
CONFIG_FILE="run/ODETPP_train/taobao_config.yaml"
MODEL_ID="ODETPP_train"
DATASET_ID="taobao"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Output directories and files
RESULTS_DIR="results/$MODEL_ID"
LOGS_DIR="$RESULTS_DIR/logs"
MAIN_CSV="$RESULTS_DIR/results.csv"
LOG_FILE="$LOGS_DIR/${DATASET_ID}.log"

# --- Script Body ---
echo "============================================================================"
echo "Running $MODEL_ID on $DATASET_ID"
echo "Config: run/ODETPP_train/taobao_config.yaml"
echo "Results CSV: $MAIN_CSV"
echo "Log file: $LOG_FILE"
echo "============================================================================"

# Ensure the output directories exist
mkdir -p "$LOGS_DIR"

# Create the algorithm-specific CSV file with a header if it doesn't exist
if [ ! -f "$MAIN_CSV" ]; then
    echo "Creating results file: $MAIN_CSV"
    echo "dataset,model,timestamp,status,test_loglike,test_rmse,test_acc,test_mae,error_message" > "$MAIN_CSV"
fi

# Run the experiment using an inline Python script.
# All output (stdout and stderr) is redirected to the log file.
# We pass shell variables as command-line arguments to Python for safety.
python3 -c "
import sys
import traceback
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# Variables passed from shell script as command-line arguments
config_path = sys.argv[1]
model_id = sys.argv[2]
dataset_id = sys.argv[3]
csv_file = sys.argv[4]
timestamp = sys.argv[5]

def write_to_csv(status, metrics={}, error_msg=''):
    '''Helper function to write a result row to the CSV file.'''
    ll = metrics.get('loglike', 'nan')
    rmse = metrics.get('rmse', 'nan')
    acc = metrics.get('acc', 'nan')
    mae = metrics.get('mae', 'nan')

    # Sanitize error message for CSV
    error_msg = str(error_msg).replace('"', '""').replace('\n', ' ')

    with open(csv_file, 'a') as f:
        f.write(f'\"{dataset_id}\",\"{model_id}\",\"{timestamp}\",\"{status}\",'
                f'{ll},{rmse},{acc},{mae},\"{error_msg[:200]}\"\n')

try:
    # 1. Load config and build the runner
    print(f'Loading config from {config_path}...')
    config = Config.build_from_yaml_file(config_path, experiment_id=model_id)
    model_runner = Runner.build_from_config(config)
    print('Model runner built successfully.')

    # 2. Train the model (which saves the best model based on validation log-likelihood)
    print(f'Starting training for {model_id} on {dataset_id}...')
    model_runner.run()
    print('Training completed.')

    # 3. Evaluate the best model on the test set
    # First, load the best model that was saved during the training process.
    print('Loading best model for final evaluation...')
    try:
        saved_model_dir = config.base_config.specs['saved_model_dir']
        model_runner._load_model(saved_model_dir)
        print(f'Successfully loaded best model from: {saved_model_dir}')
    except Exception as load_e:
        print(f'Could not load best model, will evaluate model from last epoch instead. Reason: {load_e}')

    # Now, evaluate using the loaded best model (or the last model if loading failed).
    print('Starting evaluation on the test set...')
    test_loader = model_runner.data_loader.test_loader()
    eval_metrics = model_runner._evaluate_model(test_loader)
    print(f'Evaluation metrics: {eval_metrics}')

    # 4. Write success result to CSV
    write_to_csv('Completed', eval_metrics)
    print('Results successfully recorded.')

except Exception as e:
    # On failure, log the exception and write a 'Failed' status to the CSV
    print(f'An error occurred during the experiment for {model_id} on {dataset_id}.')
    traceback.print_exc()
    write_to_csv('Failed', error_msg=e)
    sys.exit(1) # Exit with error code

" "run/ODETPP_train/taobao_config.yaml" "ODETPP_train" "taobao" "$MAIN_CSV" "$TIMESTAMP" > "$LOG_FILE" 2>&1

# --- Final Status ---
# Check the exit code of the Python script
if [ $? -eq 0 ]; then
    echo "✓ $MODEL_ID on $DATASET_ID completed successfully."
else
    echo "✗ $MODEL_ID on $DATASET_ID failed. Check log for details: $LOG_FILE"
fi

echo "============================================================================"
echo ""
