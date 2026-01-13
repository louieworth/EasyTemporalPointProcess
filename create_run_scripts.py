import os
import stat
import yaml
from string import Template

# All models and datasets from the original script
MODELS = [
    "RMTPP_train",
    "NHP_train",
    "THP_train",
    "SAHP_train",
    "IntensityFree_train",
    "ODETPP_train",
    "AttNHP_train",
    "S2P2_train",
]

DATASETS = [
    "taxi",
    "retweet",
    "taobao",
    "stackoverflow",
    "amazon",
    "e_commerce",
]

RUN_DIR = "run"

# Template for the individual shell scripts using string.Template to avoid curly brace conflicts
SH_TEMPLATE = Template("""#!/bin/bash
# Runs $model on $dataset dataset.
# This script is generated automatically.
set -e

# Navigate to project root to ensure paths are correct
PROJECT_ROOT=$$(git rev-parse --show-toplevel)
cd "$$PROJECT_ROOT" || exit 1

# --- Configuration ---
# Experiment-specific variables
CONFIG_FILE="$config_path"
MODEL_ID="$model"
DATASET_ID="$dataset"
TIMESTAMP=$$(date '+%Y-%m-%d %H:%M:%S')

# Output directories and files
RESULTS_DIR="results/$$MODEL_ID"
LOGS_DIR="$$RESULTS_DIR/logs"
MAIN_CSV="$$RESULTS_DIR/results.csv"
LOG_FILE="$$LOGS_DIR/$${DATASET_ID}.log"

# --- Script Body ---
echo "============================================================================"
echo "Running $$MODEL_ID on $$DATASET_ID"
echo "Config: $config_path"
echo "Results CSV: $$MAIN_CSV"
echo "Log file: $$LOG_FILE"
echo "============================================================================"

# Ensure the output directories exist
mkdir -p "$$LOGS_DIR"

# Create the algorithm-specific CSV file with a header if it doesn't exist
if [ ! -f "$$MAIN_CSV" ]; then
    echo "Creating results file: $$MAIN_CSV"
    echo "dataset,model,timestamp,status,test_loglike,test_rmse,test_acc,test_mae,error_message" > "$$MAIN_CSV"
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
    error_msg = str(error_msg).replace('"', '""').replace('\\n', ' ')

    with open(csv_file, 'a') as f:
        f.write(f'\\"{dataset_id}\\",\\"{model_id}\\",\\"{timestamp}\\",\\"{status}\\",'
                f'{ll},{rmse},{acc},{mae},\\"{error_msg[:200]}\\"\\n')

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

" "$config_path" "$model" "$dataset" "$$MAIN_CSV" "$$TIMESTAMP" > "$$LOG_FILE" 2>&1

# --- Final Status ---
# Check the exit code of the Python script
if [ $$? -eq 0 ]; then
    echo "✓ $$MODEL_ID on $$DATASET_ID completed successfully."
else
    echo "✗ $$MODEL_ID on $$DATASET_ID failed. Check log for details: $$LOG_FILE"
fi

echo "============================================================================"
echo ""
""")

def generate_scripts():
    """
    Generates individual run scripts and configuration files for every combination
    of model and dataset.
    """
    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR)
        print(f"Created directory: {RUN_DIR}")

    for model in MODELS:
        model_dir = os.path.join(RUN_DIR, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for dataset in DATASETS:
            # Determine which base config file to use
            if dataset == "e_commerce":
                base_config_path = "my_dataset_config.yaml"
            else:
                base_config_path = "examples/configs/experiment_config.yaml"

            if not os.path.exists(base_config_path):
                print(f"Warning: Base config '{base_config_path}' not found. Skipping {model} on {dataset}.")
                continue

            try:
                with open(base_config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error: Could not read or parse YAML file {base_config_path}: {e}")
                continue

            if model not in full_config:
                print(f"Warning: Model '{model}' not found in '{base_config_path}'. Skipping this model.")
                break  # Move to the next model

            if dataset != 'e_commerce' and dataset not in full_config.get('data', {}):
                print(f"Warning: Dataset '{dataset}' not found in '{base_config_path}'. Skipping.")
                continue

            # --- Create the experiment-specific configuration ---
            model_config = full_config[model].copy()

            # For datasets from the original repo, update the dataset_id
            if dataset != 'e_commerce':
                if 'base_config' not in model_config:
                    model_config['base_config'] = {}
                model_config['base_config']['dataset_id'] = dataset

            # Override max_epoch for all models to 500 as requested.
            if 'trainer_config' in model_config:
                model_config['trainer_config']['max_epoch'] = 500
            else:
                print(f"Warning: Model '{model}' has no 'trainer_config'. Cannot set max_epoch.")


            experiment_config = {
                'pipeline_config_id': full_config.get('pipeline_config_id'),
                'data': full_config.get('data'),
                model: model_config
            }

            # --- Write the config file for the experiment ---
            exp_config_filename = f"{dataset}_config.yaml"
            exp_config_path = os.path.join(model_dir, exp_config_filename)
            with open(exp_config_path, 'w') as f:
                yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)

            # --- Generate and write the corresponding shell script ---
            sh_filename = f"{dataset}.sh"
            sh_path = os.path.join(model_dir, sh_filename)
            relative_config_path = os.path.join(RUN_DIR, model, exp_config_filename)

            sh_content = SH_TEMPLATE.substitute(
                model=model,
                dataset=dataset,
                config_path=relative_config_path
            )

            with open(sh_path, 'w') as f:
                f.write(sh_content)

            # Make the script executable
            os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)

            print(f"Generated: {sh_path} and {exp_config_path}")


if __name__ == "__main__":
    print("Starting script generation...")
    generate_scripts()
    print("Script generation complete. You can find the scripts in the 'run/' directory.")
