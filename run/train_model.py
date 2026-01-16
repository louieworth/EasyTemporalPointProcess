#!/usr/bin/env python3
"""
Train and evaluate a single TPP model on a dataset.
Called by run.sh
"""
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Train a TPP model')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    args = parser.parse_args()

    try:
        # Load config
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)

        # Create temp config
        config_data = {
            'pipeline_config_id': full_config['pipeline_config_id'],
            'data': full_config['data'],
            args.model: full_config[args.model]
        }

        # Update dataset and GPU
        config_data[args.model]['base_config']['dataset_id'] = args.dataset
        config_data[args.model]['trainer_config']['gpu'] = args.gpu

        # Save temp config to /tmp directory
        import tempfile
        temp_config_path = f"/tmp/temp_{args.dataset}_{args.model}_gpu{args.gpu}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load and train
        config = Config.build_from_yaml_file(temp_config_path, experiment_id=args.model)
        print(f"Configuration loaded for {args.model} on {args.dataset} (GPU {args.gpu})")

        model_runner = Runner.build_from_config(config)
        print("Model runner built successfully")

        model_runner.run()
        print("Training completed successfully")

        # Evaluate
        try:
            eval_config = Config.build_from_yaml_file(temp_config_path, experiment_id=args.model)
            eval_config.base_config.stage = 'eval'
            eval_config.base_config.pretrained_model_dir = model_runner.get_model_dir()

            eval_runner = Runner.build_from_config(eval_config)
            test_loader = eval_runner._data_loader.test_loader()
            eval_metrics = eval_runner._evaluate_model(test_loader)

            # Extract metrics
            ll = eval_metrics.get('loglike', 'nan')
            rmse = eval_metrics.get('rmse', 'nan')
            acc = eval_metrics.get('acc', 'nan')
            mae = eval_metrics.get('mae', 'nan')
            type_error_rate = eval_metrics.get('type_error_rate', 'nan')

            # Format metrics
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

            print(f"Results: LL={ll}, RMSE={rmse}, ACC={acc}, MAE={mae}, TypeErr={type_error_rate}")

            # Write to CSV
            with open(args.output, 'a') as f:
                f.write(f"{args.dataset},{args.model},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
                       f"{ll},{rmse},{acc},{mae},{type_error_rate}\n")

        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            # Only write if we got some metrics
            # Skip writing if evaluation failed

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temp config
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()


if __name__ == "__main__":
    main()
