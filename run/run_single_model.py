#!/usr/bin/env python3
"""
Script to run a single model from config file
Usage: python run_single_model.py --config my_dataset_config.yaml --model RMTPP_train --gpu 0
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Run a single TPP model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model name in config')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    args = parser.parse_args()

    # Load full config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)

    # Check if model exists
    if args.model not in full_config:
        print(f"Error: Model {args.model} not found in config!")
        sys.exit(1)

    # Create a config dict with only the specified model
    # This avoids duplicate key issues
    config_data = {
        'pipeline_config_id': full_config['pipeline_config_id'],
        'data': full_config['data'],
        args.model: full_config[args.model]
    }

    # Update GPU setting
    config_data[args.model]['trainer_config']['gpu'] = args.gpu

    # Save temporary config
    temp_config_path = f"temp_{args.model}_gpu{args.gpu}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)

    try:
        # Load config
        config = Config.build_from_yaml_file(temp_config_path, experiment_id=args.model)
        print(f"Configuration loaded for {args.model} on GPU {args.gpu}")

        # Build model
        model_runner = Runner.build_from_config(config)
        print("Model runner built successfully")

        # Run training
        print(f"Starting training at {datetime.now()}")
        model_runner.run()
        print("Training completed successfully")

        # Evaluation
        try:
            # Create eval config
            eval_config = Config.build_from_yaml_file(temp_config_path, experiment_id=args.model)
            eval_config.base_config.stage = 'eval'
            eval_config.base_config.pretrained_model_dir = model_runner.get_model_dir()

            eval_runner = Runner.build_from_config(eval_config)

            # Load test data and evaluate
            test_loader = eval_runner._data_loader.test_loader()
            eval_metrics = eval_runner._evaluate_model(test_loader)

            # Extract metrics
            ll = eval_metrics.get('loglike', 'nan')
            rmse = eval_metrics.get('rmse', 'nan')
            acc = eval_metrics.get('acc', 'nan')
            mae = eval_metrics.get('mae', 'nan')
            type_error_rate = eval_metrics.get('type_error_rate', 'nan')

            # Calculate type_error_rate if it's nan and acc is available
            if isinstance(type_error_rate, float) and type_error_rate != type_error_rate:  # check if nan
                if isinstance(acc, float) and acc == acc:  # check if acc is not nan
                    type_error_rate = 1 - acc
                    print(f"  - Type Error Rate (calculated): {type_error_rate:.4f}")
                else:
                    type_error_rate = 'nan'

            print(f"Evaluation results:")
            print(f"  - Log-Likelihood: {ll:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - Type Error Rate: {type_error_rate:.4f}")

            # Format metrics to 4 decimal places for CSV
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

            # Save results to CSV if output path provided
            if args.output:
                import csv
                with open(args.output, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        args.model,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Completed',
                        ll, rmse, acc, mae, type_error_rate,
                        '', args.gpu
                    ])

        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            if args.output:
                import csv
                with open(args.output, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        args.model,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'TrainedOnly',
                        'nan', 'nan', 'nan', 'nan', 'nan',
                        f"Evaluation failed: {str(e)[:100]}",
                        args.gpu
                    ])

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

        if args.output:
            import csv
            with open(args.output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    args.model,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Failed',
                    'nan', 'nan', 'nan', 'nan', 'nan',
                    str(e)[:100],
                    args.gpu
                ])
        sys.exit(1)

    finally:
        # Clean up temporary config
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()

    print(f"Finished {args.model} on GPU {args.gpu} at {datetime.now()}")


if __name__ == "__main__":
    main()