#!/usr/bin/env python3
"""
Display experiment results summary
"""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Display TPP experiment results')
    parser.add_argument('--results', type=str, default='results/results.csv',
                       help='Results CSV file path')
    parser.add_argument('--log-dir', type=str, default='results/log',
                       help='Log directory path')
    args = parser.parse_args()

    results_file = Path(args.results)

    if not results_file.exists():
        print(f"❌ Results file not found: {args.results}")
        return

    # Read results
    df = pd.read_csv(args.results)

    total = len(df)
    has_results = df['test_loglike'].notna().sum()

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total Experiments: {total}")
    print(f"  ✅ With results: {has_results}")
    print(f"  ❌ Missing results: {total - has_results}")

    # Results by dataset
    if len(df) > 0:
        print("\n" + "-" * 80)
        print("Results by Dataset:")
        print("-" * 80)
        for dataset in sorted(df['dataset'].unique()):
            df_ds = df[df['dataset'] == dataset]
            ds_completed = df_ds[df_ds['test_loglike'].notna()]
            success_rate = len(ds_completed) / len(df_ds) * 100 if len(df_ds) > 0 else 0
            print(f"  {dataset:<15} | Total: {len(df_ds):2} | Completed: {len(ds_completed):2} | "
                  f"Success Rate: {success_rate:.1f}%")

    # Results by model
    if len(df) > 0:
        print("\n" + "-" * 80)
        print("Results by Model:")
        print("-" * 80)
        for model in sorted(df['model'].unique()):
            df_model = df[df['model'] == model]
            model_completed = df_model[df_model['test_loglike'].notna()]
            if len(model_completed) > 0:
                avg_ll = model_completed['test_loglike'].astype(float).mean()
                print(f"  {model:<20} | Completed: {len(model_completed):2}/{len(df_model):2} | "
                      f"Avg LL: {avg_ll:.4f}")
            else:
                print(f"  {model:<20} | Completed: {len(model_completed):2}/{len(df_model):2}")

    # Best results by dataset
    completed = df[df['test_loglike'].notna()]
    if len(completed) > 0 and 'dataset' in df.columns:
        print("\n" + "-" * 80)
        print("🏆 Best Model by Dataset (by Log-Likelihood):")
        print("-" * 80)
        for dataset in sorted(completed['dataset'].unique()):
            df_ds = completed[completed['dataset'] == dataset]
            best_idx = df_ds['test_loglike'].astype(float).idxmax()
            best = df_ds.loc[best_idx]
            print(f"  {dataset:<15} | {best['model']:<20} | "
                  f"LL={best['test_loglike']} | RMSE={best['test_rmse']}")

    # File locations
    print("\n" + "=" * 80)
    print(f"📁 Results file: {args.results}")
    print(f"📄 Logs directory: {args.log_dir}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
