#!/bin/bash

# ==============================================================================
# TPP Multi-Dataset Multi-Model Experiment Runner
# ==============================================================================
#
# 🎯 Purpose: Run all TPP models on multiple datasets with multi-GPU parallel support
#
# 📝 HOW TO CONFIGURE:
#    Just modify the configuration variables below in this script!
#
# 📁 Output File: results/results.csv (append mode - keeps history)
# 📄 Log Files: results/log/output_*.log
#
# ==============================================================================

# ==============================================================================
# ⚙️  CONFIGURATION - EDIT THIS SECTION
# ==============================================================================

# ----- Dataset Selection -----
# Choose which datasets to run (space-separated)
# Available: customer_journey, retweet, taxi, taobao, stackoverflow, amazon
DATASETS="retweet"

# ----- Model/Algorithm Selection -----
# Choose which models to run (space-separated)
# Available: RMTPP_train, NHP_train, THP_train, SAHP_train, FullyNN_train,
#            IntensityFree_train(IFTPP), ODETPP_train, AttNHP_train, S2P2_train
MODELS="RMTPP_train NHP_train THP_train SAHP_train FullyNN_train IntensityFree_train ODETPP_train AttNHP_train S2P2_train"

# ----- GPU Configuration -----
# Which GPUs to use (comma-separated, e.g., "0,1" or "0,1,2,3")
GPUS="0,1"

# ----- File Paths -----
CONFIG_FILE="run/experiment_config.yaml"
OUTPUT_DIR="results"
OUTPUT_FILE="$OUTPUT_DIR/results.csv"
LOG_DIR="results/log"

# ==============================================================================
# 📖 DATASET INFORMATION
# ==============================================================================
#
# retweet       : Social media retweet data (3 event types)
# taxi          : Taxi trajectory data (10 event types)
# taobao        : E-commerce taobao data (20 event types)
# stackoverflow : StackOverflow post data (22 event types)
# amazon        : Amazon review data (16 event types)
# customer_journey    : Your custom e-commerce data (3 event types)
#
# ==============================================================================
# 🤖 MODEL INFORMATION
# ==============================================================================
#
# RMTPP_train       : Recurrent Marked Temporal Point Process (KDD'16)
# NHP_train         : Neural Hawkes Process (NeurIPS'17)
# THP_train         : Transformer Hawkes Process (NeurIPS'19)
# SAHP_train        : Self-Attentive Hawkes Process (ICML'20)
# IntensityFree_train: Intensity-Free TPP (ICLR'20)
# ODETPP_train      : ODE-based TPP (ICLR'21)
# AttNHP_train      : Attention-based NHP (ICLR'22)
# S2P2_train        : Scalable Sequential Point Process (NeurIPS'25)
#
# ==============================================================================

# Use uv to run Python commands
PYTHON_CMD="uv run python"

echo "============================================================"
echo "TPP Multi-Dataset Multi-Model Experiment Runner"
echo "============================================================"
echo "Starting time: $(date)"
echo ""

# Convert to arrays
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
GPU_COUNT=${#GPU_ARRAY[@]}
IFS=' ' read -ra DATASETS_ARRAY <<< "$DATASETS"
IFS=' ' read -ra MODELS_ARRAY <<< "$MODELS"

# Create directories
mkdir -p $OUTPUT_DIR $LOG_DIR

echo "Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Log dir: $LOG_DIR"
echo "  GPUs: ${GPU_ARRAY[@]}"
echo "  Datasets: ${DATASETS_ARRAY[@]}"
echo "  Models: ${MODELS_ARRAY[@]}"
echo "  Total: $((${#DATASETS_ARRAY[@]} * ${#MODELS_ARRAY[@]})) experiments"
echo ""

# Check config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create CSV header (append mode)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "dataset,model,timestamp,test_loglike,test_rmse,test_acc,test_mae,test_type_error_rate" > $OUTPUT_FILE
    echo "Created new results file"
else
    echo "Appending to existing results file"
fi

# Run experiment function
run_experiment() {
    local dataset=$1
    local model=$2
    local gpu_id=$3

    $PYTHON_CMD run/train_model.py \
        --config "$CONFIG_FILE" \
        --model "$model" \
        --dataset "$dataset" \
        --gpu "$gpu_id" \
        --output "$OUTPUT_FILE" \
        > "$LOG_DIR/output_${dataset}_${model}_gpu${gpu_id}.log"

    if [ $? -eq 0 ]; then
        echo "✓ $model on $dataset (GPU $gpu_id)"
    else
        echo "✗ $model on $dataset (GPU $gpu_id) - see $LOG_DIR/output_${dataset}_${model}_gpu${gpu_id}.log"
    fi
}

# Run all experiments
echo ""
echo "Running experiments..."
total=$((${#DATASETS_ARRAY[@]} * ${#MODELS_ARRAY[@]}))
current=0
exp_idx=0

for dataset in "${DATASETS_ARRAY[@]}"; do
    for model in "${MODELS_ARRAY[@]}"; do
        ((current++))
        gpu_id=${GPU_ARRAY[$((exp_idx % GPU_COUNT))]}
        ((exp_idx++))

        echo "[$current/$total] $model on $dataset (GPU $gpu_id)"

        # Run in parallel on different GPUs
        run_experiment "$dataset" "$model" "$gpu_id" &
    done
done

echo ""
echo "Waiting for all experiments to complete..."
wait

# Summary
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""

# Display results
if [ -f "$OUTPUT_FILE" ]; then
    python3 run/show_results.py --results "$OUTPUT_FILE" --log-dir "$LOG_DIR"
fi

echo ""
