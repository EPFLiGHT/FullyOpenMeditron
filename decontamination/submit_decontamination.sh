#!/bin/bash

#decontamination/submit_decontamination.sh
# ==============================================================================
# Original Source: https://github.com/swiss-ai/posttraining-data/tree/main/04-decontamination
# Project: Swiss AI - Post-Training Data Pipeline
#
# This wrapper script is adapted from the Swiss AI decontamination workflow.
# ==============================================================================

# Script to submit decontamination SLURM jobs
# Usage: ./submit_decontamination.sh <input_dataset_path> <output_dataset_path>

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repository root (parent of 04-decontamination)
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dataset_path> <output_dataset_path>"
    echo "Example: $0 /path/to/input/dataset /path/to/output/dataset"
    exit 1
fi

# Get input arguments
INPUT_PATH="$1"
OUTPUT_PATH="$2"

# Validate input path exists
if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Input dataset path does not exist: $INPUT_PATH"
    exit 1
fi

# Extract dataset name from path for job naming
DATASET_NAME=$(basename "$INPUT_PATH")

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# Generate unique job script name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_SCRIPT="slurm_logs/decontaminate_${DATASET_NAME}_${TIMESTAMP}.slurm"

# Create the SLURM job script
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J decontam_${DATASET_NAME}
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=slurm_logs/decontam_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --error=slurm_logs/decontam_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

# Set environment variables
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

# Set cache directory on capstor
export DECONTAMINATION_CACHE_DIR="/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache"

# Create cache directory if it doesn't exist
mkdir -p \${DECONTAMINATION_CACHE_DIR}

# Change to project directory
cd ${REPO_ROOT}
source venv/bin/activate

# Run decontamination
python 04-decontamination/decontamination.py \\
      "${INPUT_PATH}" \\
      --output "${OUTPUT_PATH}" \\
      --decontamination_prompts "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts" \\
      --tokenizer_name "alehc/swissai-tokenizer" \\
      --report_path "${OUTPUT_PATH}/contamination_reports" \\
      --cache_dir "\${DECONTAMINATION_CACHE_DIR}" \\
      --ngram_length 8 \\
      --diff_threshold 0.5 \\
      --num_proc 16 \\
      --show_contaminated

# Check exit status
if [ \$? -eq 0 ]; then
    echo "Decontamination completed successfully."
else
    echo "Decontamination failed with exit code \$?"
    exit 1
fi
EOF

# Submit the job
echo "Submitting decontamination job for dataset: $DATASET_NAME"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Job script: $JOB_SCRIPT"

sbatch "$JOB_SCRIPT"

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo "Job submitted successfully."
    echo ""
    echo "To monitor the job output, run:"
    echo "  tail -f slurm_logs/decontam_${DATASET_NAME}_${TIMESTAMP}.out"
else
    echo "Error: Failed to submit job"
    exit 1
fi