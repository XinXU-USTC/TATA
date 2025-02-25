set -ex

MODEL_LIST=(
    # put your model path here
)

DATA_LIST=(
    "math"
    "gsm8k"
    "collegemath"
    "olympiadbench"
    "mawps"
    "svamp"
)
SPLIT="test"
PROMPT_TYPE="alpaca_zs"
#PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    echo "Using model: $MODEL_NAME_OR_PATH"

    # Loop over each dataset
    for DATA_NAME in "${DATA_LIST[@]}"; do
        echo "Processing dataset: $DATA_NAME"

        CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        python -um infer.inference \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --data_name ${DATA_NAME} \
            --split ${SPLIT} \
            --prompt_type ${PROMPT_TYPE} \
            --num_test_sample ${NUM_TEST_SAMPLE} \
            --seed 0 \
            --temperature 0 \
            --n_sampling 1 \
            --top_p 1 \
            --start 0 \
            --end -1
    done
done