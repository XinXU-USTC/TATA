set -ex


MODEL_NAME_OR_PATH=Qwen2.5-3B # path to the model
NO=$1

START=$((NO  *  1000))
END=($(( (NO + 1)  *  1000)))


for ((i=$START;i<$END;i+=200)); do
    echo "Processing range: $i to $((i + 200))"

    # Run the first python command
    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
    python -um infer.select \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --prompt_type "pot" \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start $i \
    --end $((i + 200)) \
    --split dm-hard-gsm8k_filter_by_dm \
    --test_split gsm8k-all_k-means100

    # Run the second python command
    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
    python -um infer.select \
    --model_name_or_path ${MODEL_NAME_OR_PATH}  \
    --prompt_type "cot" \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start $i \
    --end $((i + 200)) \
    --split dm-hard-gsm8k_filter_by_dm \
    --test_split gsm8k-all_k-means100

    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
    python -um infer.select \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --prompt_type "pot" \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start $i \
    --end $((i + 200)) \
    --split dm-hard-math_filter_by_dm \
    --test_split math_k-means100

    # Run the second python command
    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
    python -um infer.select \
    --model_name_or_path ${MODEL_NAME_OR_PATH}  \
    --prompt_type "cot" \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start $i \
    --end $((i + 200)) \
    --split dm-hard-math_filter_by_dm \
    --test_split math_k-means100
     
done