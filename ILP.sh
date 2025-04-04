# ILP
CUDA_VISIBLE_DEVICES=3 python ILP.py \
--model_path ~/models/llama2_7b  \
--model_name Llama-2-7b-hf \
--output_dir ./log/llama-2-7b-ILP-test \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--cal_size 16 \
--acc_la_path ./llama-2-7b-ILP-test_la.pth \
--log_name llama-2-7b-ILP-test

CUDA_VISIBLE_DEVICES=3 python eval.py \
--quant_model ./pre_quantized_models/llama-2-7b-w4q4a4kv4-asym \
--eval_batch_size 32 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
