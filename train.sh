CUDA_VISIBLE_DEVICES=2 python main.py \
--model_path ~/models/llama2_7b  \
--model_name Llama-2-7b-hf \
--output_dir ./log/llama-2-7b-w4q4a4kv4-asym \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--kv_group_size 128 \
--kv_mode static \
--w_asym \
--input_asym \
--kv_asym \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--mse_init \
--save_quant_dir ./pre_quantized_models/llama-2-7b-w4q4a4kv4-asym 
# --epochs 20
# --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \

# --mse_init False \
# --epochs 20

# ILP
CUDA_VISIBLE_DEVICES=1 python ILP.py \
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
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--save_quant_dir ./pre_quantized_models/llama-2-7b-ILP-test \
--epochs 20

CUDA_VISIBLE_DEVICES=3 python eval.py \
--quant_model ./pre_quantized_models/llama-2-7b-w4q4a4kv4-asym \
--eval_batch_size 32 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande


# HEQuant test
CUDA_VISIBLE_DEVICES=3 python main.py \
--model_path ~/models/llama2_7b  \
--model_name Llama-2-7b-hf \
--output_dir ./log/llama-2-7b-w4q4a4kv4-asym-test \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--kv_group_size 128 \
--kv_mode static \
--w_asym \
--input_asym \
--kv_asym \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--mse_init \
--extra_lr 0 \
--weight_lr 5e-6 \
--quant_lr 5e-5 \
--negative_penalty_weight 1e-8 \
--save_quant_dir ./pre_quantized_models/llama-2-7b-w4q4a4kv4-asym-test \
--un_bound \
--epochs 20