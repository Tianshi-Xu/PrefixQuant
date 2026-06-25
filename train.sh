### PrefixQuant
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 /data/home/xts/miniconda3/envs/prefixquant/bin/python main.py \
--model_path /opt/models/Llama-2-13b-hf \
--model_name Llama-2-13B \
--output_dir ./log/Llama-2-13B-w8a8-popwk1-actk2-lorar16 \
--wbits 8 \
--w_popcount_k 1 \
--use_lora_residual \
--lora_residual_rank 16 \
--input_bits 8 \
--input_popcount_k 2 \
--input_mode static \
--v_bits 16 \
--k_bits 16 \
--s_bits 16 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,lambada,openbookqa \
--max_memory 32GiB \
--ppl_seqlen 1024 \
--eval_batch_size 8 \
--save_quant_dir ./pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2-lorar16
