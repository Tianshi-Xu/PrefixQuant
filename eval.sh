CUDA_VISIBLE_DEVICES=2 python eval.py \
--quant_model ./pre_quantized_models/Llama-2-7b-hf-w4a4q4s8kv4 \
--eval_batch_size 192 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande