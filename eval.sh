<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=3 python eval.py \
--quant_model ./pre_quantized_models/Llama-2-7b-hf-w4a4q4s8kv4-finetune \
=======
CUDA_VISIBLE_DEVICES=0 /data/home/xts/miniconda3/envs/prefixquant/bin/python eval.py \
--quant_model_path ./pre_quantized_models/Llama-2-13B-w8a16 \
--output_dir ./log/eval-Llama-2-13B-w8a16 \
>>>>>>> submission
--eval_batch_size 64 \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
