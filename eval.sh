CUDA_VISIBLE_DEVICES=0 /data/home/xts/miniconda3/envs/prefixquant/bin/python eval.py \
--quant_model_path ./pre_quantized_models/Llama-2-13B-w8a16 \
--output_dir ./log/eval-Llama-2-13B-w8a16 \
--eval_batch_size 64 \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
