# Llama-2-13B W8A16 Popcount-k=1 Eval Results

## Setup

- Model: `pre_quantized_models/Llama-2-13B-w8a16-popk1`
- Baseline model for comparison: `pre_quantized_models/Llama-2-13B-w8a16`
- Weight constraint: `popcount(abs(q)) <= 1` for the signed int8 quantized weight code `q`
- Effective k=1 codebook: `{0, +/-1, +/-2, +/-4, +/-8, +/-16, +/-32, +/-64}`
- Eval batch size used for the successful runs: `16`
- PPL was not rerun for the extra eval tasks.

## Reproduction Commands

Full k=1 eval:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 \
/data/home/xts/miniconda3/envs/prefixquant/bin/python eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-13B-w8a16-popk1 \
  --output_dir ./log/eval-Llama-2-13B-w8a16-popk1-full-b16 \
  --eval_batch_size 16 \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,lambada,openbookqa
```

Note: in this lm-eval version, `lambada` expands to `lambada_openai` and `lambada_standard`, with an aggregate `lambada` row.

## Accuracy Results

| Task | Metric | k=1 |
| --- | --- | ---: |
| winogrande | acc | 72.69 |
| hellaswag | acc | 57.14 |
| hellaswag | acc_norm | 75.33 |
| arc_challenge | acc | 43.52 |
| arc_challenge | acc_norm | 45.56 |
| arc_easy | acc | 76.56 |
| arc_easy | acc_norm | 74.54 |
| piqa | acc | 78.18 |
| piqa | acc_norm | 79.65 |
| openbookqa | acc | 33.80 |
| openbookqa | acc_norm | 44.20 |
| lambada | acc | 71.25 |
| lambada | perplexity | 3.6435 |
| lambada_openai | acc | 74.77 |
| lambada_openai | perplexity | 3.2034 |
| lambada_standard | acc | 67.73 |
| lambada_standard | perplexity | 4.0836 |

Average over the original five-task set:

| Average | k=1 |
| --- | ---: |
| Average Acc | 65.62 |
| Average Acc with norm | 69.56 |

## Related Quantization Stats

| Model | WikiText PPL | Int8 bit-one ratio | Count mode |
| --- | ---: | ---: | --- |
| W8A16 baseline | 5.43 | 49.4358% | two's-complement, old stat |
| W8A16 popcount k=1 | 6.0165 | 12.3127% | magnitude, `popcount(abs(q))` |
