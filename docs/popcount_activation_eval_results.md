# Popcount Weight/Activation Eval Results

Date: 2026-06-21

Eval tasks:

```text
piqa,arc_easy,arc_challenge,hellaswag,winogrande,lambada,openbookqa
```

Common settings:

- Weight: 8-bit symmetric, `w_popcount_k=1`
- Activation constrained runs: 8-bit symmetric, `input_popcount_k=2` or `input_popcount_k=3`
- K/V/S cache: not quantized, `k_bits=16`, `v_bits=16`, `s_bits=16`
- `ppl_seqlen=1024`, `eval_batch_size=8`
- `lambada` expands to aggregate `lambada`, `lambada_openai`, and `lambada_standard` rows in this lm-eval version.

## Summary

| Model | Config | WikiText2 PPL | Five-task Avg Acc | Five-task Avg Acc with norm |
| --- | --- | ---: | ---: | ---: |
| Llama-2-13B | W8A16, weight k=1 | 6.0165 | 65.62 | 69.56 |
| Llama-2-13B | W8A16, weight k=2 | 5.48 | 66.97 | 71.52 |
| Llama-2-13B | W8A8, weight k=1, activation k=3 | 6.02 | 65.47 | 69.74 |
| Llama-2-13B | W8A8, weight k=1, activation k=2 | 6.09 | 65.12 | 69.42 |
| Llama-2-13B | W8A8, weight k=1, activation k=2, residual LoRA r=16 | 6.05 | 65.20 | 69.35 |
| Llama-2-7B | W8A16, weight k=1 | 6.98 | 62.29 | 65.92 |
| Llama-2-7B | W8A16, weight k=2 | 6.16 | 64.51 | 68.57 |
| Llama-2-7B | W8A8, weight k=1, activation k=3 | 7.00 | 62.19 | 65.75 |
| Llama-2-7B | W8A8, weight k=1, activation k=2 | 7.11 | 61.60 | 65.29 |
| Llama-2-7B | W8A8, weight k=1, activation k=2, residual LoRA r=16 | 7.09 | 61.94 | 65.67 |
| Llama-2-7B | W8A8, weight k=1, activation k=2, residual LoRA r=16, ep20 LoRA-only fine-tune | 6.96 | 62.37 | 65.69 |

Note: averages are computed over the original five-task set: `winogrande`, `hellaswag`, `arc_challenge`, `arc_easy`, and `piqa`. `lambada` and `openbookqa` are reported as individual task results below.

## Quantized Model Paths
| Model       | Config                                               | Path                                                         |
| -------------| ------------------------------------------------------| --------------------------------------------------------------|
| Llama-2-13B | W8A16, weight k=1                                    | `pre_quantized_models/Llama-2-13B-w8a16-popk1`               |
| Llama-2-13B | W8A16, weight k=2                                    | `pre_quantized_models/Llama-2-13B-w8a16-popwk2`              |
| Llama-2-13B | W8A8, weight k=1, activation k=3                     | `pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk3`         |
| Llama-2-13B | W8A8, weight k=1, activation k=2                     | `pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2`         |
| Llama-2-13B | W8A8, weight k=1, activation k=2, residual LoRA r=16 | `pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2-lorar16` |
| Llama-2-7B  | W8A16, weight k=1                                    | `pre_quantized_models/Llama-2-7B-w8a16-popwk1`               |
| Llama-2-7B  | W8A16, weight k=2                                    | `pre_quantized_models/Llama-2-7B-w8a16-popwk2`               |
| Llama-2-7B  | W8A8, weight k=1, activation k=3                     | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk3`          |
| Llama-2-7B  | W8A8, weight k=1, activation k=2                     | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2`          |
| Llama-2-7B  | W8A8, weight k=1, activation k=2, residual LoRA r=16 | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2-lorar16`  |
| Llama-2-7B  | W8A8, weight k=1, activation k=2, residual LoRA r=16, ep20 LoRA-only fine-tune | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2-lorar16-ep20-fp32train` |

## Weight Bit-One Stats

Count mode: `popcount(abs(q))` for reconstructed signed int8 weight codes. `Avg ones / weight` is `one_bits / int_values`; `Bit-one ratio` is divided by 8 storage bits. These are weight-only stats, so activation k only changes the run configuration, not the weight constraint.

| Model       | Config                           | Avg ones / weight | Bit-one ratio |
| -------------| ----------------------------------| ------------------:| --------------:|
| Llama-2-13B | W8A16, weight k=1                | 0.985019          | 12.3127%      |
| Llama-2-13B | W8A16, weight k=2                | 1.693066          | 21.1633%      |
| Llama-2-13B | W8A8, weight k=1, activation k=3 | 0.984723          | 12.3090%      |
| Llama-2-13B | W8A8, weight k=1, activation k=2 | 0.984723          | 12.3090%      |
| Llama-2-7B  | W8A16, weight k=1                | 0.984938          | 12.3117%      |
| Llama-2-7B  | W8A16, weight k=2                | 1.694658          | 21.1832%      |
| Llama-2-7B  | W8A8, weight k=1, activation k=3 | 0.984938          | 12.3117%      |
| Llama-2-7B  | W8A8, weight k=1, activation k=2 | 0.984939          | 12.3117%      |

## Llama-2-13B W8A16 Weight k=1

| Task             | Metric     | Value  |
| ------------------| ------------| -------:|
| openbookqa       | acc        | 33.80  |
| openbookqa       | acc_norm   | 44.20  |
| lambada          | acc        | 71.25  |
| lambada          | perplexity | 3.6435 |
| lambada_openai   | acc        | 74.77  |
| lambada_openai   | perplexity | 3.2034 |
| lambada_standard | acc        | 67.73  |
| lambada_standard | perplexity | 4.0836 |
| winogrande       | acc        | 72.69  |
| hellaswag        | acc        | 57.14  |
| hellaswag        | acc_norm   | 75.33  |
| arc_challenge    | acc        | 43.52  |
| arc_challenge    | acc_norm   | 45.56  |
| arc_easy         | acc        | 76.56  |
| arc_easy         | acc_norm   | 74.54  |
| piqa             | acc        | 78.18  |
| piqa             | acc_norm   | 79.65  |


## Llama-2-13B W8A16 Weight k=2

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 35.20 |
| openbookqa | acc_norm | 46.40 |
| lambada | acc | 73.86 |
| lambada | perplexity | 3.3281 |
| lambada_openai | acc | 77.08 |
| lambada_openai | perplexity | 2.9938 |
| lambada_standard | acc | 70.64 |
| lambada_standard | perplexity | 3.6624 |
| winogrande | acc | 72.38 |
| hellaswag | acc | 59.49 |
| hellaswag | acc_norm | 78.06 |
| arc_challenge | acc | 45.82 |
| arc_challenge | acc_norm | 49.15 |
| arc_easy | acc | 78.66 |
| arc_easy | acc_norm | 77.48 |
| piqa | acc | 78.51 |
| piqa | acc_norm | 80.52 |


## Llama-2-13B W8A8 Weight k=1 Activation k=3

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 33.60 |
| openbookqa | acc_norm | 45.20 |
| lambada | acc | 71.29 |
| lambada | perplexity | 3.6647 |
| lambada_openai | acc | 75.06 |
| lambada_openai | perplexity | 3.2052 |
| lambada_standard | acc | 67.51 |
| lambada_standard | perplexity | 4.1243 |
| winogrande | acc | 72.06 |
| hellaswag | acc | 57.25 |
| hellaswag | acc_norm | 75.63 |
| arc_challenge | acc | 42.58 |
| arc_challenge | acc_norm | 46.93 |
| arc_easy | acc | 77.10 |
| arc_easy | acc_norm | 75.04 |
| piqa | acc | 78.35 |
| piqa | acc_norm | 79.05 |

## Llama-2-13B W8A8 Weight k=1 Activation k=2

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 33.20 |
| openbookqa | acc_norm | 44.00 |
| lambada | acc | 71.26 |
| lambada | perplexity | 3.7108 |
| lambada_openai | acc | 74.81 |
| lambada_openai | perplexity | 3.2296 |
| lambada_standard | acc | 67.71 |
| lambada_standard | perplexity | 4.1920 |
| winogrande | acc | 72.06 |
| hellaswag | acc | 57.13 |
| hellaswag | acc_norm | 75.38 |
| arc_challenge | acc | 42.75 |
| arc_challenge | acc_norm | 46.16 |
| arc_easy | acc | 76.52 |
| arc_easy | acc_norm | 74.62 |
| piqa | acc | 77.15 |
| piqa | acc_norm | 78.89 |


## Llama-2-13B W8A8 Weight k=1 Activation k=2 Residual LoRA r=16

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 30.80 |
| openbookqa | acc_norm | 44.80 |
| lambada | acc | 72.51 |
| lambada | perplexity | 3.5910 |
| lambada_openai | acc | 76.42 |
| lambada_openai | perplexity | 3.1101 |
| lambada_standard | acc | 68.60 |
| lambada_standard | perplexity | 4.0719 |
| winogrande | acc | 70.96 |
| hellaswag | acc | 57.26 |
| hellaswag | acc_norm | 75.69 |
| arc_challenge | acc | 42.92 |
| arc_challenge | acc_norm | 45.90 |
| arc_easy | acc | 76.89 |
| arc_easy | acc_norm | 75.21 |
| piqa | acc | 77.97 |
| piqa | acc_norm | 79.00 |


## Llama-2-7B W8A16 Weight k=1

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 31.20 |
| openbookqa | acc_norm | 42.20 |
| lambada | acc | 66.75 |
| lambada | perplexity | 4.5452 |
| lambada_openai | acc | 69.30 |
| lambada_openai | perplexity | 4.1486 |
| lambada_standard | acc | 64.20 |
| lambada_standard | perplexity | 4.9418 |
| winogrande | acc | 68.59 |
| hellaswag | acc | 54.21 |
| hellaswag | acc_norm | 72.63 |
| arc_challenge | acc | 39.33 |
| arc_challenge | acc_norm | 41.89 |
| arc_easy | acc | 72.90 |
| arc_easy | acc_norm | 69.28 |
| piqa | acc | 76.44 |
| piqa | acc_norm | 77.20 |


## Llama-2-7B W8A16 Weight k=2

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 33.80 |
| openbookqa | acc_norm | 44.20 |
| lambada | acc | 70.79 |
| lambada | perplexity | 3.7976 |
| lambada_openai | acc | 72.77 |
| lambada_openai | perplexity | 3.5049 |
| lambada_standard | acc | 68.81 |
| lambada_standard | perplexity | 4.0903 |
| winogrande | acc | 70.09 |
| hellaswag | acc | 57.26 |
| hellaswag | acc_norm | 76.37 |
| arc_challenge | acc | 43.17 |
| arc_challenge | acc_norm | 44.88 |
| arc_easy | acc | 74.83 |
| arc_easy | acc_norm | 73.19 |
| piqa | acc | 77.20 |
| piqa | acc_norm | 78.35 |


## Llama-2-7B W8A8 Weight k=1 Activation k=3

| Task | Metric | Value |
| --- | --- | ---: |
| openbookqa | acc | 30.80 |
| openbookqa | acc_norm | 42.00 |
| lambada | acc | 66.66 |
| lambada | perplexity | 4.5706 |
| lambada_openai | acc | 69.38 |
| lambada_openai | perplexity | 4.1683 |
| lambada_standard | acc | 63.94 |
| lambada_standard | perplexity | 4.9729 |
| winogrande | acc | 68.51 |
| hellaswag | acc | 54.09 |
| hellaswag | acc_norm | 72.62 |
| arc_challenge | acc | 39.51 |
| arc_challenge | acc_norm | 41.38 |
| arc_easy | acc | 72.39 |
| arc_easy | acc_norm | 68.86 |
| piqa | acc | 76.44 |
| piqa | acc_norm | 77.37 |

## Llama-2-7B W8A8 Weight k=1 Activation k=2

| Task             | Metric     | Value  |
| ------------------| ------------| -------:|
| openbookqa       | acc        | 30.60  |
| openbookqa       | acc_norm   | 41.60  |
| lambada          | acc        | 65.91  |
| lambada          | perplexity | 4.7059 |
| lambada_openai   | acc        | 68.60  |
| lambada_openai   | perplexity | 4.2778 |
| lambada_standard | acc        | 63.23  |
| lambada_standard | perplexity | 5.1339 |
| winogrande       | acc        | 66.61  |
| hellaswag        | acc        | 53.69  |
| hellaswag        | acc_norm   | 72.20  |
| arc_challenge    | acc        | 39.51  |
| arc_challenge    | acc_norm   | 41.47  |
| arc_easy         | acc        | 71.84  |
| arc_easy         | acc_norm   | 68.81  |
| piqa             | acc        | 76.33  |
| piqa             | acc_norm   | 77.37  |

## Llama-2-7B W8A8 Weight k=1 Activation k=2 Residual LoRA r=16

| Task             | Metric     | Value  |
| ------------------| ------------| -------:|
| openbookqa       | acc        | 30.20  |
| openbookqa       | acc_norm   | 42.20  |
| lambada          | acc        | 67.15  |
| lambada          | perplexity | 4.5503 |
| lambada_openai   | acc        | 69.49  |
| lambada_openai   | perplexity | 4.1288 |
| lambada_standard | acc        | 64.80  |
| lambada_standard | perplexity | 4.9718 |
| winogrande       | acc        | 68.27  |
| hellaswag        | acc        | 53.60  |
| hellaswag        | acc_norm   | 72.29  |
| arc_challenge    | acc        | 38.74  |
| arc_challenge    | acc_norm   | 41.72  |
| arc_easy         | acc        | 72.64  |
| arc_easy         | acc_norm   | 69.02  |
| piqa             | acc        | 76.44  |
| piqa             | acc_norm   | 77.04  |


## Llama-2-7B W8A8 Weight k=1 Activation k=2 Residual LoRA r=16, ep20 LoRA-only fine-tune

| Task             | Metric     | Value  |
| ------------------| ------------| -------:|
| openbookqa       | acc        | 29.00  |
| openbookqa       | acc_norm   | 41.20  |
| lambada          | acc        | 66.98  |
| lambada          | perplexity | 4.2565 |
| lambada_openai   | acc        | 69.49  |
| lambada_openai   | perplexity | 3.8737 |
| lambada_standard | acc        | 64.47  |
| lambada_standard | perplexity | 4.6393 |
| winogrande       | acc        | 69.22  |
| hellaswag        | acc        | 53.56  |
| hellaswag        | acc_norm   | 71.77  |
| arc_challenge    | acc        | 39.33  |
| arc_challenge    | acc_norm   | 41.13  |
| arc_easy         | acc        | 73.61  |
| arc_easy         | acc_norm   | 68.94  |
| piqa             | acc        | 76.12  |
| piqa             | acc_norm   | 77.42  |
