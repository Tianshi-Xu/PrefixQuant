# Popcount 约束实验验收结果

NZ (Non-Zero bit count) 即 popcount_k,限制量化后整数码的二进制表示中 1 的个数不超过 k。

## 验收配置与模型可用性

| 编号 | 验收配置 | 模型 | w_popcount_k | input_popcount_k | 模型路径 | 模型状态 |
|---|---|---|---:|---:|---|---|
| 1 | Llama2-7B-W8NZ2 | Llama-2-7B | 2 | — | `pre_quantized_models/Llama-2-7B-w8a16-popwk2` | ✅ 可直接 eval |
| 2 | Llama2-7B-W8NZ1 | Llama-2-7B | 1 | — | `pre_quantized_models/Llama-2-7B-w8a16-popwk1` | ❌ 需重新量化 |
| 3 | Llama2-13B-W8NZ2 | Llama-2-13B | 2 | — | `pre_quantized_models/Llama-2-13B-w8a16-popwk2` | ✅ 可直接 eval |
| 4 | Llama2-13B-W8NZ1 | Llama-2-13B | 1 | — | `pre_quantized_models/Llama-2-13B-w8a16-popk1` | ❌ 需重新量化 |
| 5 | Llama2-7B-W8NZ1A8NZ3 | Llama-2-7B | 1 | 3 | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk3` | ❌ 需重新量化 |
| 6 | Llama2-7B-W8NZ1A8NZ2 | Llama-2-7B | 1 | 2 | `pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2` | ❌ 需重新量化 |
| 7 | Llama2-13B-W8NZ1A8NZ3 | Llama-2-13B | 1 | 3 | `pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk3` | ❌ 需重新量化 |
| 8 | Llama2-13B-W8NZ1A8NZ2 | Llama-2-13B | 1 | 2 | `pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2` | ❌ 需重新量化 |

**结论：8 个配置中仅 2 个（#1 Llama2-7B-W8NZ2、#3 Llama2-13B-W8NZ2）有现成模型可直接 eval。其余 6 个模型此前已保存但后被删除,需重新量化。**

---

## 评测任务

```
arc_challenge, arc_easy, hellaswag, lambada, openbookqa, piqa, winogrande
```

---

## 之前实验结果（供对照）

通用设置：`pre_rotate`, `down_online_had`, `qk_online_had`, `set_prefixed_tokens`, `mse_init`, `k_bits=16`, `v_bits=16`, `s_bits=16`, `eval_batch_size=8`

### 总览

| 配置 | WikiText2 PPL | ARC-C acc | ARC-C acc_norm | ARC-E acc | ARC-E acc_norm | HellaSwag acc | HellaSwag acc_norm | Lambada acc | Lambada ppl | OpenBookQA acc | OpenBookQA acc_norm | PiQA acc | PiQA acc_norm | Winogrande acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama2-7B-W8NZ2 | 6.16 | 43.17 | 44.88 | 74.83 | 73.19 | 57.26 | 76.37 | 70.79 | 3.7976 | 33.80 | 44.20 | 77.20 | 78.35 | 70.09 |
| Llama2-7B-W8NZ1 | 6.98 | 39.33 | 41.89 | 72.90 | 69.28 | 54.21 | 72.63 | 66.75 | 4.5452 | 31.20 | 42.20 | 76.44 | 77.20 | 68.59 |
| Llama2-13B-W8NZ2 | 5.48 | 45.82 | 49.15 | 78.66 | 77.48 | 59.49 | 78.06 | 73.86 | 3.3281 | 35.20 | 46.40 | 78.51 | 80.52 | 72.38 |
| Llama2-13B-W8NZ1 | 6.02 | 43.52 | 45.56 | 76.56 | 74.54 | 57.14 | 75.33 | 71.25 | 3.6435 | 33.80 | 44.20 | 78.18 | 79.65 | 72.69 |
| Llama2-7B-W8NZ1A8NZ3 | 7.00 | 39.51 | 41.38 | 72.39 | 68.86 | 54.09 | 72.62 | 66.66 | 4.5706 | 30.80 | 42.00 | 76.44 | 77.37 | 68.51 |
| Llama2-7B-W8NZ1A8NZ2 | 7.11 | 39.51 | 41.47 | 71.84 | 68.81 | 53.69 | 72.20 | 65.91 | 4.7059 | 30.60 | 41.60 | 76.33 | 77.37 | 66.61 |
| Llama2-13B-W8NZ1A8NZ3 | 6.02 | 42.58 | 46.93 | 77.10 | 75.04 | 57.25 | 75.63 | 71.29 | 3.6647 | 33.60 | 45.20 | 78.35 | 79.05 | 72.06 |
| Llama2-13B-W8NZ1A8NZ2 | 6.09 | 42.75 | 46.16 | 76.52 | 74.62 | 57.13 | 75.38 | 71.26 | 3.7108 | 33.20 | 44.00 | 77.15 | 78.89 | 72.06 |

注：Lambada 行为聚合值。`lambada_openai` 和 `lambada_standard` 的分项见下方详细表。

### 权重 Bit-One 统计

统计模式 `popcount(abs(q))`,除以 8 个存储比特得到 ratio。无约束 int8 基线 ≈ 49.44%。

| 配置 | Avg ones / weight | Bit-one ratio |
|---|---:|---:|
| Llama2-7B-W8NZ1 | 0.984938 | 12.3117% |
| Llama2-7B-W8NZ2 | 1.694658 | 21.1832% |
| Llama2-13B-W8NZ1 | 0.985019 | 12.3127% |
| Llama2-13B-W8NZ2 | 1.693066 | 21.1633% |
| Llama2-7B-W8NZ1A8NZ2 | 0.984939 | 12.3117% |
| Llama2-7B-W8NZ1A8NZ3 | 0.984938 | 12.3117% |
| Llama2-13B-W8NZ1A8NZ2 | 0.984723 | 12.3090% |
| Llama2-13B-W8NZ1A8NZ3 | 0.984723 | 12.3090% |

k=1 码本：`{0, ±1, ±2, ±4, ±8, ±16, ±32, ±64}`

---

## 逐任务详细结果

### Llama2-7B-W8NZ2

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 43.17 | 44.88 | |
| arc_easy | 74.83 | 73.19 | |
| hellaswag | 57.26 | 76.37 | |
| lambada | 70.79 | | 3.7976 |
| lambada_openai | 72.77 | | 3.5049 |
| lambada_standard | 68.81 | | 4.0903 |
| openbookqa | 33.80 | 44.20 | |
| piqa | 77.20 | 78.35 | |
| winogrande | 70.09 | | |

### Llama2-7B-W8NZ1

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 39.33 | 41.89 | |
| arc_easy | 72.90 | 69.28 | |
| hellaswag | 54.21 | 72.63 | |
| lambada | 66.75 | | 4.5452 |
| lambada_openai | 69.30 | | 4.1486 |
| lambada_standard | 64.20 | | 4.9418 |
| openbookqa | 31.20 | 42.20 | |
| piqa | 76.44 | 77.20 | |
| winogrande | 68.59 | | |

### Llama2-13B-W8NZ2

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 45.82 | 49.15 | |
| arc_easy | 78.66 | 77.48 | |
| hellaswag | 59.49 | 78.06 | |
| lambada | 73.86 | | 3.3281 |
| lambada_openai | 77.08 | | 2.9938 |
| lambada_standard | 70.64 | | 3.6624 |
| openbookqa | 35.20 | 46.40 | |
| piqa | 78.51 | 80.52 | |
| winogrande | 72.38 | | |

### Llama2-13B-W8NZ1

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 43.52 | 45.56 | |
| arc_easy | 76.56 | 74.54 | |
| hellaswag | 57.14 | 75.33 | |
| lambada | 71.25 | | 3.6435 |
| lambada_openai | 74.77 | | 3.2034 |
| lambada_standard | 67.73 | | 4.0836 |
| openbookqa | 33.80 | 44.20 | |
| piqa | 78.18 | 79.65 | |
| winogrande | 72.69 | | |

### Llama2-7B-W8NZ1A8NZ3

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 39.51 | 41.38 | |
| arc_easy | 72.39 | 68.86 | |
| hellaswag | 54.09 | 72.62 | |
| lambada | 66.66 | | 4.5706 |
| lambada_openai | 69.38 | | 4.1683 |
| lambada_standard | 63.94 | | 4.9729 |
| openbookqa | 30.80 | 42.00 | |
| piqa | 76.44 | 77.37 | |
| winogrande | 68.51 | | |

### Llama2-7B-W8NZ1A8NZ2

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 39.51 | 41.47 | |
| arc_easy | 71.84 | 68.81 | |
| hellaswag | 53.69 | 72.20 | |
| lambada | 65.91 | | 4.7059 |
| lambada_openai | 68.60 | | 4.2778 |
| lambada_standard | 63.23 | | 5.1339 |
| openbookqa | 30.60 | 41.60 | |
| piqa | 76.33 | 77.37 | |
| winogrande | 66.61 | | |

### Llama2-13B-W8NZ1A8NZ3

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 42.58 | 46.93 | |
| arc_easy | 77.10 | 75.04 | |
| hellaswag | 57.25 | 75.63 | |
| lambada | 71.29 | | 3.6647 |
| lambada_openai | 75.06 | | 3.2052 |
| lambada_standard | 67.51 | | 4.1243 |
| openbookqa | 33.60 | 45.20 | |
| piqa | 78.35 | 79.05 | |
| winogrande | 72.06 | | |

### Llama2-13B-W8NZ1A8NZ2

| Task | acc | acc_norm | perplexity |
|---|---:|---:|---:|
| arc_challenge | 42.75 | 46.16 | |
| arc_easy | 76.52 | 74.62 | |
| hellaswag | 57.13 | 75.38 | |
| lambada | 71.26 | | 3.7108 |
| lambada_openai | 74.81 | | 3.2296 |
| lambada_standard | 67.71 | | 4.1920 |
| openbookqa | 33.20 | 44.00 | |
| piqa | 77.15 | 78.89 | |
| winogrande | 72.06 | | |

---

## 复现命令

### 通用环境变量

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/data/home/xts/miniconda3/envs/prefixquant/bin/python
TASKS=arc_challenge,arc_easy,hellaswag,lambada,openbookqa,piqa,winogrande
```

### 第一步：重新量化 6 个缺失模型

以下 6 个命令可并行跑（每个占一张 GPU,通过 `CUDA_VISIBLE_DEVICES` 指定）。

```bash
# #2 Llama2-7B-W8NZ1
CUDA_VISIBLE_DEVICES=0 $PYTHON main.py \
  --model_path /opt/models/Llama-2-7b-hf --model_name Llama-2-7B \
  --output_dir ./log/Llama-2-7B-w8a16-popwk1 \
  --save_quant_dir ./pre_quantized_models/Llama-2-7B-w8a16-popwk1 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 16 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8

# #4 Llama2-13B-W8NZ1
CUDA_VISIBLE_DEVICES=1 $PYTHON main.py \
  --model_path /opt/models/Llama-2-13b-hf --model_name Llama-2-13B \
  --output_dir ./log/Llama-2-13B-w8a16-popk1 \
  --save_quant_dir ./pre_quantized_models/Llama-2-13B-w8a16-popk1 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 16 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8

# #5 Llama2-7B-W8NZ1A8NZ3
CUDA_VISIBLE_DEVICES=0 $PYTHON main.py \
  --model_path /opt/models/Llama-2-7b-hf --model_name Llama-2-7B \
  --output_dir ./log/Llama-2-7B-w8a8-popwk1-actk3 \
  --save_quant_dir ./pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk3 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 8 --input_popcount_k 3 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8

# #6 Llama2-7B-W8NZ1A8NZ2
CUDA_VISIBLE_DEVICES=0 $PYTHON main.py \
  --model_path /opt/models/Llama-2-7b-hf --model_name Llama-2-7B \
  --output_dir ./log/Llama-2-7B-w8a8-popwk1-actk2 \
  --save_quant_dir ./pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 8 --input_popcount_k 2 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8

# #7 Llama2-13B-W8NZ1A8NZ3
CUDA_VISIBLE_DEVICES=1 $PYTHON main.py \
  --model_path /opt/models/Llama-2-13b-hf --model_name Llama-2-13B \
  --output_dir ./log/Llama-2-13B-w8a8-popwk1-actk3 \
  --save_quant_dir ./pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk3 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 8 --input_popcount_k 3 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8

# #8 Llama2-13B-W8NZ1A8NZ2
CUDA_VISIBLE_DEVICES=1 $PYTHON main.py \
  --model_path /opt/models/Llama-2-13b-hf --model_name Llama-2-13B \
  --output_dir ./log/Llama-2-13B-w8a8-popwk1-actk2 \
  --save_quant_dir ./pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2 \
  --wbits 8 --w_popcount_k 1 \
  --input_bits 8 --input_popcount_k 2 --input_mode static \
  --k_bits 16 --v_bits 16 --s_bits 16 --kv_group_size 128 --kv_mode static \
  --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --max_memory 32GiB --ppl_seqlen 1024 --eval_batch_size 8
```

注：7B 模型约需 ~13GB 磁盘 + ~32GB 显存；13B 模型约需 ~25GB 磁盘 + ~32GB 显存。量化阶段不需要 `--eval_tasks`,量化完单独跑 eval 即可。

### 第二步：对 8 个模型跑 eval

```bash
# #1 Llama2-7B-W8NZ2 (已有模型)
CUDA_VISIBLE_DEVICES=0 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-7B-w8a16-popwk2 \
  --output_dir ./log/eval-Llama2-7B-W8NZ2 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #2 Llama2-7B-W8NZ1
CUDA_VISIBLE_DEVICES=0 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-7B-w8a16-popwk1 \
  --output_dir ./log/eval-Llama2-7B-W8NZ1 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #3 Llama2-13B-W8NZ2 (已有模型)
CUDA_VISIBLE_DEVICES=1 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-13B-w8a16-popwk2 \
  --output_dir ./log/eval-Llama2-13B-W8NZ2 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #4 Llama2-13B-W8NZ1
CUDA_VISIBLE_DEVICES=1 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-13B-w8a16-popk1 \
  --output_dir ./log/eval-Llama2-13B-W8NZ1 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #5 Llama2-7B-W8NZ1A8NZ3
CUDA_VISIBLE_DEVICES=0 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk3 \
  --output_dir ./log/eval-Llama2-7B-W8NZ1A8NZ3 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #6 Llama2-7B-W8NZ1A8NZ2
CUDA_VISIBLE_DEVICES=0 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-7B-w8a8-popwk1-actk2 \
  --output_dir ./log/eval-Llama2-7B-W8NZ1A8NZ2 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #7 Llama2-13B-W8NZ1A8NZ3
CUDA_VISIBLE_DEVICES=1 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk3 \
  --output_dir ./log/eval-Llama2-13B-W8NZ1A8NZ3 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB

# #8 Llama2-13B-W8NZ1A8NZ2
CUDA_VISIBLE_DEVICES=1 $PYTHON eval.py \
  --quant_model_path ./pre_quantized_models/Llama-2-13B-w8a8-popwk1-actk2 \
  --output_dir ./log/eval-Llama2-13B-W8NZ1A8NZ2 \
  --eval_tasks $TASKS --eval_batch_size 8 --max_memory 32GiB
```

---

## 数据来源

- 之前的实验结果来自 `log/` 下对应实验目录的 `log_rank0_*.txt` 以及 `docs/popcount_activation_eval_results.md`。
- 权重 bit-one 统计来自 `scripts/stat_quantized_weight_bit_ones.py`。
- 6 个缺失模型曾保存过（日志中有 "save model to ... success" 记录），但 `pre_quantized_models/` 中的文件已被删除,需重新量化。
