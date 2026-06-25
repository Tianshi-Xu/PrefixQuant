### PrefixQuant weight bit-one stats

PYTHON=/data/home/xts/miniconda3/envs/prefixquant/bin/python

if [ "$#" -gt 0 ]; then
  MODEL_DIRS="$@"
else
  MODEL_DIRS="
./pre_quantized_models/Llama-2-13B-w8a16-popwk2
./pre_quantized_models/Llama-2-7B-w8a16-popwk2
"
fi

for MODEL_DIR in $MODEL_DIRS; do
  echo "===== ${MODEL_DIR} ====="
  ${PYTHON} scripts/stat_quantized_weight_bit_ones.py \
  ${MODEL_DIR} \
  --storage-bits 8 \
  --count-mode magnitude
done
