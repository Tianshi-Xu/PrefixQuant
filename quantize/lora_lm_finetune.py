"""
Standard LoRA fine-tuning pipeline with cross-entropy (next-token prediction) loss.

Unlike block_ap.py which trains layer-by-layer with MSE reconstruction loss,
this module runs the full model forward pass and optimizes LoRA A/B parameters
to minimize language modeling loss (cross-entropy).

Key design:
  - Only LoRA A/B parameters are trainable; all scale/zero_point are frozen.
  - Weights are already quantized in-place (by block_ap + quant_inplace).
  - Activation quantization runs online during forward (STE allows gradient flow).
  - Supports prefixed_key_values (prefix quantization context).
  - Supports fine-tuning on either train split or test split of a dataset.
"""

import torch
import math
import random
import re
from contextlib import nullcontext
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module

from utils.quant_utils import (
    lora_parameters,
    set_lora_parameters,
    set_quant_state,
    set_lora_forward,
    trainable_parameters_num,
)
from utils.train_utils import NativeScalerWithGradNormCount
from utils.model_utils import mv_kv_cache, get_kv_cache
from utils.data_utils import get_loaders


def _warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_factor=0.1):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_factor) + min_factor
    return LambdaLR(optimizer, lr_lambda)


def _hellaswag_preprocess(text):
    """Preprocess hellaswag text (same as lm_eval)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _load_downstream_data(tokenizer, seqlen, seed, logger=None):
    """Load downstream task evaluation data for fine-tuning.

    Constructs (context, answer) pairs from 7 downstream tasks:
    piqa, arc_easy, arc_challenge, hellaswag, winogrande, lambada, openbookqa.

    Returns list of (input_ids, labels) tuples where labels mask context tokens with -100.
    """
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    samples = []
    task_counts = {}

    def add_sample(context, answer):
        """Tokenize context+answer, mask context tokens in labels."""
        ctx_ids = tokenizer.encode(context, add_special_tokens=True)
        ans_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(ans_ids) == 0:
            return

        input_ids_list = ctx_ids + ans_ids
        labels_list = [-100] * len(ctx_ids) + list(ans_ids)

        # Truncate from left if too long (keep answer intact)
        if len(input_ids_list) > seqlen:
            if len(ans_ids) >= seqlen:
                input_ids_list = ans_ids[-seqlen:]
                labels_list = list(ans_ids[-seqlen:])
            else:
                keep_ctx = seqlen - len(ans_ids)
                input_ids_list = ctx_ids[-keep_ctx:] + ans_ids
                labels_list = [-100] * keep_ctx + list(ans_ids)

        input_ids = torch.tensor([input_ids_list], dtype=torch.long)
        labels = torch.tensor([labels_list], dtype=torch.long)
        samples.append((input_ids, labels))

    # 1. PIQA (validation split, ~1838 samples)
    try:
        ds = datasets.load_dataset("piqa", split="validation", trust_remote_code=True)
        for doc in ds:
            ctx = f"Question: {doc['goal']}\nAnswer:"
            ans = " " + [doc['sol1'], doc['sol2']][doc['label']]
            add_sample(ctx, ans)
        task_counts['piqa'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load piqa: {e}")

    # 2. ARC-Easy (test split, ~2376 samples)
    try:
        ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
        for doc in ds:
            ctx = f"Question: {doc['question']}\nAnswer:"
            labels_list = doc['choices']['label']
            answer_idx = labels_list.index(doc['answerKey'])
            ans = " " + doc['choices']['text'][answer_idx]
            add_sample(ctx, ans)
        task_counts['arc_easy'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load arc_easy: {e}")

    # 3. ARC-Challenge (test split, ~1172 samples)
    try:
        ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        for doc in ds:
            ctx = f"Question: {doc['question']}\nAnswer:"
            labels_list = doc['choices']['label']
            answer_idx = labels_list.index(doc['answerKey'])
            ans = " " + doc['choices']['text'][answer_idx]
            add_sample(ctx, ans)
        task_counts['arc_challenge'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load arc_challenge: {e}")

    # 4. HellaSwag (validation split, ~10003 samples)
    try:
        ds = datasets.load_dataset("hellaswag", split="validation", trust_remote_code=True)
        for doc in ds:
            if doc['label'] == "":
                continue
            ctx = _hellaswag_preprocess(doc['activity_label'] + ": " + doc['ctx_a'] + " " + doc['ctx_b'].capitalize())
            gold = int(doc['label'])
            ans = " " + _hellaswag_preprocess(doc['endings'][gold])
            add_sample(ctx, ans)
        task_counts['hellaswag'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load hellaswag: {e}")

    # 5. Winogrande (validation split, ~1767 samples)
    try:
        ds = datasets.load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        for doc in ds:
            idx = doc['sentence'].index('_')
            answer_to_num = {"1": 0, "2": 1}
            gold = answer_to_num[doc['answer']]
            options = [doc['option1'], doc['option2']]
            ctx = doc['sentence'][:idx] + options[gold]
            ans = " " + doc['sentence'][idx + 1:].strip()
            add_sample(ctx, ans)
        task_counts['winogrande'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load winogrande: {e}")

    # 6. LAMBADA (test split, ~5153 samples)
    try:
        ds = datasets.load_dataset("lambada", split="test", trust_remote_code=True)
        for doc in ds:
            words = doc['text'].split(' ')
            ctx = ' '.join(words[:-1])
            ans = " " + words[-1]
            add_sample(ctx, ans)
        task_counts['lambada'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load lambada: {e}")

    # 7. OpenBookQA (test split, ~500 samples)
    try:
        ds = datasets.load_dataset("openbookqa", "main", split="test", trust_remote_code=True)
        for doc in ds:
            ctx = doc['question_stem']
            labels_list = doc['choices']['label']
            answer_idx = labels_list.index(doc['answerKey'].lstrip())
            ans = " " + doc['choices']['text'][answer_idx]
            add_sample(ctx, ans)
        task_counts['openbookqa'] = len(ds)
    except Exception as e:
        if logger: logger.warning(f"Failed to load openbookqa: {e}")

    if logger:
        logger.info(f"Downstream task sample counts: {task_counts}")
        logger.info(f"Total downstream samples: {len(samples)}")

    random.seed(seed)
    random.shuffle(samples)
    n_train = max(1, int(len(samples) * 0.9))
    return samples[:n_train], samples[n_train:]


def _load_ft_data(tokenizer, dataset_name, seqlen, seed, split, train_size, val_size, logger=None):
    """Load fine-tuning data.

    split='train': use train/val split from get_loaders (standard).
    split='test':  use the test set, split 90/10 into train/val.
    dataset_name='downstream_tasks': load 7 downstream task eval sets.

    Returns list of (input_ids, labels) tuples.
    """
    if dataset_name == "downstream_tasks":
        return _load_downstream_data(tokenizer, seqlen, seed, logger)

    if split == "test":
        testenc = get_loaders(dataset_name, tokenizer, seed=seed, seqlen=seqlen, test_only=True)
        input_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
        nsamples = input_ids.numel() // seqlen
        samples = []
        for i in range(nsamples):
            inp = input_ids[:, i * seqlen:(i + 1) * seqlen]
            tar = inp.clone()
            samples.append((inp, tar))
        n_train = max(1, int(nsamples * 0.9))
        return samples[:n_train], samples[n_train:]
    else:
        trainloader, valloader = get_loaders(
            dataset_name, tokenizer,
            train_size=train_size, val_size=val_size,
            seed=seed, seqlen=seqlen,
        )
        train_samples = [(inp, tar) for inp, tar in trainloader]
        val_samples = [(inp, tar) for inp, tar in valloader]
        return train_samples, val_samples


def lora_lm_finetune(model, tokenizer, prefixed_key_values, args, logger):
    """
    Standard LoRA fine-tuning with cross-entropy (next-token prediction) loss.

    Only LoRA A/B parameters are trainable; all scale/zero_point are frozen.

    Prerequisites (handled by block_ap before this function is called):
      - Weights quantized in-place via quant_inplace.
      - LoRA initialized via init_lora_from_current_residual.
    """
    logger.info("=== Start LoRA LM fine-tuning (cross-entropy loss) ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Dispatch model to GPU -------------------------------------------
    use_cache_orig = model.config.use_cache
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(
        model,
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
        no_split_module_classes=[block_class_name],
    )
    model = dispatch_model(model, device_map=device_map, skip_keys="past_key_values")
    prefixed_key_values = mv_kv_cache(prefixed_key_values, model)

    # --- 2. Freeze all params, enable grad only for LoRA A/B -----------------
    for p in model.parameters():
        p.requires_grad = False
    set_lora_parameters(model, True)

    # Cast LoRA params to FP32 for proper gradient scaling (model is in FP16)
    for p in lora_parameters(model):
        p.data = p.data.float()

    lora_params = list(lora_parameters(model))
    if len(lora_params) == 0:
        logger.warning("No LoRA parameters found! Ensure --use_lora_residual is set. Skipping fine-tuning.")
        remove_hook_from_module(model, recurse=True)
        model = model.cpu()
        return model

    trainable_num = trainable_parameters_num(model)
    logger.info(f"Trainable parameters: {trainable_num / 1e6:.2f}M")

    # --- 3. Ensure quant state: weights already quantized, act quant online --
    set_quant_state(model, weight_quant=False, act_quant=True)
    set_lora_forward(model, True)

    # --- 4. Config: use_cache must be True for prefixed_key_values -----------
    model.config.use_cache = True

    # --- 5. No wrapper; pass past_key_values directly via get_kv_cache -------
    #    (WrappedPrefixCausalLM passes legacy tuple which is incompatible
    #     with gradient checkpointing recomputation in transformers 4.40.1)
    train_model = model

    # --- 6. Load data ---------------------------------------------------------
    train_samples, val_samples = _load_ft_data(
        tokenizer, args.lora_ft_dataset, args.lora_ft_seqlen,
        args.seed, args.lora_ft_split,
        args.lora_ft_train_size, args.lora_ft_val_size,
        logger=logger,
    )
    logger.info(
        f"FT data: {len(train_samples)} train, {len(val_samples)} val "
        f"(dataset={args.lora_ft_dataset}, split={args.lora_ft_split}, seqlen={args.lora_ft_seqlen})"
    )

    # --- 7. Optimizer & scheduler --------------------------------------------
    optimizer = AdamW(lora_params, lr=args.lora_ft_lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_samples) // args.lora_ft_batch_size)
    total_steps = args.lora_ft_epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.lora_ft_warmup_ratio)
    scheduler = _warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    loss_scaler = NativeScalerWithGradNormCount()

    # --- 8. AMP context -------------------------------------------------------
    use_bf16 = getattr(args, "use_bf16", False)
    use_fp32 = getattr(args, "use_fp32", False)
    if use_bf16:
        traincast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif use_fp32:
        traincast = nullcontext
    else:
        traincast = torch.cuda.amp.autocast

    # --- 9. Training loop -----------------------------------------------------
    for epoch in range(args.lora_ft_epochs):
        model.train()
        epoch_loss = 0.0
        num_steps = 0

        indices = list(range(len(train_samples)))
        random.shuffle(indices)

        pbar = tqdm(
            range(0, len(indices), args.lora_ft_batch_size),
            desc=f"LoRA-FT epoch {epoch + 1}/{args.lora_ft_epochs}",
        )
        for i in pbar:
            batch_idx = indices[i:i + args.lora_ft_batch_size]
            if len(batch_idx) < args.lora_ft_batch_size:
                continue

            # Each sample is (input_ids, labels) tuple; labels may mask context tokens with -100
            input_ids = torch.cat([train_samples[j][0] for j in batch_idx], dim=0).to(dev)
            labels = torch.cat([train_samples[j][1] for j in batch_idx], dim=0).to(dev)
            bs = input_ids.shape[0]
            past_kv = get_kv_cache(prefixed_key_values, bs=bs) if prefixed_key_values is not None else None

            with traincast():
                outputs = train_model(input_ids=input_ids, labels=labels, past_key_values=past_kv)
                loss = outputs.loss

            if not math.isfinite(loss.item()):
                logger.warning(f"Loss NaN/Inf at step {num_steps}, skipping")
                continue

            optimizer.zero_grad()
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad,
                parameters=lora_params,
            )
            scheduler.step()

            epoch_loss += loss.item()
            num_steps += 1
            if num_steps % 10 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.2e}"})

        avg_loss = epoch_loss / max(num_steps, 1)
        logger.info(
            f"LoRA-FT epoch {epoch + 1}/{args.lora_ft_epochs} "
            f"avg_train_loss: {avg_loss:.6f}"
        )

        # --- Validation ---
        if val_samples:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for i in range(0, len(val_samples), args.lora_ft_batch_size):
                    batch_idx = list(range(i, min(i + args.lora_ft_batch_size, len(val_samples))))
                    if len(batch_idx) < args.lora_ft_batch_size:
                        continue
                    input_ids = torch.cat([val_samples[j][0] for j in batch_idx], dim=0).to(dev)
                    labels = torch.cat([val_samples[j][1] for j in batch_idx], dim=0).to(dev)
                    bs = input_ids.shape[0]
                    past_kv = get_kv_cache(prefixed_key_values, bs=bs) if prefixed_key_values is not None else None
                    with traincast():
                        outputs = train_model(input_ids=input_ids, labels=labels, past_key_values=past_kv)
                        val_loss_sum += outputs.loss.item()
                        val_steps += 1
            val_loss = val_loss_sum / max(val_steps, 1)
            logger.info(f"LoRA-FT epoch {epoch + 1} val_loss: {val_loss:.6f}")

    # --- 10. Cleanup ----------------------------------------------------------
    optimizer.zero_grad()
    del optimizer, scheduler

    # Cast LoRA params back to FP16 to match the rest of the model for eval
    for p in lora_parameters(model):
        p.data = p.data.half()

    # Remove accelerate dispatch hooks, move back to CPU for saving / eval
    remove_hook_from_module(model, recurse=True)
    model = model.cpu()
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache_orig
    logger.info("=== LoRA LM fine-tuning finished ===")
    return model
