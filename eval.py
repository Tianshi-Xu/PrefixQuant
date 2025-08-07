import os
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention
import quantize.int_linear_fake as int_linear_fake
from utils.quant_utils import init_s_quantizer
from quantize.quantizer import UniformAffineQuantizer
import functools
from safetensors.torch import load_file
import ast
torch.backends.cudnn.benchmark = True

def compute_bacc(model):
    b_acc_map = {}
    for _,module in model.named_modules():
        if isinstance(module,LlamaDecoderLayer):
            for name,layer in module.named_modules():
                if isinstance(layer,int_linear_fake.QuantLinear):
                    weight_int = layer.weight_quantizer.get_int(layer.weight)
                    print("name,weight_int.shape",name,weight_int.shape)
                    weight_neg = torch.where(weight_int < 0, weight_int, torch.zeros_like(weight_int))
                    weight_pos = torch.where(weight_int > 0, weight_int, torch.zeros_like(weight_int))
                    weight_neg = torch.sum(torch.abs(weight_neg),dim=-1,keepdim=True)
                    weight_pos = torch.sum(weight_pos,dim=-1,keepdim=True)
                    weight_max = torch.max(weight_neg,weight_pos)
                    b_acc = int(4 + torch.max(torch.ceil(torch.log2(weight_max))))
                    if name in b_acc_map:
                        b_acc_map[name] = max(b_acc_map[name],b_acc)
                    else:
                        b_acc_map[name] = b_acc
    return b_acc_map

stat = {}
def add_overflow_hook(model):
    
    def overflow_stat(module, x, name):
        # print(name)
        x = x[0]
        # print(x.shape)
        x_int = module.get_int(x)
        # print(x_int.shape)
        ema_factor = 0.99
        # x_below_qmin = torch.where(x_int < module.qmin, x_int, torch.zeros_like(x_int))
        # x_above_qmax = torch.where(x_int > module.qmax, x_int, torch.zeros_like(x_int))
        if name+"_min" not in stat:
            stat[name+"_min"] = torch.min(x_int)
        else:
            stat[name+"_min"] = torch.min(stat[name+"_min"],torch.min(x_int))
        if name+"_max" not in stat:
            stat[name+"_max"] = torch.max(x_int)
        else:
            stat[name+"_max"] = torch.max(stat[name+"_max"],torch.max(x_int))
        if name+"_var" not in stat:
            stat[name+"_var"] = torch.var(x_int)
        else:
            stat[name+"_var"] = ema_factor * stat[name+"_var"] + (1-ema_factor) * torch.var(x_int)
        
        # for bit in [4,5,6,7,8]:
        #     lower_bound = -2**(bit-1)
        #     upper_bound = 2**(bit-1) - 1
        #     x_in_range_rate = torch.sum((x_int >= lower_bound) & (x_int <= upper_bound))/torch.numel(x_int)
        #     x_out_range_rate = 1 - x_in_range_rate
        #     if name + "_out_"+str(bit) not in stat:
        #         stat[name + "_out_"+str(bit)] = x_out_range_rate
        #     else:
        #         stat[name + "_out_"+str(bit)] = ema_factor * stat[name + "_out_"+str(bit)] + (1-ema_factor) * x_out_range_rate
        #     # print(f"name:{name}, bit:{bit}, x_out_range_rate:{x_out_range_rate}, ema:{stat[name + '_out_'+str(bit)]}")
        
    
    for name,module in model.named_modules():
        if isinstance(module, UniformAffineQuantizer) and module.quant_type == "activation":
            module.register_forward_pre_hook(functools.partial(overflow_stat, name=name))

def compute_bound(model):
    stat = torch.load("llama3_stat.pth")
    for _,module in model.named_modules():
        if isinstance(module,LlamaDecoderLayer):
            for name,layer in module.named_modules():
                if isinstance(layer,int_linear_fake.QuantLinear):
                    print(name)
                    # weight_int = layer.weight_quantizer.get_int(layer.weight)
                    # weight_neg = torch.where(weight_int < 0, weight_int, torch.zeros_like(weight_int))
                    # weight_pos = torch.where(weight_int > 0, weight_int, torch.zeros_like(weight_int))
                    # weight_neg = torch.sum(torch.abs(weight_neg),dim=-1,keepdim=True)
                    # weight_pos = torch.sum(weight_pos,dim=-1,keepdim=True)
                    # weight_max = torch.max(weight_neg,weight_pos)
                    # T_bound = (2**3 - 1) * weight_max
                    # k = T_bound/

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    logger.info(f"args: {args}")

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None
    prefix_len = len(quant_config.prefixed_tokens)
    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path,trust_remote_code=True)
    config._attn_implementation = "eager"
    config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    for name,layer in model.named_modules():
        if isinstance(layer,LlamaAttention):
            layer.prefix_len = prefix_len
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config,
                    online_had=quant_config.qk_online_had)

    # init weight quantizer
    if quant_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(quant_config, model, minmax_init=False)

    # init input quantizer
    if quant_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(quant_config, model,  minmax_init=False)

    # init kv quantizer
    if quant_config.v_bits < 16:
        logger.info('init v quantizer')
        init_v_quantizer(quant_config, model,  minmax_init=False)

    # if True:
    if quant_config.k_bits < 16:
        # consistently init for wrap rope 
        logger.info('init k quantizer')
        init_k_quantizer(quant_config, model,  minmax_init=False)
    
    if quant_config.s_bits < 16:
        logger.info('init s quantizer')
        init_s_quantizer(quant_config, model, minmax_init=False)

    # model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    model.half()    # to make sure same evaluation results with main
    logger.info(model)
    compute_bound(model)
    # b_acc = compute_bacc(model)
    # add_overflow_hook(model)
    # logger.info(f"b_acc: {b_acc}")

    evaluate(model, tokenizer, prefixed_key_values,  args,logger)
    # torch.save(stat, "llama2_stat.pth")



if __name__ == "__main__":
    # print(sys.argv)
    # checkpoint = load_file(os.path.join("pre_quantized_models/Llama-3-8B-w4a4q4s8kv4/", "model-00001-of-00004.safetensors"))
    # for key in checkpoint.keys():
    #     if "quantizer" in key:
    #         print(key,checkpoint[key].shape)
    # exit(0)
    
    # print(stat)
    main()
    # stat = torch.load("llama2_stat.txt")
    # compute_bound(stat,model)
    # print(stat)
