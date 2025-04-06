import os
import sys
import random
import numpy as np
import torch
import time
import math
from utils.data_utils import get_loaders, test_ppl
from quantize.block_ap import block_ap
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, get_act_stat, init_k_quantizer, init_v_quantizer,get_quant_config,check_quantizer,get_quantized_stat,get_grad_stat
from utils import train_utils
from quantize.int_linear_fake import QuantLinear
from quantize.quant_norm import QuantRMSNorm
from utils.rotation_utils import QKRotationWrapper
from pulp import LpVariable,LpProblem,LpInteger,LpMinimize,GLPK_CMD,LpStatus,value
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from quantize.quantizer import UniformAffineQuantizer
import functools
torch.backends.cudnn.benchmark = True

def main():
    import argparse

    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None,help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,  help="model path of resumed quantized model")
    # -----------------quantization setting------------------------------------
    parser.add_argument("--wbits", type=int, default=16, help="quantization bits")
    parser.add_argument("--w_group_size", type=int, default=-1, help="quantization group size")
    parser.add_argument("--w_asym", dest="w_asym", action="store_true", help="Set w_asym to True")
    parser.add_argument("--w_sym", dest="w_asym", action="store_false", help="Set w_asym to False")
    parser.set_defaults(w_asym=False)
    parser.add_argument("--input_bits", type=int, default=16, help="quantization bits")
    parser.add_argument("--input_group_size", type=int, default=-1, help="quantization group size")
    parser.add_argument("--input_mode", type=str, default='dynamic',help="quantization type")
    parser.add_argument("--input_asym", dest="input_asym", action="store_true", help="Set input_asym to True")
    parser.add_argument("--input_sym", dest="input_asym", action="store_false", help="Set input_asym to False")
    parser.set_defaults(input_asym=False)
    parser.add_argument("--k_bits", type=int, default=16,help="")
    parser.add_argument("--v_bits", type=int, default=16,help="")
    parser.add_argument("--kv_group_size", type=int, default=128,help="default as head-wise")
    parser.add_argument("--k_pre_rope", action="store_true")
    parser.add_argument("--kv_mode", type=str, default='dynamic',help="quantization type")
    parser.add_argument("--kv_asym", dest="kv_asym", action="store_true", help="Set kv_asym to True")
    parser.add_argument("--kv_sym", dest="kv_asym", action="store_false", help="Set kv_asym to False")
    parser.set_defaults(kv_asym=False)
    parser.add_argument("--mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--asym_mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--skip_qk_weight_init", action="store_true")
    parser.add_argument("--block_qk_weight_init", action="store_true")
    parser.add_argument("--mse_init_size", type=int, default=8, help="sample number used in mse_init; actually, even 4 or 2 is enough")
    parser.add_argument("--fp_mse_init", action="store_true", help="use full-precision block input during the mse init process")
    # ----------------- rotation and prefix setting------------------------------------
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--rotate_mode", type=str, default='hadamard')
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    parser.add_argument("--set_prefixed_tokens", action="store_true")
    parser.add_argument("--outlier_threshold", type=int, default=64, help="\eta in Eq.(3), indicating the oitlier threshold ratio detect outlier tokens, ")
    parser.add_argument("--activation_clipping", action="store_true",help="layer-wise activation clipping for dynamic quantization")
    # -----------------training setting------------------------------------
    parser.add_argument("--quant_lr", type=float, default=5e-5, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=5e-6, help="lr of fp weights")
    parser.add_argument("--min_lr_factor", type=float, default=10, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--constant_wlr", action="store_true")
    parser.add_argument("--train_size", type=int, default=512, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=1024, help="lenth of the training sequence.")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--calib_dataset",type=str,default="pile",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
        help="Where to extract calibration data from.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size.")
    parser.add_argument("--loss_type", type=str, default="mse",help="")
    parser.add_argument("--training_target",type=str,default="fp_input",
        choices=["fp_input", "quant_input", "both"],
        help="what is the source of the input to obatin the training target")
    # -----------------evaluation setting------------------------------------
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="65GiB",help="The maximum memory of each GPU")
    # ------------------ others ------------------------------------------
    parser.add_argument("--max_outlier", type=float, default=5,help="")
    parser.add_argument("--max_item_index", type=int, default=5,help="")
    parser.add_argument("--set_outlier_zero", action="store_true")
    parser.add_argument("--modified_index", type=int, default=0,help="")
    # ------------------ ablation ------------------------------------------
    parser.add_argument("--ablate_prefix_number", type=int, default=None,help="")
    # ------------------ ILP ------------------------------------------
    parser.add_argument("--budget", type=int, default=4,help="")
    parser.add_argument("--log_name", type=str,help="")
    parser.add_argument("--cal_size", type=int, default=32,help="")
    parser.add_argument("--acc_la_path", type=str, default=None,help="")
    parser.add_argument("--grad_stat_path", type=str, default=None,help="")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = train_utils.create_logger(output_dir,name=args.log_name)
    logger.info(args)
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
        logger.info(f"model_name is None, setting as {args.model_name}")

    if args.resume_quant:
        # directly load quantized model for evaluation
        model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.group_size)
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,legacy=False,trust_remote_code=True)
        dtype = torch.float16 if not args.use_fp32 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map='cpu',torch_dtype=dtype,trust_remote_code=True,use_safetensors=False)
        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model)
            # logger.info("skip rotate_model")
            rotation_utils.rotate_model(model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            model.half()
        for param in model.parameters():
            param.requires_grad = False
        wrap_to_quant_model(model)
        print("wrap_to_quant_model done")
        # register on-line hadadamrd transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model)
        # wrap rope for online_had and rope output capture
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn, 
                        rope_function_name, 
                        config=model.config,
                        online_had=args.qk_online_had)   

        prefixed_tokens = None                
        prefixed_key_values = None
        args.prefixed_length = 0
        activation_stat = None  
        include_static = (args.input_mode == "static" and args.input_bits < 16 ) or (args.kv_mode == "static" and (args.k_bits<16 or args.v_bits<16))            
        if args.set_prefixed_tokens or include_static:
            from utils.stat_utils import get_prefixed_tokens
            # model and data prepaer
            if model.device.type == 'cpu':
                original_device = 'cpu'
                block_class_name = model.model.layers[0].__class__.__name__
                device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
                model = dispatch_model(model, device_map=device_map)
            else:
                original_device = 'cuda'
            print("get_loaders begin")
            cal_dataloader, _ = get_loaders(
                args.calib_dataset,
                tokenizer,
                train_size=args.cal_size, # 1 for test, should be 64
                val_size=0,
                seed=args.seed,
                seqlen=512,
            )
            print("get_loaders done")
            # get prefixed tokens
            if args.set_prefixed_tokens:
                tick = time.time()
                prefixed_tokens = get_prefixed_tokens(cal_dataloader, model, tokenizer, args.model_name, outlier_threshold=args.outlier_threshold, activation_type='down_proj')
                logger.info(f"get {len(prefixed_tokens)} prefixed tokens; token id:{prefixed_tokens}; text: {tokenizer.decode(prefixed_tokens)}")
                logger.info(f"time to get prefixed token:{time.time()-tick:.0}s")
                model.config.prefixed_tokens = prefixed_tokens
                args.prefixed_length = len(prefixed_tokens)
                use_cache = model.config.use_cache
                model.config.use_cache = True
                if args.ablate_prefix_number is not None:
                    prefixed_tokens = prefixed_tokens[:args.ablate_prefix_number]
                    logger.info(f'ablation:set prefix as {prefixed_tokens}')
                output = model(torch.tensor([prefixed_tokens],device=model.device),return_dict=True)
                prefixed_key_values = output.past_key_values
                model.config.use_cache = use_cache
                
            # get activation statistic for activation quantization
            if include_static:
                # assert args.input_mode == "static" or args.kv_mode == "static","mse_init require static quantization"
                activation_stat = get_act_stat(model, cal_dataloader, 'max', prefixed_tokens, args.down_online_had)
                if args.grad_stat_path is None:
                    get_grad_stat(model, cal_dataloader,logger=logger, prefixed_tokens=prefixed_tokens)
            if original_device == 'cpu':
                remove_hook_from_module(model, recurse=True)
                model = model.cpu()
            # exit(0)
                
        # init weight quantizer
        if args.wbits < 16:
            logger.info('init weight quantizer')
            init_weight_quantizer(args, model)

        # init input quantizer
        if args.input_bits < 16:
            logger.info('init input quantizer')
            init_input_quantizer(args, model, activation_stat)

        # init K quantizer
        if args.v_bits < 16:
            logger.info('init v quantizer')
            init_v_quantizer(args, model, activation_stat)

        # init V quantizer
        if args.k_bits < 16:
            # consistently init for wrap rope 
            logger.info('init k quantizer')
            init_k_quantizer(args, model, activation_stat)
            
        train_utils.cleanup_memory()

    model.half()
    logger.info(model)
    torch.cuda.empty_cache()
    grad_stat = {}
    if args.grad_stat_path is None:
        for name,module in model.named_modules():
            if isinstance(module,LlamaDecoderLayer):
                grad_stat[name] = module.output_grad
        torch.save(grad_stat, args.grad_stat_path)
    else:
        grad_stat = torch.load(args.grad_stat_path)
    print("begin ILP")
    # exit(0)
    ILP(args,model,logger,activation_stat,grad_stat)



def next_power_2(d):
    p = math.ceil(math.log2(d))
    return int(pow(2,p))

# cal_rot and cal_latency are used to compute latency of each layer, each bitwidth
# n: poly degree, (m,d1,d2): GEMM dimension
def cal_rot(n,m,d1,d2):
    # logger.info("n,m,d1,d2,b: %d %d %d %d %d",n,m,d1,d2,b)
    min_rot = 1e8
    d_min = int(min(d2,d1))
    final_mp = 0
    final_d = 0
    for ri in range(1,(d_min)+1):
        for ro in range(1,(d_min)+1):
            d=int(ri*ro)
            m_p=int(n/d)
            if m*d_min<n:
                if d!=d_min:
                    continue
                i = 1
                while i<=m:
                    next_pow_2 = next_power_2(i)
                    if next_pow_2*d>n:
                        break
                    i+=1
                m_p=i-1
            if d>d_min or m_p>m or m_p<=0:
                continue
            tmp=m*d1*(ri-1)/(m_p*d)+m*d2*(ro-1)/(m_p*d)
            if tmp<min_rot:
                min_rot=tmp
                final_d=d
                final_mp = m_p
    # logger.info("final_mp,final_d: %d %d",final_mp,final_d)
    mul = math.ceil(1.0*m/final_mp)*math.ceil(1.0*d1/final_d)*math.ceil(1.0*d2/final_d)*final_d
    return min_rot, mul, final_mp, final_d

# compute latency given layer dimension
# (HW,C,K)=(d1,d2,d3), this fuction can be applied to both convolution and GEMM, t is the b_acc
def cal_latency(HW,C,K,t):
    # logger.info("HW,C,K,b: %d %d %d %d",HW,C,K,b)
    bandwidth = 384*8*(10**6)  # 384MBps
    n=8192
    num_m=1
    q = 2*t+9
    RNS_scale=1.5
    rot, mul, final_mp, final_d = cal_rot(n,HW,C,K)
    num_cts = num_m*math.ceil(HW/final_mp)*(math.ceil(C/final_d)+math.ceil(K/final_d))
    comm = num_cts * n * q * 2
    la_comm = comm/bandwidth
    rot = rot*num_m
    mul = mul*num_m
    la_compute = rot+0.135*mul
    # if plaintext bitwidth > 26-bit, there should be 2 RNS
    if t > 26:
        la_compute = la_compute*RNS_scale
    # print("la_compute,la_comm:",la_compute,la_comm)
    return torch.tensor(la_compute + la_comm).item()

# find the minimum latency given layer dimension and accumulation bitwidth and t_min
# t_min is the minimum bitwidth of the plaintext
def find_min_latency_per_layer(layer,HW,C,K,bw,ba,t_min,b_acc=None):
    # estimate b_acc with the weight
    if b_acc is None:
        # print(layer)
        # print("HW,C,K,bw,ba,t_min:",HW,C,K,bw,ba,t_min)
        layer.weight_quantizer.set_bw(bw,layer.weight)
        weight_int = layer.weight_quantizer.get_int(layer.weight)
        # print("weight_int.shape:",weight_int.shape)
        # b_acc = int(ba + torch.max(torch.ceil(torch.log2(torch.sum(torch.abs(weight_int),dim=(1,2,3))))))
        weight_neg = torch.where(weight_int < 0, weight_int, torch.zeros_like(weight_int))
        weight_pos = torch.where(weight_int > 0, weight_int, torch.zeros_like(weight_int))
        # print("weight_neg:",weight_neg)
        # print("weight_pos:",weight_pos)
        # print("weight_neg.shape:",weight_neg.shape)
        # print("weight_pos.shape:",weight_pos.shape)
        weight_neg = torch.sum(torch.abs(weight_neg),dim=1,keepdim=True)
        weight_pos = torch.sum(weight_pos,dim=1,keepdim=True)
        weight_max = torch.max(weight_neg,weight_pos)
        # print("weight_max.shape:",weight_max.shape)
        b_acc = int(ba + torch.max(torch.ceil(torch.log2(weight_max))))
        # print("b_acc:",b_acc)
        # print("bw+ba+e:",bw+ba+layer.e) 
    b = math.ceil(t_min/b_acc)
    if b==1:
        return cal_latency(HW,C,K,b_acc)
    latency1 = cal_latency(math.ceil(HW/b),C,K,b*b_acc)
    latency2 = cal_latency(HW,math.ceil(C/b),K,b*b_acc)
    latency3 = cal_latency(HW,C,math.ceil(K/b),b*b_acc)
    # print("latency1,latency2,latency3:",latency1,latency2,latency3)
    return min(latency1,latency2,latency3)

def find_min_latency_QK(HW,C,K,bw,ba,t_min,b_acc=None):
    # print("HW,C,K,bw,ba,t_min:",HW,C,K,bw,ba,t_min)
    # estimate b_acc with the weight
    if b_acc is None:
        # print(layer)
        # print("HW,C,K,bw,ba,t_min:",HW,C,K,bw,ba,t_min)
        b_acc = bw+ba+torch.ceil(torch.log2(torch.tensor(C)))
        # print("b_acc:",b_acc)
        # print("bw+ba+e:",bw+ba+layer.e) 
    b = math.ceil(t_min/b_acc)
    if b==1:
        return cal_latency(HW,C,K,b_acc)
    latency1 = cal_latency(math.ceil(HW/b),C,K,b*b_acc)
    latency2 = cal_latency(HW,math.ceil(C/b),K,b*b_acc)
    latency3 = cal_latency(HW,C,math.ceil(K/b),b*b_acc)
    # print("latency1,latency2,latency3:",latency1,latency2,latency3)
    return 2*min(latency1,latency2,latency3)

# return the latency of a transformer block with ba and bw
def cal_latency_per_block(transformer_block,bw,ba,t_min,b_acc=None):
    total_latency = 0
    hidden_dim = 0
    # linear projection
    for name,module in transformer_block.named_modules():
        # print("module:",module)
        if isinstance(module,QuantLinear):
            if hidden_dim == 0:
                hidden_dim = module.in_features
                # print("module.infeatures:",module.infeatures)
            total_latency += find_min_latency_per_layer(module, 1024, module.in_features, module.out_features, bw, ba, t_min, b_acc)   # we fix HW=1024 for now
    # QK^T
    total_latency += find_min_latency_QK(1024, hidden_dim, 1024, bw, ba,t_min,b_acc)
    return total_latency

def cal_b_acc_per_block(transformer_block,bw,ba):
    max_b_acc = 0
    hidden_dim = 0
    # linear projection
    for name,module in transformer_block.named_modules():
        if isinstance(module,QuantLinear):
            if hidden_dim == 0:
                hidden_dim = module.in_features
            module.weight_quantizer.set_bw(bw,module.weight)
            weight_int = module.weight_quantizer.get_int(module.weight)
            # b_acc = int(ba + torch.max(torch.ceil(torch.log2(torch.sum(torch.abs(weight_int),dim=(1,2,3))))))
            weight_neg = torch.where(weight_int < 0, weight_int, torch.zeros_like(weight_int))
            weight_pos = torch.where(weight_int > 0, weight_int, torch.zeros_like(weight_int))
            # print("weight_neg.shape:",weight_neg.shape)
            # print("weight_pos.shape:",weight_pos.shape)
            weight_neg = torch.sum(torch.abs(weight_neg),dim=1,keepdim=True)
            weight_pos = torch.sum(weight_pos,dim=1,keepdim=True)
            weight_max = torch.max(weight_neg,weight_pos)
            b_acc = int(ba + torch.max(torch.ceil(torch.log2(weight_max))))
            max_b_acc = max(max_b_acc,b_acc)
    b_acc_qk = bw+ba+torch.ceil(torch.log2(torch.tensor(hidden_dim)))
    # print("b_acc_qk:",b_acc_qk)
    # print("max_b_acc:",max_b_acc)
    return max(max_b_acc,b_acc_qk)

# need to collect the mean of input X and conduct the block-wise reconstruction
# compute |Z'-Z|*H*|Z'-Z|^T, where Z is the output of the block,H is the mean of Hessian matrix
# Z = layer(X,W), Z' = layer(X',W'), X', W' are the quantized X and W
# in the cal dataset, we use the mean of input X to conduct the block-wise reconstruction for efficiency, instead of use X_i one by one
# TODO: use mse init
def cal_sensitivity_per_block(transformer_block,layer_name,activation_stat,bw,ba,logger,grad_stat=None):
    X = activation_stat[f'{layer_name}.input']
    X = X.half().cuda()
    transformer_block.half().cuda()
    logger.info(f"transformer block name:{layer_name}")
    # print("transformer block name:",layer_name)
    # print("X.dtype:",X.dtype)
    # print("X.device:",X.device)
    hook = []
    act = {}
    def stat_output_hook(module,input,output,name):
        act[name] = output
        
    # for name,module in transformer_block.named_modules():
    #     if isinstance(module,UniformAffineQuantizer) and module.quant_type == "activation":
    #         module.n_bits = 16
    #         hook.append(module.register_forward_hook(functools.partial(stat_output_hook, name=name)))
    Z = transformer_block(X)[0]
    for h in hook:
        h.remove()
    
    def check_mse_hook(module,input,output,name):
        if name in act:
            logger.info(f"{name},MSE:{torch.mean((act[name]-output)**2)}")
    
    # for name,module in transformer_block.named_modules():
    #     if isinstance(module,UniformAffineQuantizer) and module.quant_type == "activation":
    #         hook.append(module.register_forward_hook(functools.partial(check_mse_hook, name=name)))
    
    # for ba in range(2,7):
    for name,module in transformer_block.named_modules():
        if isinstance(module,QuantLinear):
            if 'q_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
            elif 'k_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
            elif 'v_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
                module.output_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.output'].half().cuda())
            elif 'o_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
                module.input_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.input'].half().cuda())
            elif 'gate_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
            elif 'up_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
            elif 'down_proj' in name:
                module.weight_quantizer.set_bw(bw,module.weight.data)
                module.input_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.input'].half().cuda())
            else:
                raise ValueError(f"Unknown layer: {name}")
            
        elif isinstance(module,QuantRMSNorm):
            module.output_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.output'].half().cuda())
            # module.output_quantizer.n_bits = 16
            # x1 = module(X)
            # if "input_layernorm" in name:
            #     for bit in range(2,7):
            #         module.output_quantizer.set_bw(bit,activation_stat[f'{layer_name}.{name}.output'].half().cuda())
            #         # module.cuda()
            #         x2 = module(X)
            #         logger.info(f"{layer_name}.{name}.n_bits:{module.output_quantizer.n_bits}, MSE:{torch.mean((x1-x2)**2)}")
        elif isinstance(module,QKRotationWrapper):
            module.q_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.output_Q'].half().cuda())
            module.k_quantizer.set_bw(ba,activation_stat[f'{layer_name}.{name}.output_K'].half().cuda())
    Z_prime = transformer_block(X)[0]
    logger.info(f"{layer_name},bw:{bw},ba:{ba},mean((Z-Z_prime)**2):{torch.mean((Z-Z_prime)**2)}")
    if grad_stat is not None:
        sensitivity = torch.sum(((Z-Z_prime)**2) * ((grad_stat[layer_name] * 1e3) **2)) # multiply 1e7 to avoid 0
    else:
        sensitivity = torch.sum(((Z-Z_prime)**2))* ((transformer_block.output_grad * 1e3) **2)
    return sensitivity.item()

def ILP(args,model,logger,activation_stat,grad_stat):
    target_bw = args.budget
    idx = 0
    origin_latency = 0
    cir_idx = []
    wb_list = [2,3,4,5,6]
    ab_list = [2,3,4,5,6]
    sensitivity = {}
    logger.info("target_bw:"+str(target_bw))
    t_min = 18
    latency_accumulation = {}
    for wb in wb_list:
        for ab in ab_list:
            sensitivity["w"+str(wb)+"a"+str(ab)] = []
    # in LLM, each layer has the same latency
    for i in range(6,27):
        latency_accumulation["b"+str(i)] = 0
    if args.acc_la_path is not None:
        latency_accumulation = torch.load(args.acc_la_path)
    for name,layer in model.named_modules():
        if isinstance(layer, LlamaDecoderLayer):
            cir_idx.append(idx)
            
            for wb in wb_list:
                for ab in ab_list:
                    sensitivity["w"+str(wb)+"a"+str(ab)].append(cal_sensitivity_per_block(layer,name,activation_stat,wb,ab,logger,grad_stat))
                    logger.info("idx:"+str(idx)+"sensitivity_w"+str(wb)+"a"+str(ab)+":"+str(sensitivity["w"+str(wb)+"a"+str(ab)][-1]))
            # logger.info("d1:"+str(layer.d1))
            if latency_accumulation["b6"] == 0:
                for i in range(6,27):
                    latency_accumulation["b"+str(i)] += cal_latency_per_block(layer,1,1,t_min,i)
                logger.info("latency_accumulation_b"+str(i)+":"+str(latency_accumulation["b"+str(i)]))
            # origin_latency += cal_latency_per_block(layer,args.budget,args.budget,t_min) 
            # latency_weights_b32.append(cal_latency(layer.d1,layer.in_channels,layer.out_channels,32,space))
            idx+=1
    if args.acc_la_path is None:
        torch.save(latency_accumulation,f"./{args.log_name}_la.pth")
    origin_latency = 90710761
    logger.info("origin_latency:"+str(origin_latency))
    # params = hessian_comp.params
    
    logger.info("layer_idx:"+str(cir_idx))
    
    # print(traces)
    num_variable = len(cir_idx)
    variable = {}
    for i in range(num_variable):
        for wb in wb_list:
            for ab in ab_list:
                variable[f"wb{wb}ab{ab}_{i}"] = LpVariable(f"wb{wb}ab{ab}_{i}", 0, 1, cat=LpInteger)
    
    prob = LpProblem("Bitwidth", LpMinimize)
    idx = 0
    tmp = None
    acc_list = {}
    for name,layer in model.named_modules():
        if isinstance(layer, LlamaDecoderLayer):
            for wb in wb_list:
                for ab in ab_list:
                    total_bw = cal_b_acc_per_block(layer,wb,ab)
                    total_bw = int(total_bw)
                    # print("total_bw:",total_bw)
                    if tmp is None:
                        tmp = variable[f"wb{wb}ab{ab}_{idx}"]*latency_accumulation[f"b{total_bw}"]
                    else:
                        tmp += variable[f"wb{wb}ab{ab}_{idx}"]*latency_accumulation[f"b{total_bw}"]
                    acc_list[f"wb{wb}ab{ab}_{idx}"] = total_bw
            idx+=1
        
    prob += (tmp-origin_latency <= 0.01)
    
    #one layer only have one blocksize
    for i in range(num_variable):
        tmp = None
        for wb in wb_list:
            for ab in ab_list:
                if tmp is None:
                    tmp = variable[f"wb{wb}ab{ab}_{i}"]
                else:
                    tmp += variable[f"wb{wb}ab{ab}_{i}"]
        prob += (tmp == 1)

    # delta_weights_b32 = np.array(delta_weights_b32)
    # sensitivity_b32 = traces * delta_weights_b32
    # optimization target: minimize the sensitivity
    idx = 0
    tmp2 = None
    for name,layer in model.named_modules():
        if "first" in name:
            continue
        if isinstance(layer, LlamaDecoderLayer):
            for wb in wb_list:
                for ab in ab_list:
                    if tmp2 is None:
                        tmp2 = variable[f"wb{wb}ab{ab}_{idx}"]*sensitivity[f"w{wb}a{ab}"][idx]
                    else:
                        tmp2 += variable[f"wb{wb}ab{ab}_{idx}"]*sensitivity[f"w{wb}a{ab}"][idx]
            idx += 1
    prob += tmp2
    # prob += sum(variable[f"b2_{i}"]*sensitivity_b2[i] + variable[f"b4_{i}"]*sensitivity_b4[i] +variable[f"b8_{i}"]*sensitivity_b8[i] +variable[f"b16_{i}"]*sensitivity_b16[i] for i in range(num_variable))

    status = prob.solve(GLPK_CMD(msg=1, mip=1, options=["--tmlim", "10000","--simplex"]))
    
    LpStatus[status]

    bw_result = []
    ba_result = []
    acc_result = []
    current_latency = 0
    for i in range(num_variable):
        for wb in wb_list:
            for ab in ab_list:
                if value(variable[f"wb{wb}ab{ab}_{i}"]) == 1:
                    bw_result.append(wb)
                    ba_result.append(ab)
                    acc_result.append(acc_list[f"wb{wb}ab{ab}_{i}"])
                    break
        # elif block_size == 32:
        #     current_latency += latency_weights_b32[i]
    logger.info(str(len(bw_result)))
    idx = 0
    logger.info("target: w"+str(args.budget)+"a"+str(args.budget))
    logger.info("bw_result:"+str(bw_result))
    logger.info("ba_result:"+str(ba_result))
    logger.info("acc_result:"+str(acc_result))

if __name__ == "__main__":
    print(sys.argv)
    main()
