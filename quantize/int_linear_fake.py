import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import utils.hadamard_utils as hadamard_utils


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had = False
        self.use_temporary_parameter = False

        self.use_lora_residual = False
        self.lora_rank = 0
        self.lora_alpha = 1.0
        self.lora_scaling = 1.0
        self.lora_A = None
        self.lora_B = None
        self.enable_lora_forward = True

    def enable_lora_residual(self, rank: int, alpha: float = None):
        rank = int(rank)
        if rank <= 0:
            self.use_lora_residual = False
            self.lora_rank = 0
            return
        rank = min(rank, self.in_features, self.out_features)
        alpha = float(rank if alpha is None else alpha)
        self.use_lora_residual = True
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.lora_scaling = alpha / rank
        device = self.weight.device
        dtype = self.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype), requires_grad=False)

    @torch.no_grad()
    def fit_lora_residual(self, quantized_weight: torch.Tensor, eps: float = 1e-12):
        if not self.use_lora_residual or self.lora_rank <= 0:
            return
        if self.lora_A is None or self.lora_B is None:
            self.enable_lora_residual(self.lora_rank, self.lora_alpha)

        rank = min(self.lora_rank, self.in_features, self.out_features)
        residual = (self.weight.detach().float() - quantized_weight.detach().float())
        out_features, in_features = residual.shape

        # Exact deterministic truncated SVD through the smaller covariance matrix.
        if out_features <= in_features:
            cov = residual @ residual.t()
            eigvals, eigvecs = torch.linalg.eigh(cov)
            top_vals = eigvals[-rank:].clamp_min(0).flip(0)
            u = eigvecs[:, -rank:].flip(1)
            singular = torch.sqrt(top_vals)
            safe_singular = singular.clamp_min(eps)
            vh = (u.t() @ residual) / safe_singular[:, None]
        else:
            cov = residual.t() @ residual
            eigvals, eigvecs = torch.linalg.eigh(cov)
            top_vals = eigvals[-rank:].clamp_min(0).flip(0)
            v = eigvecs[:, -rank:].flip(1)
            singular = torch.sqrt(top_vals)
            safe_singular = singular.clamp_min(eps)
            u = (residual @ v) / safe_singular[None, :]
            vh = v.t()

        sqrt_s = torch.sqrt(singular).to(residual.dtype)
        scaling = float(self.lora_scaling) if self.lora_scaling != 0 else 1.0
        lora_b = u * sqrt_s[None, :]
        lora_a = (sqrt_s[:, None] * vh) / scaling
        self.lora_A.data.copy_(lora_a.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.data.copy_(lora_b.to(device=self.lora_B.device, dtype=self.lora_B.dtype))

    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else:
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias

        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            input = self.input_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        if self.use_lora_residual and self.enable_lora_forward and self.lora_A is not None and self.lora_B is not None:
            lora_hidden = F.linear(input, self.lora_A, None)
            out = out + self.lora_scaling * F.linear(lora_hidden, self.lora_B, None)

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
