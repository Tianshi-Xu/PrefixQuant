import torch
import torch.nn as nn
from quantize.quantizer import UniformAffineQuantizer

def clamp_mse(outlier_threshold=50):
    def func(output, label):
        output = output.clamp(min=-outlier_threshold, max=outlier_threshold)               
        label = label.clamp(min=-outlier_threshold, max=outlier_threshold)               
        return torch.nn.functional.mse_loss(output, label)
    return func

def skip_mse(skip_num=3):
    def func(output, label):
        # adaptive select the outlier token
        mean_value =  label.abs().mean(dim=-1)
        select_index =  mean_value.topk(skip_num,dim=-1)[1]
        mask = torch.ones_like(mean_value)
        batch_indices = torch.arange(mean_value.shape[0]).unsqueeze(1)
        mask[batch_indices, select_index] = 0
        mask = mask.unsqueeze(-1)
        return ((output-label)**2*mask).mean()
        # else:
        #     return ((output[:, prefix_length:]-label[:, prefix_length:])**2).mean()
            
    return func

def normalized_mse():
    def func(output, label):
        max_value = label.abs().max(dim=2,keepdim=True)[0]
        return torch.nn.functional.mse_loss(output/max_value, label/max_value)
    return func


def cosine_loss():
    def func(output, label):
        cosine = torch.nn.functional.cosine_similarity(output, label, -1)
        return  1 - cosine.mean()
    return  func


def get_recon_loss(loss_type, prefixed=False, prefix_length=0):
    if loss_type == "mse":
        loss_func = torch.nn.MSELoss()
    elif loss_type == "clamp_mse":
        loss_func = clamp_mse(10)
    elif loss_type == "skip_mse":
        loss_func = skip_mse()
    elif loss_type == "normalized_mse":
        loss_func = normalized_mse()
    elif loss_type == "cosine":
        loss_func = cosine_loss()
    else:
        raise NotImplementedError
    return loss_func


class NegativeActivationPenaltyLoss(nn.Module):
    def __init__(self, base_loss_fn, penalty_weight=0.1):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.penalty_weight = penalty_weight
        
    def forward(self, output, target, model=None):
        base_loss = self.base_loss_fn(output, target)
        
        # If no model is provided, return only the base loss
        if model is None:
            return base_loss
            
        # Collect negative activation penalties
        neg_penalty = 0.0
        for name, module in model.named_modules():
            if isinstance(module, UniformAffineQuantizer) and hasattr(module, 'last_output'):
                # print(name)
                # Calculate penalty for negative values: sum of squares of negative values
                # print("module.last_output.shape", module.last_output.shape)
                tmp_max = torch.max(module.last_output)
                
                # print("group_size", module.group_size)
                # print(name)
                # print("tmp_max.shape", tmp_max.shape)
                neg_penalty += tmp_max**2
        print(f"neg_penalty: {neg_penalty}")
        print(f"negative_penalty_weight*neg_penalty: {self.penalty_weight*neg_penalty}")
        print(f"base_loss: {base_loss}")
        # exit(0)
        total_loss = base_loss + self.penalty_weight * neg_penalty
        return total_loss

def get_negative_penalty_loss(base_loss_type, penalty_weight=0.1):
    base_loss = get_recon_loss(base_loss_type)
    return NegativeActivationPenaltyLoss(base_loss, penalty_weight)