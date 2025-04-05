import torch
import torch.nn as nn
if __name__ == "__main__":
    grad_stat = torch.load("./llama2-7b-grad.pth")
    print(grad_stat.keys())
    for key in grad_stat.keys():
        print(key)
        print(grad_stat[key].shape)
