import torch
import torch.nn as nn
if __name__ == "__main__":
    latency_accumulation = torch.load("./llama-2-7b-ILP-test_la.pth")
    print(latency_accumulation.keys())
