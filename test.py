import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
import scipy.stats as st

def test_bolt_acc():
    X = torch.linspace(-2.7, 2.7, 1000)
    layer = nn.SiLU()
    gelu = nn.GELU()
    ## draw the silu function picture, X=input, Y=silu(X)
    Y = gelu(X)
    plt.plot(X.detach().numpy(), Y.detach().numpy(), label="silu")
    # plt.savefig("silu.png")

    abs_X = torch.abs(X)
    a = 0.020848611754127593
    b = -0.18352506127082727
    c = 0.5410550166368381
    d = -0.03798164612714154
    e = 0.001620808531841547
    y_smooth = a * (abs_X ** 4) + b * (abs_X ** 3) + c * (abs_X ** 2) + d * abs_X + e+0.5*X
    print("mse of gelu:", np.mean((Y.detach().numpy() - y_smooth.detach().numpy())**2))

    # Y = gelu(X)
    plt.plot(X.detach().numpy(), y_smooth.detach().numpy(), label="poly")
    plt.legend()
    plt.savefig("gelu.png")
    plt.close()

def test_silu_acc():
    point = 4.6
    X = np.linspace(0, point, 1000)
    layer = nn.SiLU()
    ## draw the silu function picture, X=input, Y=silu(X)
    Y = layer(torch.tensor(X)).detach().numpy()
    
    tmp_Y = Y-0.5*X
    coefficients = np.polyfit(X, tmp_Y, 4)
    print("coefficients:", coefficients)
    a,b,c,d,e = coefficients
    
    X = np.linspace(-15, 15, 1000)
    Y = layer(torch.tensor(X)).detach().numpy()
    plt.plot(X, Y, label="silu")
    abs_X = np.abs(X)

    y_smooth = a * (abs_X ** 4) + b * (abs_X ** 3) + c * (abs_X ** 2) + d * abs_X + e + 0.5*X
    y_smooth[X>point] = X[X>point]
    y_smooth[X<-point] = 0
    print("mse of silu:", np.mean((Y - y_smooth)**2))

    plt.plot(X, y_smooth, label="poly")
    plt.legend()
    plt.savefig("silu.png")
    plt.close()

def compute_worse_prob(model_name):
    if model_name == "llama3":
        stat = torch.load("llama3_bound.pth",weights_only=False)
    elif model_name == "llama2":
        stat = torch.load("llama2_bound.pth",weights_only=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Dictionary to collect values for each component across all layers
    component_k_values = {}  # For k values (smaller is better)
    component_p_values = {}  # For probability values (larger is worse)
    
    # Iterate through all 32 layers
    for layer_idx in range(32):
        layer_data = stat[layer_idx]
        
        for key, value in layer_data.items():
            # Extract component name by removing the layer-specific prefix
            # e.g., "model.layers.0.self_attn.q_proj" -> "self_attn.q_proj"
            component_name = ".".join(key.split(".")[3:])  # Skip "model.layers.X"
            
            if key.endswith("_k"):
                # This is a k value, remove "_k" suffix to get component name
                component_name = component_name[:-2]  # Remove "_k"
                if component_name not in component_k_values:
                    component_k_values[component_name] = []
                component_k_values[component_name].append(value)
            else:
                # This is a probability value
                if component_name not in component_p_values:
                    component_p_values[component_name] = []
                component_p_values[component_name].append(value)
    
    # Compute min k and max p for each component
    results = {}
    
    print("Component-wise statistics across all 32 layers:")
    print("Component\t\t\t\tMin_K\t\tMax_P")
    print("-" * 80)
    
    # Get all unique component names
    all_components = set(component_k_values.keys()) | set(component_p_values.keys())
    
    for component in sorted(all_components):
        min_k = min(component_k_values[component]) if component in component_k_values else float('inf')
        max_p = max(component_p_values[component]) if component in component_p_values else 0.0
        
        results[component] = {
            'min_k': min_k,
            'max_p': max_p
        }
        
        print(f"{component:<40}\t{min_k:.6f}\t{max_p:.6e}")
    
    # Find global minimum k and maximum p across all components
    all_min_k_values = [results[comp]['min_k'] for comp in results if results[comp]['min_k'] != float('inf')]
    all_max_p_values = [results[comp]['max_p'] for comp in results]
    
    global_min_k = min(all_min_k_values) if all_min_k_values else float('inf')
    global_max_p = max(all_max_p_values) if all_max_p_values else 0.0
    
    print(f"\nGlobal minimum k across all components: {global_min_k}")
    print(f"Global maximum p across all components: {global_max_p}")
    
    # Save processed statistics
    torch.save(results, f"{model_name}_component_worse_prob_analysis.pth")
    
    return {
        'global_min_k': global_min_k,
        'global_max_p': global_max_p,
        'component_stats': results
    }

if __name__ == "__main__":
    # test_silu_acc()
    # test_bolt_acc()
    # a = compute_worse_prob("llama2")
    # print(a)
    z_score = 45.4
    prob_tail = st.norm.sf(z_score)
    p_overflow = 2 * prob_tail
    print(p_overflow)
