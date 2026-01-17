from attention import BDHLinearAttention

import time
import torch
import torch.nn as nn

def test_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BDHLinearAttention(dim=128, expansion_factor=4, heads=4).to(device)
    model.eval()
    
    # Warm up the GPU
    _ = model(torch.randn(1, 10000, 128).to(device)) 
    
    print("[TEST 1]")
    
    N_tiny = 1000
    x_tiny = torch.randn(1, N_tiny, 128).to(device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    _ = model(x_tiny)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_tiny = time.time() - start
    
    N_huge = 10000
    x_huge = torch.randn(1, N_huge, 128).to(device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    try:
        _ = model(x_huge)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_huge = time.time() - start
        
        scaling_ratio = t_huge/t_tiny
        print(f"Time N={N_tiny}: {t_tiny}")
        print(f"Time N={N_huge}: {t_huge}")
        print(f"Scaling Ration: {scaling_ratio}")
        
        if scaling_ratio < 15.0:
            print("TEST PASSED")
        else:
            print("WARNING: SCALING MIGHT BE QUADRATIC")
    
    except RuntimeError as e:
        print(f"OUT OF MEMORY on N={N_huge} \n {e}")

if __name__ == "__main__":
    test_attention()