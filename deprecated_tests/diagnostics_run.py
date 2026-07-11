import torch
import torch.nn as nn
from model import StabilHybridTransformer
from config import ModelConfig
import os

def run_diagnostics():
    print("\n--- SOTA DIAGNOSZTIKAI ÉRTÉKELÉS ---")
    config = ModelConfig(d_model=768, n_layers=6, vocab_size=5000)
    model = StabilHybridTransformer(config)
    model.eval()

    # 1. Memória Stabilitás Teszt
    print("\n[1/2] Holografikus memória stabilitás vizsgálata...")
    memory_norms = []
    test_input = torch.randint(0, config.vocab_size, (1, 32))
    
    with torch.no_grad():
        for i in range(100):
            # Forward pass triggereli az update_memory-t a model.py-ban
            _ = model(test_input)
            norm = torch.norm(model.holographic_memory).item()
            memory_norms.append(norm)
            
    start_norm, end_norm = memory_norms[0], memory_norms[-1]
    print(f"Memória norma: Kezdő={start_norm:.4f}, Vége={end_norm:.4f}")
    
    if abs(end_norm - start_norm) > 10.0:
        print("FIGYELEM: Memória divergencia detektálva!")
    else:
        print("Memória stabilitás: OK")

    # 2. Router Diagnosztika
    print("\n[2/2] Router szétválás vizsgálata...")
    # Beállítunk egy dummy inputot
    test_input = torch.randn(1, 32, config.d_model)
    
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            # A router kimenetén softmax van, a forward-ban már használjuk
            x_norm = layer.norm1(test_input)
            logits = layer.router(x_norm)
            weights = layer.softmax(logits) # [batch, seq_len, 2]
            
            ssm_avg = weights[..., 0].mean().item()
            attn_avg = weights[..., 1].mean().item()
            
            print(f"Layer {i} Router | SSM: {ssm_avg:.4f} | Attn: {attn_avg:.4f}")
    
    print("\n--- DIAGNOSZTIKA VÉGE ---")

if __name__ == "__main__":
    run_diagnostics()
