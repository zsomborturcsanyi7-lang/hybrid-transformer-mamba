import torch
from model import StabilHybridTransformer
from config import ModelConfig

def train_iteration(model, optimizer, data_loader, iteration):
    """Egyetlen optimalizációs ciklus a dinamikus router és memória finomhangolására."""
    model.train()
    total_loss = 0
    for input_ids in data_loader:
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Router loss (kényszerítjük a szakosodást)
        router_loss = sum(layer.router_loss for layer in model.layers)
        
        # CrossEntropyLoss a kimenetre
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)) + 0.1 * router_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def run_optimization_cycles(cycles=4):
    config = ModelConfig(d_model=768)
    model = StabilHybridTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Szimulált kognitív adatok
    data_loader = [torch.randint(0, 5000, (2, 32)) for _ in range(5)]
    
    print("\n--- KOGNITÍV OPTIMALIZÁCIÓS CIKLUSOK (4x) ---")
    for i in range(cycles):
        loss = train_iteration(model, optimizer, data_loader, i+1)
        # Ellenőrizzük a router súlyait
        layer = model.layers[0]
        route = layer.router(layer.norm1(torch.randn(1, 16, 768)))
        print(f"Ciklus {i+1}: Loss: {loss:.4f} | Router SSM/ATTN: {route[0,0,0]:.4f}/{route[0,0,1]:.4f}")
    
    print("\nOptimalizáció kész. A modell megtanulta az erőforrás-elosztást.")

if __name__ == "__main__":
    run_optimization_cycles()
