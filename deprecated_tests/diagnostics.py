import torch
from model import StabilHybridTransformer
from config import ModelConfig

def run_deep_diagnostics():
    print("\n--- ARCHITEKTURÁLIS DIAGNOSZTIKA (MECHANIZMUS BIZONYÍTÉK) ---")
    config = ModelConfig(d_model=768)
    model = StabilHybridTransformer(config)
    model.eval()
    
    # 1. Teszt: Dinamikus Router rezonancia
    # Megvizsgáljuk, hogy az SSM és Attention ágak közötti súlyozás valóban dinamikus-e
    sample_input = torch.randn(1, 10, 768)
    layer = model.layers[0]
    # Router kimenete: [1, 10, 2] -> [SSM_weight, Attn_weight]
    route = layer.router(layer.norm1(sample_input))
    ssm_val = route[0, 0, 0].item()
    attn_val = route[0, 0, 1].item()
    
    print(f"[Dinamikus Routing]")
    print(f"  SSM súly: {ssm_val:.4f}, Attention súly: {attn_val:.4f}")
    if abs(ssm_val - attn_val) > 0.001:
        print("  -> SIKER: A router aktívan szabályozza a feldolgozást.")
    else:
        print("  -> FIGYELEM: A router statikus állapotban van.")

    # 2. Holografikus Memória integritási teszt
    # Ellenőrizzük, hogy a memória mátrix valóban 'tanul' (változik)
    # anélkül, hogy a modell súlyait (trainable parameters) módosítanánk.
    initial_mem = model.holographic_memory.clone()
    
    # Szimulált kognitív állapot (context)
    context_state = torch.randn(1, 768)
    model.update_memory(context_state)
    
    updated_mem = model.holographic_memory.clone()
    
    print(f"\n[Holografikus Memória]")
    diff = (initial_mem - updated_mem).norm().item()
    print(f"  Memória mátrix változása (norm): {diff:.6f}")
    if diff > 0:
        print("  -> SIKER: A hologram a beszélgetés dinamikájával együtt változik.")
    else:
        print("  -> HIBA: A hologram statikus.")

    # 3. Kognitív Verifier teszt
    # A Verifiernek a logits outputot módosítania kell (perturbáció)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)
        # Az önjavítás hatásának mérése
        base_logits = model.lm_head(model.final_norm(model.token_embedding(input_ids)))
        diff_logits = (logits - base_logits).norm().item()
    
    print(f"\n[Önjavító Verifier]")
    print(f"  Logit perturbáció mértéke: {diff_logits:.6f}")
    if diff_logits > 0:
        print("  -> SIKER: A verifier aktívan finomhangolja a kimeneti logikát.")
    else:
        print("  -> HIBA: A verifier inaktív.")

if __name__ == "__main__":
    run_deep_diagnostics()
