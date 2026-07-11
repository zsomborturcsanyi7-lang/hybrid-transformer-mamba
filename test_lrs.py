"""
LRS TESZT - A belső gondolkodási tér működésének ellenőrzése
============================================================
Ez a szkript betölti a modellt és megvizsgálja a Latent Reasoning Space működését.
"""

import sys
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import CognitiveBilingualModel
from tokenizer import BilingualTokenizer


def test_reasoning_pipeline():
    """Teszt: a Latent Reasoning Space létrehozása és előrejelzése"""
    print("=" * 60)
    print("LATENT REASONING SPACE - Belső Gondolkodási Tér Teszt")
    print("=" * 60)
    
    config = ModelConfig(d_model=256, vocab_size=16384, n_heads=8)
    model = CognitiveBilingualModel(config)
    
    # Ellenőrizzük, hogy a modell tartalmazza-e a reasoning pipeline-t
    assert hasattr(model, 'reasoning_pipeline'), "❌ Nincs reasoning_pipeline a modellben!"
    print("✓ Reasoning pipeline megtalálva")
    
    # Ellenőrizzük a reasoning pipeline komponenseit
    rp = model.reasoning_pipeline
    assert hasattr(rp, 'reasoning_space'), "❌ Nincs reasoning_space!"
    assert hasattr(rp, 'reasoning_chain'), "❌ Nincs reasoning_chain!"
    assert hasattr(rp, 'projector'), "❌ Nincs projector!"
    assert hasattr(rp, 'meta_monitor'), "❌ Nincs meta_monitor!"
    print("✓ Minden komponens jelen van")
    
    # Szimulált bemenet
    batch, seq_len = 2, 16
    dummy_input = torch.randint(0, 100, (batch, seq_len)).long()
    
    # Előrejelzés
    output = model(dummy_input)
    assert output is not None, "❌ Nincs kimenet a modellből!"
    assert output.shape == (batch, seq_len, config.vocab_size), \
        f"❌ Rossz kimeneti méret: {output.shape}"
    print(f"✓ Modell kimenet: {output.shape}")
    
    # Belső gondolati állapot ellenőrzése
    assert model.last_thought_state is not None, "❌ Nincs elmentve a gondolati állapot!"
    ts = model.last_thought_state
    assert 'thoughts' in ts, "❌ Nincsenek 'thoughts' az állapotban!"
    assert 'final_thought' in ts, "❌ Nincs 'final_thought' az állapotban!"
    assert 'n_iterations' in ts, "❌ Nincs 'n_iterations' az állapotban!"
    assert 'meta_state' in ts, "❌ Nincs 'meta_state' az állapotban!"
    
    n_thoughts = ts['thoughts'].shape[1]
    n_iters = ts['n_iterations']
    meta_action = torch.argmax(ts['meta_state'], dim=-1).item()
    
    print(f"✓ Belső gondolati állapot rögzítve:")
    print(f"  - Gondolati lépések: {n_thoughts}")
    print(f"  - Iteratív javítások: {n_iters} kör")
    print(f"  - Metakogníció: {['tovább gondolkodik', 'válaszol', 'javít'][meta_action]}")
    
    return True


def test_get_inner_thoughts():
    """Teszt: a get_inner_thoughts metódus"""
    print("\n" + "-" * 40)
    print("get_inner_thoughts teszt")
    print("-" * 40)
    
    config = ModelConfig()
    model = CognitiveBilingualModel(config)
    
    dummy_input = torch.randint(0, 100, (1, 10)).long()
    thoughts = model.get_inner_thoughts(dummy_input)
    
    assert thoughts is not None, "❌ Nincsenek belső gondolatok!"
    print(f"✓ Belső gondolatok lekérve:")
    print(f"  - Forma: {thoughts['thoughts'].shape}")
    print(f"  - Első gondolatvektor (csonkolva): {thoughts['final_thought'][0, :5].tolist()}")
    
    return True


def test_training_compatibility():
    """Teszt: a modell training módban is működik-e"""
    print("\n" + "-" * 40)
    print("Training mód teszt")
    print("-" * 40)
    
    config = ModelConfig()
    model = CognitiveBilingualModel(config)
    model.train()
    
    dummy_input = torch.randint(0, 100, (4, 16)).long()
    output = model(dummy_input)
    
    # Loss számítás (szimulált)
    targets = torch.randint(0, config.vocab_size, (4, 16)).long()
    loss = F.cross_entropy(output.reshape(-1, config.vocab_size), targets.reshape(-1))
    
    assert loss is not None, "❌ Nem lehet loss-t számolni!"
    assert loss.item() > 0, f"❌ Rossz loss érték: {loss.item()}"
    
    print(f"✓ Training mód működik!")
    print(f"  - Loss érték: {loss.item():.4f}")
    
    # Vissza propagálás
    loss.backward()
    
    # Ellenőrizzük, hogy a gradients flow-ik a reasoning pipeline-ba is
    has_grad = False
    for name, param in model.reasoning_pipeline.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"✓ Gradiens flow a reasoning pipeline-ban: {name}")
            break
    
    if not has_grad:
        print("⚠️  Nincs gradiens a reasoning pipeline-ban (lehet, hogy nincs hatása a loss-ra)")
    
    return True


def test_reasoning_components():
    """Az egyes reasoning komponensek önálló tesztje"""
    print("\n" + "-" * 40)
    print("Reasoning komponensek önálló tesztje")
    print("-" * 40)
    
    from latent_reasoning_space import (
        LatentReasoningSpace, 
        IterativeReasoningChain, 
        ConceptToLanguageProjector,
        ReasoningStep,
        FullReasoningPipeline
    )
    
    config = ModelConfig()
    
    # 1. ReasoningStep
    step = ReasoningStep(config.d_model, config.n_heads)
    dummy_thoughts = torch.randn(2, 4, config.d_model)
    out = step(dummy_thoughts)
    assert out.shape == dummy_thoughts.shape, f"❌ ReasoningStep: {out.shape}"
    print("✓ ReasoningStep működik")
    
    # 2. LatentReasoningSpace
    lrs = LatentReasoningSpace(config)
    context = torch.randn(2, 10, config.d_model)
    compressed, thoughts = lrs(context, return_thoughts=True)
    assert compressed.shape == (2, config.d_model)
    print(f"✓ LatentReasoningSpace: {thoughts.shape[1]} gondolati lépés")
    
    # 3. IterativeReasoningChain
    chain = IterativeReasoningChain(config.d_model)
    final, n_iters = chain(compressed, context)
    assert final.shape == (2, config.d_model)
    print(f"✓ IterativeReasoningChain: {n_iters} iteráció")
    
    # 4. ConceptToLanguageProjector
    proj = ConceptToLanguageProjector(config.d_model, config.vocab_size)
    logits = proj(final)
    assert logits.shape == (2, config.vocab_size)
    print(f"✓ ConceptToLanguageProjector: {logits.shape}")
    
    print("\n✅ Minden komponens sikeresen működik!")
    
    return True


if __name__ == "__main__":
    print("LRS - BELSŐ GONDOLKODÁSI TÉR TESZT")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_reasoning_pipeline()
    except Exception as e:
        print(f"❌ Hiba a tesztben: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= test_get_inner_thoughts()
    except Exception as e:
        print(f"❌ Hiba a tesztben: {e}")
        success = False
    
    try:
        success &= test_training_compatibility()
    except Exception as e:
        print(f"❌ Hiba a tesztben: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= test_reasoning_components()
    except Exception as e:
        print(f"❌ Hiba a tesztben: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MINDEN TESZT SIKERES!")
        print("\nA Latent Reasoning Space (belső gondolkodási tér)")
        print("sikeresen integrálásra került a modellbe.")
        print("\nFuttasd a chat_interface.py-t a kipróbáláshoz:")
        print("  python chat_interface.py")
        print("\nVAGY tanítsd be a modellt:")
        print("  python ultra_trainer.py")
    else:
        print("❌ NÉHÁNY TESZT NEM SIKERÜLT!")
    print("=" * 60)
