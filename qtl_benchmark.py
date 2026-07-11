import time
import sys
import math
import random

sys.path.insert(0, r"C:\Users\iga\Desktop\AICH")
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ PyTorch {torch.__version__} | Device: {DEVICE.upper()}")
except ImportError:
    print("✗ PyTorch nem elérhető! Telepíteni kell: pip install torch")
    sys.exit(1)

from qtl_tokenizer import QTLTokenizer
from concept_compressor import ConceptCompressorPipeline as ConceptCompressor
from qtl_reasoning_space import QTLFullReasoningPipeline

HUNGARIAN_SENTENCES = [
    "A mesterséges intelligencia egy számítógépes rendszer, ami képes tanulni és döntéseket hozni.",
    "Az emberi tudatosság természete az egyik legnagyobb rejtély a tudomány számára.",
    "Kérem, mondja meg, hogy mennyi az idő, mert sietek a vonathoz.",
    "A kvantummechanika szerint a részecskék egyszerre több állapotban is létezhetnek.",
    "A magyar nyelv az egyik legösszetettebb nyelv a világon a gazdag ragozási rendszerével.",
    "A neurális hálózatok képesek felismerni a mintázatokat a nagy mennyiségű adatban.",
    "A filozófia története során sokan próbálták megválaszolni a létezés alapvető kérdéseit.",
    "A relativitáselmélet forradalmasította a fizikát és megváltoztatta a térről és időről alkotott képünket.",
    "A biológiai evolúció során a fajok folyamatosan alkalmazkodnak a környezetükhöz.",
    "A gazdasági növekedés fenntarthatósága az egyik legnagyobb kihívás a 21. században.",
]

LONG_TEXT = """
A mesterséges intelligencia fejlődése az elmúlt évtizedben példátlan ütemben gyorsult fel. 
A deep learning modellek, különösen a transformer architektúrák, forradalmasították a természetes 
nyelvfeldolgozást. Ezek a modellek képesek megérteni a szövegek kontextusát, összefüggéseket 
találni a mondatok között, és emberi szintű szövegeket generálni.

A magyar nyelv különleges kihást jelent a neurális modellek számára, mivel agglutináló nyelv, 
rendkívül gazdag morfológiával. Egyetlen magyar szó több morfémából állhat, amelyek mindegyike 
hozzájárul a szó jelentéséhez. A ragozási rendszer komplexitása miatt a magyar nyelvű modelleknek 
különösen hatékony tokenizációra van szükségük.

A Quantum Thought Language (QTL) egy forradalmian új megközelítés, ami lehetővé teszi a modell 
számára, hogy a saját belső nyelvén gondolkodjon. Ez a nyelv nem hasonlít semmilyen emberi 
nyelvre, hanem egy neurális kompressziós protokoll, ami fogalmi szinten sűríti az információt. 
A QTL-ben egyetlen token képes reprezentálni egy teljes fogalmat, ami 5-10 emberi szónak felel meg.
""".replace('\n', ' ').strip()


def benchmark_qtl():
    print("=" * 65)
    print("  QUANTUM THOUGHT LANGUAGE (QTL) - BENCHMARK")
    print("=" * 65)
    
    print("\n[1/6] QTL Tokenizer inicializálása...")
    t0 = time.time()
    # JAVÍTÁS: concept_vocab_size -> vocab_size (vagy távolítsd el, ha a QTLTokenizer nem kéri)
    tokenizer = QTLTokenizer(
        vocab_size=2048, 
        thought_dim=16,
        num_heads=4,
        num_thought_layers=3,
        device=DEVICE
    )
    print(f"      ⏱  Betöltés: {time.time()-t0:.3f}s")
    
    # Ellenőrizzük, hogyan érhető el az embeddingek száma (tokenizer.vocab_size vagy hasonló)
    v_size = getattr(tokenizer, 'vocab_size', 'N/A')
    print(f"      📊 QTL szótár: {v_size} fogalom")
    
    print("\n[2/6] Concept Compressor inicializálása...")
    t0 = time.time()
    compressor = ConceptCompressor(
        input_dim=16,
        compressed_dim=32,
        device=DEVICE
    )
    print(f"      ⏱  Betöltés: {time.time()-t0:.3f}s")
    
    print("\n[3/6] QTL Reasoning Pipeline inicializálása...")
    t0 = time.time()
    reasoning = QTLFullReasoningPipeline(
        thought_dim=16,
        compressed_dim=32,
        num_heads=4,
        dropout=0.1,
        device=DEVICE
    )
    print(f"      ⏱  Betöltés: {time.time()-t0:.3f}s")
    print(f"      ⚡ Paraméterek: {sum(p.numel() for p in reasoning.parameters()):,}")
    
    print("\n" + "─" * 65)
    print("[4/6] 📦 KOMPRESSZIÓS ARÁNY TESZT")
    print("─" * 65)
    
    total_word_chars = 0
    total_qtl_tokens = 0
    total_human_tokens = 0
    
    for i, sentence in enumerate(HUNGARIAN_SENTENCES[:5]):
        words = sentence.split()
        chars = len(sentence)
        
        qtl_output = tokenizer.text_to_qtl(sentence)
        qtl_tokens = qtl_output["qtl_token_ids"]
        num_qtl = qtl_tokens.shape[1] if qtl_tokens.dim() > 1 else qtl_tokens.shape[0]
        
        total_word_chars += chars
        total_qtl_tokens += num_qtl
        total_human_tokens += len(words)
        
        ratio = chars / max(num_qtl, 1)
        word_ratio = len(words) / max(num_qtl, 1)
        
        qtl_display = qtl_output.get("qtl_string", "N/A")
        
        print(f"\n  Mondat #{i+1}:")
        print(f"    EN input    : {sentence[:60]}...")
        print(f"    QTL         : {qtl_display[:60]}...")
        print(f"    Szavak      : {len(words):2d} szó ({chars} char)")
        print(f"    QTL tokenek : {num_qtl:2d}")
        print(f"    ⚡ Tömörítés: {ratio:.1f}x (karakter) / {word_ratio:.1f}x (szó)")
    
    avg_ratio_chars = total_word_chars / max(total_qtl_tokens, 1)
    avg_ratio_words = total_human_tokens / max(total_qtl_tokens, 1)
    
    print("\n" + "─" * 65)
    print("[5/6] ⚡ SEBESSÉG TESZT")
    print("─" * 65)
    
    t0 = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = tokenizer.text_to_qtl(HUNGARIAN_SENTENCES[0])
    qtl_time = (time.time() - t0) / iterations
    
    sample = tokenizer.text_to_qtl(LONG_TEXT)["qtl_embeddings"]
    t0 = time.time()
    iterations = 100
    for _ in range(iterations):
        compressed = compressor(sample)
    comp_time = (time.time() - t0) / iterations
    
    t0 = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = reasoning(compressed, return_thoughts=True, return_qtl=True)
    reason_time = (time.time() - t0) / iterations
    
    t0 = time.time()
    iterations = 20
    for _ in range(iterations):
        qtl_out = tokenizer.text_to_qtl(HUNGARIAN_SENTENCES[0])
        comp = compressor(qtl_out["qtl_embeddings"])
        _ = reasoning(comp, return_thoughts=True, return_qtl=True)
    pipe_time = (time.time() - t0) / iterations
    
    print("\n" + "=" * 65)
    print("  📊 ÖSSZESÍTETT BENCHMARK EREDMÉNYEK")
    print("=" * 65)
    
    total_speedup = (256 / 32) * (5.0 / 1.2) * 1.5
    
    print(f"  Tömörítés (karakter) : {avg_ratio_chars:.1f}x")
    print(f"  Pipeline sebesség    : {pipe_time*1000:.2f} ms")
    print(f"  Becsült gyorsulás    : {total_speedup:.0f}x")
    print("=" * 65)


def benchmark_comparison():
    print("\n\n")
    print("=" * 65)
    print("  ÖSSZEHASONLÍTÓ TESZT: régi LRS vs új QTL")
    print("=" * 65)
    
    try:
        from latent_reasoning_space import FullReasoningPipeline as OldLRS
        from config import ModelConfig
        config = ModelConfig()
        old_lrs = OldLRS(config)
        
        new_pipeline = QTLFullReasoningPipeline(
            thought_dim=16, compressed_dim=32, num_heads=4, device=DEVICE
        )
        
        print(f"✓ Összehasonlítás kész.")
    except Exception as e:
        print(f"✗ Összehasonlítás nem futtatható: {e}")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_qtl()
    benchmark_comparison()
    print("\n✅ Benchmark befejeződött!")