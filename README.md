# AICH — Hibrid Transformer-Mamba Architektúra Latent Reasoning Space-szel

**Egy 15-30M paraméteres, lokálisan tanítható hibrid nyelvi modell, amely Transformer figyelmi rétegeket és Mamba (SSM) rétegeket kombinál, kiegészítve egy belső gondolkodási térrel (Latent Reasoning Space).**

## 🧠 Leírás

Az AICH egy kompakt, mégis erőteljes nyelvi modell architektúra, amely:

- **Hibrid felépítés:** 2 Transformer attention réteg + 4 Mamba/SSM réteg
- **Latent Reasoning Space:** Belső gondolkodási nyelv (QTL — Quantum Thought Language), amely lehetővé teszi a modell számára, hogy tokenek előtt "gondolkodjon"
- **~15-30M paraméter** — lokális GPU-n (akár CPU-n is) tanítható
- **Ontologikus motor:** Strukturált tudásreprezentáció és következtetés
- **Magyar és angol nyelvű korpusz** támogatás

### Fő komponensek

| Komponens | Leírás |
|-----------|--------|
| `model.py` | Transformer-Mamba hibrid modell architektúra |
| `layers.py` | Egyedi rétegek (SSM, attention, FFN) |
| `latent_reasoning_space.py` | Belső gondolkodási tér implementáció |
| `qtl_reasoning_space.py` | QTL — Quantum Thought Language |
| `qtl_tokenizer.py` | QTL tokenizáló |
| `concept_compressor.py` | Fogalom tömörítés a gondolkodási térbe |
| `ontological_engine.py` | Ontologikus következtető motor |
| `linguistic_resonator.py` | Nyelvi rezonancia elemző |
| `autonomous_trainer.py` | Autonóm tanító loop |
| `ultra_trainer.py` | Optimalizált tréner |
| `evolution_cycle.py` | Evolúciós ciklus menedzser |
| `config.py` | Modell konfiguráció |

## 📁 Fájlszerkezet

```
AICH/
├── config.py                    # Modell konfiguráció (~15-30M param)
├── model.py                     # Hibrid Transformer-Mamba modell
├── layers.py                    # Egyedi rétegek
├── tokenizer.py                 # Tokenizáló
├── latent_reasoning_space.py    # Latens gondolkodási tér
├── qtl_reasoning_space.py       # QTL implementáció
├── qtl_tokenizer.py             # QTL tokenizáló
├── concept_compressor.py        # Fogalom tömörítő
├── ontological_engine.py        # Ontologikus motor
├── linguistic_resonator.py      # Nyelvi rezonátor
├── autonomous_trainer.py        # Autonóm tréner
├── ultra_trainer.py             # Optimalizált tréner
├── evolution_cycle.py           # Evolúciós ciklus
├── optimizer_cycle.py           # Optimalizáló ciklus
├── live_world_trainer.py        # Élő világ tréner
├── synthetic_data.py            # Szintetikus adat generátor
├── b2_hungarian_corpus.py       # Magyar korpusz (B2 szint)
├── expanded_hungarian_corpus.py # Bővített magyar korpusz
├── logic_instruction_corpus.py  # Logikai instrukciós korpusz
├── logic_tester.py              # Logikai tesztelő
├── chat_interface.py            # Chat interfész
├── final_interface.py           # Végső felhasználói interfész
├── test_lrs.py                  # LRS teszt
├── qtl_benchmark.py             # QTL benchmark
└── deprecated_tests/            # Korábbi tesztek
    ├── benchmark.py
    ├── bbh_test.py
    ├── bbh_validator.py
    ├── diagnostics.py
    ├── goldbach_test.py
    ├── iq_test.py
    ├── paradox_test.py
    ├── research_eval.py
    └── stress_test.py
```

## 🚀 Használat

### Modell betöltése és chat

```bash
python chat_interface.py
```

### Végső interfész

```bash
python final_interface.py
```

### Tanítás

```bash
python autonomous_trainer.py
```

### Tesztelés

```bash
python test_lrs.py
python qtl_benchmark.py
python logic_tester.py
```

### Konfiguráció módosítása

A `config.py` fájlban állítható:
- `d_model` — modell dimenzió (alap: 256)
- `n_layers` — rétegek száma (alap: 6)
- `n_mamba_layers` — Mamba rétegek (alap: 4)
- `n_attention_layers` — Attention rétegek (alap: 2)
- `vocab_size` — szótár méret (alap: 16384)

## 📦 Függőségek

```bash
pip install torch numpy transformers tokenizers
```

- **PyTorch** 2.0+
- **Transformers** (HuggingFace)
- **tokenizers** (HuggingFace)

## 🎯 Célok

- Kompakt, lokálisan futtatható nyelvi modell
- Magyar és angol nyelv támogatása
- Belső gondolkodási térrel jobb következtetési képesség
- Kutatási platform új architektúrák tesztelésére
