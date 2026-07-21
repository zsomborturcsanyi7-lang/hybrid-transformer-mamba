# AICH — Hybrid Transformer-Mamba (15-30M param) Latent Reasoning Space-szel

**Status:** ⚠️ Prototype — architektúra definiálva, config kész, training nem futott le

15-30M paraméteres hibrid nyelvi modell, ami Transformer attention rétegeket kombinál Mamba (SSM) rétegekkel, kiegészítve egy belső Latent Reasoning Space-szel (QTL — Quantum Thought Language).

## ⚠️ THIS PROJECT IS UNFINISHED — FEEL FREE TO CONTINUE IT ⚠️

**Ez a projekt NINCS KÉSZEN. Bárki folytathatja, aki akarja!**
Ezt a projektet Zsombi & Hermes Agent (Nous Research) közösen fejlesztette, de egyik projekt sincs 100%-osan befejezve.

---

## Komponensek
| Komponens | Leírás |
|-----------|--------|
| `model.py` | Transformer-Mamba hibrid modell architektúra |
| `layers.py` | Egyedi rétegek (SSM, attention, FFN) |
| `latent_reasoning_space.py` | Belső reasoning space |
| `qtl_reasoning_space.py` | QTL — Quantum Thought Language |
| `qtl_tokenizer.py` | QTL tokenizer |
| `ontological_engine.py` | Ontológiai következtető |
| `autonomous_trainer.py` | Autonóm training loop |
| `ultra_trainer.py` | Optimalizált trainer |

## Fejlesztő
Zsombi & Hermes Agent (Nous Research)
