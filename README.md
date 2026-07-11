# AICH — Hybrid Transformer-Mamba Architecture with Latent Reasoning Space

**Status:** ⚠️ Prototype — architecture defined, config ready, training not executed


**A 15–30M parameter, locally trainable hybrid language model that combines Transformer attention layers with Mamba (SSM) layers, augmented with an internal Latent Reasoning Space.**

## 🧠 Description

AICH is a compact yet powerful language model architecture that features:

- **Hybrid design:** 2 Transformer attention layers + 4 Mamba/SSM layers
- **Latent Reasoning Space:** An internal reasoning language (QTL — Quantum Thought Language) that allows the model to "think" before producing tokens
- **~15–30M parameters** — trainable on a local GPU (or even CPU)
- **Ontological engine:** Structured knowledge representation and inference
- **Hungarian and English corpus** support

### Core Components

| Component | Description |
|-----------|-------------|
| `model.py` | Transformer-Mamba hybrid model architecture |
| `layers.py` | Custom layers (SSM, attention, FFN) |
| `latent_reasoning_space.py` | Internal reasoning space implementation |
| `qtl_reasoning_space.py` | QTL — Quantum Thought Language |
| `qtl_tokenizer.py` | QTL tokenizer |
| `concept_compressor.py` | Concept compression into the reasoning space |
| `ontological_engine.py` | Ontological inference engine |
| `linguistic_resonator.py` | Linguistic resonance analyzer |
| `autonomous_trainer.py` | Autonomous training loop |
| `ultra_trainer.py` | Optimized trainer |
| `evolution_cycle.py` | Evolution cycle manager |
| `config.py` | Model configuration |

## 📁 File Structure

```
AICH/
├── config.py                    # Model configuration (~15–30M params)
├── model.py                     # Hybrid Transformer-Mamba model
├── layers.py                    # Custom layers
├── tokenizer.py                 # Tokenizer
├── latent_reasoning_space.py    # Latent reasoning space
├── qtl_reasoning_space.py       # QTL implementation
├── qtl_tokenizer.py             # QTL tokenizer
├── concept_compressor.py        # Concept compressor
├── ontological_engine.py        # Ontological engine
├── linguistic_resonator.py      # Linguistic resonator
├── autonomous_trainer.py        # Autonomous trainer
├── ultra_trainer.py             # Optimized trainer
├── evolution_cycle.py           # Evolution cycle
├── optimizer_cycle.py           # Optimizer cycle
├── live_world_trainer.py        # Live world trainer
├── synthetic_data.py            # Synthetic data generator
├── b2_hungarian_corpus.py       # Hungarian corpus (B2 level)
├── expanded_hungarian_corpus.py # Expanded Hungarian corpus
├── logic_instruction_corpus.py  # Logic instruction corpus
├── logic_tester.py              # Logic tester
├── chat_interface.py            # Chat interface
├── final_interface.py           # Final user interface
├── test_lrs.py                  # LRS test
├── qtl_benchmark.py             # QTL benchmark
└── deprecated_tests/            # Previous tests
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

## 🚀 Usage

### Load model and chat

```bash
python chat_interface.py
```

### Final interface

```bash
python final_interface.py
```

### Training

```bash
python autonomous_trainer.py
```

### Testing

```bash
python test_lrs.py
python qtl_benchmark.py
python logic_tester.py
```

### Modifying configuration

The following can be set in `config.py`:
- `d_model` — model dimension (default: 256)
- `n_layers` — number of layers (default: 6)
- `n_mamba_layers` — Mamba layers (default: 4)
- `n_attention_layers` — Attention layers (default: 2)
- `vocab_size` — vocabulary size (default: 16384)

## 📦 Dependencies

```bash
pip install torch numpy transformers tokenizers
```

- **PyTorch** 2.0+
- **Transformers** (HuggingFace)
- **tokenizers** (HuggingFace)

## 🎯 Goals

- Compact, locally executable language model
- Hungarian and English language support
- Improved reasoning capability via internal reasoning space
- Research platform for testing new architectures

## Author
Zsombi & Hermes Agent (Nous Research)
