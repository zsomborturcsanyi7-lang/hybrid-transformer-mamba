# hybrid-transformer-mamba

Hybrid Transformer-Mamba model implementation and sequence benchmarks.

## Overview & Purpose
hybrid-transformer-mamba explores hybrid model architectures that combine Transformer self-attention blocks with Mamba State Space Model (SSM) layers for memory-efficient long-context modeling.

## Key Features
- Interleaved Transformer-Mamba layer definitions.
- Sequence throughput benchmarks.
- Memory usage benchmarking scripts.

## Tech Stack & Dependencies
- **Framework**: PyTorch
- **Libraries**: mamba-ssm, Transformers
- **Language**: Python 3.10+

## Project Structure
```text
hybrid-transformer-mamba/
├── model.py
├── benchmark.py
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA GPU environment

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/hybrid-transformer-mamba.git
cd hybrid-transformer-mamba
pip install torch mamba-ssm
```

## Usage Examples
```bash
python benchmark.py
```

## Status & License
Status: Experimental Code.
License: MIT
