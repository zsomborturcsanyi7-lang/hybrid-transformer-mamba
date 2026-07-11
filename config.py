# Stabil Hybrid Transformer-Mamba konfiguráció
# + LATENT REASONING SPACE (belső gondolkodási nyelv)
# ULTRA-LIGHT SOTA Architecture for Local Training

class ModelConfig:
    # Paraméter skálázás: ~15-30M (Nagyon gyors, stabil lokális futtatás)
    d_model = 256                    
    n_layers = 6
    n_mamba_layers = 4
    n_attention_layers = 2
    
    # Attention konfig
    n_heads = 8                     
    head_dim = 32                    
    attn_dropout = 0.1
    
    # Mamba/SSM konfig
    ssm_state_dim = 16               
    ssm_conv_kernel = 4
    ssm_dropout = 0.1
    
    # FFN/MLP konfig
    ffn_expansion = 4                
    ffn_dropout = 0.1
    
    # Normalizáció
    layer_norm_eps = 1e-5
    
    # Tokenizáció
    vocab_size = 16384               
    max_seq_len = 128                
    
    # Training konfig
    batch_size = 8                   
    learning_rate = 1e-3             
    weight_decay = 0.01
    grad_clip = 1.0
    
    # Stabil inicializáció
    init_gain = 0.02

    # *** ÚJ: Latent Reasoning Space konfig ***
    reasoning_steps = 4              # hány gondolkodási lépés alapértelmezetten
    reasoning_max_iterations = 8     # maximum iteráció a gondolkodási térben
    reasoning_termination_threshold = 0.8  # mikor elégedett a modell a gondolattal
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        return f"ModelConfig(d_model={self.d_model}, n_layers={self.n_layers}, max_seq={self.max_seq_len})"
