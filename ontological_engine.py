import torch
import torch.nn as nn
import torch.nn.functional as F

class VoidResonator(nn.Module):
    """
    VoidResonator: A kognitív üresség és a strukturált információ közötti hidat képezi.
    Nagy dimenziós szemantikai sűrítést és rezonanciát végez.
    """
    def __init__(self, d_model, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(depth)
        ])
        self.resonance_gate = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # Az x az alap modell rejtett állapota
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual * self.resonance_gate
        return x

class OntologicalEngine(nn.Module):
    """
    Ontological Engine: A rendszer 'értelmi' magja.
    Nem csak szavakat, hanem fogalmi összefüggéseket (ontológiákat) kezel.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.void_resonator = VoidResonator(config.d_model)
        
        # Fogalmi horgonyok (Conceptual Anchors)
        self.anchors = nn.Parameter(torch.randn(1024, config.d_model) * 0.02)
        
    def forward(self, x):
        # Rezonancia a 'void'-dal
        semantic_state = self.void_resonator(x)
        
        # Asszociatív horgonyzás
        # Kiszámítjuk a hasonlóságot a fogalmi horgonyokhoz
        batch, seq, dim = semantic_state.shape
        # semantic_state: [B, S, D], anchors: [1024, D]
        # output: [B, S, 1024]
        similarity = torch.matmul(semantic_state, self.anchors.T)
        weights = F.softmax(similarity, dim=-1)
        
        # Fogalmi terek kombinációja
        conceptual_projection = torch.matmul(weights, self.anchors)
        
        return semantic_state + conceptual_projection
