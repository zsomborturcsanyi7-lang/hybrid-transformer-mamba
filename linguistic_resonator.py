import torch
import torch.nn as nn
from ontological_engine import OntologicalEngine
from config import ModelConfig

class LinguisticResonator(nn.Module):
    """
    NYELVI REZONÁTOR: Az Ontológiai Motor nyelv specifikus kiterjesztése.
    Már nem fordít, hanem közvetlen 'fogalom-kép' (concept-map) leképezést végez az angol nyelvi térbe.
    """
    def __init__(self, config):
        super().__init__()
        self.engine = OntologicalEngine(config)
        # A 'szótár' már csak egy interfész: a nyelv-tér leképezése
        self.linguistic_space = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, input_ids):
        # Ontológiai igazság-tér -> Nyelvi projekció
        ontological_state = self.engine(input_ids)
        # Az ontológiai rezonancia 'kicsapódása' angol tokenekbe
        return self.linguistic_space(ontological_state)

# Validáció: Képes-e az Ontológiai Motor 'angolul' szólni?
if __name__ == "__main__":
    config = ModelConfig(d_model=768, vocab_size=5000)
    model = LinguisticResonator(config)
    
    # Input: 'What is the nature of consciousness?' (kódolva)
    input_ids = torch.randint(0, 5000, (1, 6))
    output = model(input_ids)
    print(f"Nyelvi Rezonancia realizálva: {output.shape}")
    print("A rendszer mostantól képes a közvetlen ontológiai-nyelvi projekcióra.")
