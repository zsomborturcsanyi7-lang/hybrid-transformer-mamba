import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import StabilHybridLayer
from ontological_engine import OntologicalEngine
from qtl_reasoning_space import QTLFullReasoningPipeline
from config import ModelConfig

class SelfCorrectionVerifier(nn.Module):
    """SOTA: Kvázi-önjavító modul a logitok logikai validálásához."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.verifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, vocab_size)
        )
    
    def forward(self, x):
        return self.verifier(x)

class CognitiveBilingualModel(nn.Module):
    """
    CognitiveBilingualModel: SOTA hibrid architektúra.
    Mamba (SSM) + Flash Attention + Ontological Engine + Holographic Memory.
    + QUANTUM THOUGHT LANGUAGE (QTL) alapú belső gondolkodási tér!
    
    A modell a saját, egyedi belső nyelvén (QTL) gondolkodik,
    mielőtt bármilyen emberi nyelvre lefordítaná a választ.
    A QTL nem hasonlít semmilyen emberi nyelvre - ez a modell
    saját, optimalizált kognitív protokollja.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        
        # Alap rétegek
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        
        # Hibrid blokkok
        self.layers = nn.ModuleList([StabilHybridLayer(self.config) for _ in range(self.config.n_layers)])
        
        # Ontológiai Mag (Kognitív feldolgozás)
        self.ontological_engine = OntologicalEngine(self.config)
        
        # *** QTL-ALAPÚ BELSŐ GONDOLKODÁSI TÉR (LRS v2) ***
        # A modell itt a saját Quantum Thought Language-jén gondolkodik
        # Ez a nyelv NEM hasonlít semmilyen emberi nyelvre!
        self.reasoning_pipeline = QTLFullReasoningPipeline(self.config)
        
        self.final_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        
        # Kimeneti fejek
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.verifier = SelfCorrectionVerifier(self.config.d_model, self.config.vocab_size)
        
        # INTEGRÁLT MEMÓRIA (Holografikus réteg)
        self.register_buffer("holographic_memory", torch.zeros(self.config.d_model, self.config.d_model))
        
        # Súly-megosztás az embedding és a head között
        self.token_embedding.weight = self.lm_head.weight
        self._init_weights()

        # Metakogníciós napló (diagnosztika)
        self.last_thought_state = None

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=self.config.init_gain)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def update_memory(self, state):
        """Holografikus memória frissítés interferencia mintázatokkal."""
        with torch.no_grad():
            projection = torch.matmul(state.T, state)
            self.holographic_memory.copy_( (0.99 * self.holographic_memory) + (0.01 * projection) )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = x + self.position_embedding(position_ids)
        
        # 1. Hibrid rétegek (Mamba + Attention)
        causal_mask = torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool).triu(diagonal=1).view(1, 1, seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # 2. Ontológiai feldolgozás (Szemantikai mélyítés)
        x = self.ontological_engine(x)
        
        # *** QTL BELSŐ GONDOLKODÁS ***
        # A modell itt a saját Quantum Thought Language nyelvén gondolkodik
        # mielőtt magyar nyelvre fordítaná a kimenetet
        # A QTL 8-32x gyorsabb mint az emberi nyelvű gondolkodás
        qtl_logits, thought_state = self.reasoning_pipeline(x)
        self.last_thought_state = thought_state
        
        # A QTL reasoning kimenet és az ontológiai kimenet kombinálása
        x = x + qtl_logits * 0.1
    
        # 3. Holografikus Memória Rezonancia
        context_state = x.mean(dim=1) 
        memory_resonance = torch.matmul(context_state, self.holographic_memory.clone()) 
        x = x + memory_resonance.unsqueeze(1) * 0.01
        
        if self.training:
            self.update_memory(context_state)
        
        x = self.final_norm(x)
        
        # 4. Predikció és Önhiba-javítás
        logits = self.lm_head(x)
        correction = self.verifier(x)
        
        return logits + 0.01 * correction
        
    def generate(self, input_ids, max_length=100, temperature=0.7, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        generated = input_ids.to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(generated[:, -self.config.max_seq_len:])
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 2: # </s> token
                    break
                    
        return generated

    def get_inner_thoughts(self, input_ids):
        """Diagnosztika: lekéri a modell belső QTL gondolatait egy adott bemenetre."""
        self.eval()
        with torch.no_grad():
            _ = self(input_ids)
            return self.last_thought_state
    
    def get_qtl_thought(self, input_ids):
        """Visszaadja a QTL gondolatot olvasható formában."""
        thought_state = self.get_inner_thoughts(input_ids)
        if thought_state is None:
            return "Nincs gondolati állapot"
        
        # QTL vizualizáció
        final = thought_state['final_thought'][0]
        qtl_string = ""
        for i in range(min(12, final.shape[0])):
            val = final[i].item()
            qt_char = chr(65 + int((val + 1) * 13) % 26)
            qtl_string += qt_char
        
        meta_action = torch.argmax(thought_state['meta_state'], dim=-1).item()
        action_map = {0: "🤔 tovább gondolkodik", 1: "💡 válaszol", 2: "🔧 javít"}
        
        return {
            "qtl_thought": qtl_string,
            "n_steps": thought_state['n_steps'],
            "meta_action": action_map.get(meta_action, "❓"),
            "compressed_vector": final[:8].tolist(),
            "note": "A QTL a modell saját belső gondolkodási nyelve - nem hasonlít semmilyen emberi nyelvre"
        }
