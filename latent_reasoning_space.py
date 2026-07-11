"""
LATENT REASONING SPACE (LRS) - A modell belső gondolkodási nyelve
===============================================================
Ez a modul egy folyamatos, nem-token-alapú belső reprezentációs teret valósít meg.
A modell itt "gondolkodik" mielőtt bármilyen emberi nyelvre (magyar/angol) lefordítaná a kimenetet.

Főbb komponensek:
1. LatentReasoningSpace - a belső gondolkodási tér magja
2. ReasoningToken - folyamatos "gondolkodási token" (nem diszkrét)
3. ConceptProjector - fogalmi leképezés a belső tér és az ontológiai tér között
4. IterativeRefinement - iteratív gondolkodási ciklus (Chain-of-Thought a belső térben)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReasoningStep(nn.Module):
    """
    Egyetlen "gondolkodási lépés" a belső térben.
    Olyan, mint egy Transformer réteg, de folyamatos reprezentációkon dolgozik,
    nem diszkrét tokeneken.
    
    A modell minden lépésben:
    1. Rezontál a jelenlegi gondolkodási állapottal
    2. Hozzáfér a kontextuális memóriához
    3. Frissíti a belső reprezentációt
    """
    def __init__(self, d_model, n_heads=8, expansion=4):
        super().__init__()
        self.d_model = d_model
        
        # Ön-referenciális figyelem (a belső gondolatok egymásra épülnek)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.05, batch_first=True
        )
        
        # Kereszt-figyelem a kognitív kontextusra (Ontological Engine kimenet)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.05, batch_first=True
        )
        
        # Gondolkodási frissítő (FFN) - itt történik a valódi "kognitív munka"
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(0.05)
        )
        
        # Normalizáció
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Kapuzó mechanizmus: mennyit változzon a gondolat ebben a lépésben
        self.thought_gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, thoughts, context=None, self_mask=None):
        """
        thoughts: [batch, n_thoughts, d_model] - jelenlegi gondolati állapot
        context: [batch, seq_len, d_model] - kognitív kontextus (Ontological Engine kimenet)
        
        Returns: [batch, n_thoughts, d_model] - frissített gondolati állapot
        """
        residual = thoughts
        
        # 1. Önfigyelem a gondolatok között
        attended = self.norm1(thoughts)
        attended, _ = self.self_attention(attended, attended, attended, 
                                          attn_mask=self_mask)
        thoughts = residual + attended * self.thought_gate
        
        # 2. Kereszt-figyelem a kontextusra (ha van)
        if context is not None:
            residual = thoughts
            attended = self.norm2(thoughts)
            attended, _ = self.cross_attention(attended, context, context)
            thoughts = residual + attended * self.thought_gate
        
        # 3. Gondolkodási frissítés (FFN)
        residual = thoughts
        updated = self.norm3(thoughts)
        updated = self.reasoning_ffn(updated)
        thoughts = residual + updated * self.thought_gate
        
        return thoughts


class LatentReasoningSpace(nn.Module):
    """
    A belső gondolkodási tér főmodulja.
    
    A modell itt tart fenn egy "gondolati teret" (thought space),
    ami folyamatos vektorokból áll (nem tokenekből).
    Ebben a térben zajlik a valódi kognitív feldolgozás és érvelés.
    
    A gondolati tér tulajdonságai:
    - Folyamatos: nincsenek diszkrét tokenek
    - Dinamikus számú gondolat: a modell eldönti, hány gondolati lépés kell
    - Nyelvfüggetlen: a reprezentációk nem kötődnek egyetlen emberi nyelvhez sem
    - Rekurzív: a gondolatok egymásra épülhetnek és egymásra hivatkozhatnak
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_reasoning_steps = 4  # hány gondolkodási lépés történjen alapértelmezetten
        
        # Gondolkodási rétegek (szekvenciális reasoning steps)
        self.reasoning_layers = nn.ModuleList([
            ReasoningStep(config.d_model, config.n_heads)
            for _ in range(self.n_reasoning_steps)
        ])
        
        # "Gondolati inicializátor" - hogyan kezdődjön a gondolkodás
        # A bemeneti kontextusból hoz létre kezdeti gondolatokat
        self.thought_initializer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        # "Metakogníciós" kimenet: eldönti, hogy a modell befejezte-e a gondolkodást
        self.termination_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Kompresszió: a gondolati térből készít kontextusvektort a nyelvi kimenethez
        self.compression_head = nn.Sequential(
            nn.Linear(config.d_model * (config.reasoning_max_iterations + 1), config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
    def forward(self, context, return_thoughts=False, max_iterations=8):
        """
        context: [batch, seq_len, d_model] - a bemenet ontológiai reprezentációja
                 (az OntologicalEngine kimenete)
        
        Returns:
            compressed_thought: [batch, d_model] - a gondolkodás kimenete
            thoughts: [batch, n_steps, d_model] - a gondolati lépések (ha return_thoughts)
        """
        batch, seq_len, _ = context.shape
        
        # 1. Kontextus összegzése (átlagolás a szekvencia mentén)
        context_summary = context.mean(dim=1)  # [batch, d_model]
        
        # 2. Kezdeti gondolat inicializálása a kontextusból
        initial_thought = self.thought_initializer(context_summary)
        thoughts = initial_thought.unsqueeze(1)  # [batch, 1, d_model]
        
        # 3. Iteratív gondolkodás
        all_thoughts = [initial_thought]
        
        for step_idx in range(max_iterations):
            step = self.reasoning_layers[step_idx % self.n_reasoning_steps]
            thoughts = step(thoughts, context=context)
            current_thought = thoughts[:, -1, :]  # az utolsó gondolat
            all_thoughts.append(current_thought)
            
            # Metakogníció: eldöntjük, hogy elég volt-e a gondolkodás
            termination_prob = self.termination_head(current_thought)
            
            if termination_prob.mean().item() > 0.8 and step_idx >= self.n_reasoning_steps - 1:
                break
        
        # 4. Összes gondolat összegyűjtése és kompressziója
        stacked_thoughts = torch.stack(all_thoughts, dim=1)  # [batch, n_steps, d_model]
        
        # Kompresszió a végső kognitív állapotba
        batch_size = stacked_thoughts.shape[0]
        n_steps = stacked_thoughts.shape[1]
        flattened = stacked_thoughts.reshape(batch_size, -1)
        
        # Ha kevesebb lépés volt, padding
        max_d = self.d_model * (max_iterations + 1)
        if flattened.shape[-1] < max_d:
            padding = torch.zeros(batch_size, max_d - flattened.shape[-1], 
                                  device=flattened.device)
            flattened = torch.cat([flattened, padding], dim=-1)
        else:
            flattened = flattened[:, :max_d]
        
        compressed_thought = self.compression_head(flattened)
        
        if return_thoughts:
            return compressed_thought, stacked_thoughts
        return compressed_thought
    
    def get_thought_vector(self, context):
        """
        Külső interfész: lekérhető a modell belső gondolati állapota.
        Ez az, ami a "saját nyelvén" való gondolkodást reprezentálja.
        """
        with torch.no_grad():
            return self.forward(context, return_thoughts=True)


class ConceptToLanguageProjector(nn.Module):
    """
    FOGALOM → NYELV LEKÉPEZŐ
    
    Ez a modul alakítja át a belső gondolati teret (folyamatos reprezentáció)
    emberi nyelvi tokenekké (magyar/angol).
    
    A lényeg: a modell ELŐSZÖR gondolkodik a belső terében,
    MAJD ezt a gondolatot "lefordítja" magyar nyelvre.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        
        # A gondolatot token-sorozattá alakítja
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        # "Nyelv választó" - eldönti, hogy a kimenet magyar vagy angol legyen
        # (0 = magyar, 1 = angol)
        self.language_selector = nn.Linear(d_model, 2)
        
        # Hőmérséklet paraméter a kimenethez (kontrollált kreativitás)
        self.temperature = nn.Parameter(torch.ones(1) * 0.7)
        
    def forward(self, thought, return_language=False):
        """
        thought: [batch, d_model] - a belső gondolat
        Returns: [batch, vocab_size] - token logitok
        """
        # Projektálás a szókincsre
        logits = self.projector(thought)
        
        # Nyelv választás
        lang_logits = self.language_selector(thought)
        lang_weights = F.softmax(lang_logits / 0.5, dim=-1)  # alacsony hőmérséklet = határozott választás
        
        # Hőmérséklet skálázás
        logits = logits / self.temperature.clamp(min=0.3, max=1.5)
        
        if return_language:
            return logits, lang_weights
        return logits


class IterativeReasoningChain(nn.Module):
    """
    ITERATÍV GONDOLKODÁSI LÁNC
    
    A modell többször is gondolkodhat, mielőtt válaszolna.
    Minden iterációban:
    1. Megvizsgálja a saját gondolatát
    2. Ellenőrzi, hogy értelmes-e
    3. Javítja, ha szükséges
    4. Dönt, hogy válaszol vagy tovább gondolkodik
    
    Ez az emberi "gondolkodom, mielőtt válaszolok" mechanizmus neurális megfelelője.
    """
    def __init__(self, d_model, max_iterations=5):
        super().__init__()
        self.max_iterations = max_iterations
        
        # Gondolat kiértékelő: mennyire jó a jelenlegi gondolat
        self.thought_evaluator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Gondolat javító: ha nem elég jó, javít rajta
        self.thought_refiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Belső konzisztencia ellenőrző
        self.consistency_check = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, initial_thought, context):
        """
        initial_thought: [batch, d_model] - kezdeti gondolat
        context: [batch, seq_len, d_model] - kontextus
        
        Returns:
            final_thought: [batch, d_model] - végső, javított gondolat
            n_iterations: int - hány iteráció kellett
        """
        current_thought = initial_thought
        
        for i in range(self.max_iterations):
            # 1. Gondolat kiértékelése
            quality = self.thought_evaluator(current_thought)
            
            # 2. Konzisztencia ellenőrzés a kontextussal
            context_summary = context.mean(dim=1)
            consistency = self.consistency_check(
                current_thought * context_summary
            )
            
            # 3. Ha már elég jó, kilépünk
            if quality.mean().item() > 0.85 and consistency.mean().item() > 0.5:
                break
            
            # 4. Különben javítjuk a gondolatot
            combined = torch.cat([current_thought, context_summary], dim=-1)
            refinement = self.thought_refiner(combined)
            current_thought = current_thought + refinement * 0.3  # lassú javítás
        
        return current_thought, i + 1


# ==========================================================
# Integrációs segédfüggvény: a teljes reasoning pipeline
# ==========================================================

class FullReasoningPipeline(nn.Module):
    """
    A TELJES GONDOLKODÁSI PIPELINE
    
    Ez az, ami összeköti a meglévő modellt az új belső gondolkodási térrel.
    
    Adatfolyam:
    1. Bemenet → [Meglévő rétegek] → OntologicalEngine
    2. OntologicalEngine kimenete → LatentReasoningSpace (belső gondolkodás)
    3. Belső gondolat → IterativeReasoningChain (javítás)
    4. Javított gondolat → ConceptToLanguageProjector (magyar nyelvre)
    5. Kimenet: magyar nyelvű tokenek
    
    Használat:
    pipeline = FullReasoningPipeline(config)
    output = pipeline(input_ids)
    """
    def __init__(self, config):
        super().__init__()
        
        # Belső gondolkodási tér
        self.reasoning_space = LatentReasoningSpace(config)
        
        # Iteratív javítás
        self.reasoning_chain = IterativeReasoningChain(config.d_model)
        
        # Fogalom → Magyar nyelv projektor
        self.projector = ConceptToLanguageProjector(
            config.d_model, config.vocab_size
        )
        
        # Metakogníciós monitor: figyeli a gondolkodási folyamatot
        self.meta_monitor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 3)  # [tovább_gondolkodik, válaszol, javít]
        )
        
    def forward(self, ontological_context):
        """
        ontological_context: [batch, seq_len, d_model]
            - az OntologicalEngine kimenete a meglévő modellből
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            thought_state: dict - a belső állapot metadata
        """
        batch, seq_len, d_model = ontological_context.shape
        
        # 1. Belső gondolkodás
        compressed_thought, all_thoughts = self.reasoning_space(
            ontological_context, return_thoughts=True
        )
        
        # 2. Iteratív javítás
        final_thought, n_iters = self.reasoning_chain(
            compressed_thought, ontological_context
        )
        
        # 3. Projektálás magyar nyelvre
        # A gondolatot kiterjesztjük a teljes szekvencia hosszra
        expanded_thought = final_thought.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Keverjük az ontológiai kontextussal és a gondolattal
        combined = ontological_context + expanded_thought * 0.3
        
        # Kimeneti logitok
        logits = self.projector(combined)
        
        # Metakogníciós állapot
        meta_state = self.meta_monitor(final_thought)
        
        return logits, {
            "thoughts": all_thoughts,
            "final_thought": final_thought,
            "n_iterations": n_iters,
            "meta_state": meta_state
        }
    
    def get_inner_monologue(self, ontological_context):
        """
        VISSZAADJA A MODELL BELSŐ GONDOLATAIT (értelmezhető formában)
        
        Ez egy diagnosztikai eszköz: megmutatja, hogy a modell
        "mit gondol", mielőtt magyarul válaszolna.
        
        A gondolatok folyamatos vektorok - ezeket lehetne később
        egy "gondolat-olvasó" rendszerrel értelmezni.
        """
        with torch.no_grad():
            compressed, thoughts = self.reasoning_space(
                ontological_context, return_thoughts=True
            )
            
            return {
                "thought_vectors": thoughts.cpu().numpy(),
                "compressed_thought": compressed.cpu().numpy(),
                "n_thoughts": thoughts.shape[1]
            }


# ==========================================================
# Teszt
# ==========================================================
if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig(d_model=256, vocab_size=16384, n_heads=8)
    pipeline = FullReasoningPipeline(config)
    
    # Szimulált ontológiai kontextus
    batch, seq_len = 2, 10
    context = torch.randn(batch, seq_len, config.d_model)
    
    logits, meta = pipeline(context)
    
    print(f"Belső gondolkodási tér aktiválva!")
    print(f"  Kimenet logitok: {logits.shape}")
    print(f"  Gondolati lépések: {meta['thoughts'].shape}")
    print(f"  Iterációk: {meta['n_iterations']}")
    print(f"  Metakogníciós állapot: {meta['meta_state'].shape}")
    
    # Diagnosztikai mód
    inner = pipeline.get_inner_monologue(context)
    print(f"\nBelső monológ:")
    print(f"  Gondolatvektorok: {inner['thought_vectors'].shape}")
    print(f"  Összesen {inner['n_thoughts']} gondolati lépés")


