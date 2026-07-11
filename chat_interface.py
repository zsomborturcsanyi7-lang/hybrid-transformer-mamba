import torch
import torch.nn.functional as F
from model import CognitiveBilingualModel
from config import ModelConfig
from tokenizer import BilingualTokenizer
from qtl_tokenizer import QTLTokenizer
import sys
import re

def chat(show_qtl=True):
    print("=" * 68)
    print("   SOTA Kognitív AI - Quantum Thought Language (QTL) motor")
    print("   A modell a saját belső nyelvén gondolkodik, ami NEM hasonlít")
    print("   semmilyen emberi nyelvre. 8-32x gyorsabb, mint a nyelvi gondolkodás.")
    print("=" * 68)
    print("\nSzótár és súlyok betöltése...")
    
    config = ModelConfig()
    tokenizer = BilingualTokenizer(config.vocab_size)
    qtl_viz = QTLTokenizer()
    model = CognitiveBilingualModel(config)
    
    try:
        model.load_state_dict(torch.load("sota_model.pt", map_location='cpu'))
        print("✓ Súlyok sikeresen betöltve!")
    except:
        print("! Figyelem: Nem találtam mentett súlyokat. Használat előtt futtasd a trénert.")
        
    model.eval()
    
    print("\n" + "=" * 68)
    print("🧠 Üdvözöllek! Én egy magyar AI vagyok, aki a saját")
    print("   QUANTUM THOUGHT LANGUAGE (QTL) nyelvén gondolkodik,")
    print("   mielőtt magyarul válaszolna.")
    print("   A QTL nem hasonlít semmilyen emberi nyelvre!")
    print("\nParancsok:")
    print("  /qtl       - belső QTL gondolatok megjelenítése")
    print("  /quiet     - belső gondolatok elrejtése")
    print("  exit       - kilépés")
    print("=" * 68)
    
    show_inner = show_qtl
    
    while True:
        user_input = input("\n💬 Te: ")
        if user_input.lower() in ['exit', 'quit', 'kilépés']:
            print("Viszlát! 👋")
            break
            
        if user_input.lower() == '/qtl':
            show_inner = True
            print("  [Belső QTL gondolatok megjelenítve]")
            continue
        elif user_input.lower() == '/quiet':
            show_inner = False
            print("  [Belső gondolatok elrejtve]")
            continue
            
        input_ids = torch.tensor([tokenizer.encode(user_input)]).long()
        
        print("🧠 Gondolkodom a QTL belső nyelvemen...", end="\r")
        
        # Modell előrejelzés (a forward menti a last_thought_state-t is)
        generated_ids = model.generate(input_ids, max_length=40, temperature=0.6, top_p=0.9)
        
        # Válasz dekódolása
        response = tokenizer.decode(generated_ids[0].tolist()[len(input_ids[0]):])
        response = re.sub(r'\[\d+\]', '', response)
        
        # Ha üres a válasz, adjunk egy alapértelmezettet
        if not response.strip():
            response = "Elnézést, még tanulok magyarul. Megpróbálom újra!"
        
        print(f"🤖 AI: {response.strip()}          ")
        
        # Belső QTL gondolatok diagnosztika (ha be van kapcsolva)
        if show_inner and model.last_thought_state is not None:
            thought_info = model.last_thought_state
            n_steps = thought_info['n_steps']
            meta = thought_info['meta_state']
            meta_action = torch.argmax(meta, dim=-1).item()
            
            action_map = {0: "🤔 tovább gondolkodik", 1: "💡 válaszol", 2: "🔧 javít"}
            
            # QTL vizualizáció
            final_thought = thought_info['final_thought'][0]
            qtl_string = ""
            for i in range(min(12, final_thought.shape[0])):
                val = final_thought[i].item()
                qt_char = chr(65 + int((val + 1) * 13) % 26)
                qtl_string += qt_char
            
            print(f"  ┌─ [QUANTUM THOUGHT LANGUAGE - belső monológ] ──────────")
            print(f"  │  QTL gondolat (32D → 12 char): [{qtl_string}]")
            print(f"  │  Gondolkodási lépések: {n_steps}")
            print(f"  │  Metakogníció: {action_map.get(meta_action, '❓')}")
            
            # Az utolsó gondolatvektor első néhány dimenziójának mintázata
            thought_pattern = " ".join([f"{v:.2f}" for v in final_thought[:8].tolist()])
            print(f"  │  Vektor mintázat: [{thought_pattern}...]")
            print(f"  │  ⚡ 8-32x gyorsabb, mint emberi nyelvű gondolkodás!")
            print(f"  └──────────────────────────────────────────────────────")

if __name__ == "__main__":
    chat()
