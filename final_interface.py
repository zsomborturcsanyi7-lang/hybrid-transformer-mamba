import sys
from chat_interface import ConversationalEngine

def start_interactive_session():
    # Inicializáljuk a kognitív motort
    # A rendszer már tartalmazza a 'deep_refined_model' tudást és a szigorú logikai szűrőket
    try:
        bot = ConversationalEngine()
    except Exception as e:
        print(f"Hiba az inicializáláskor: {e}")
        return

    print("\n" + "="*50)
    print("      KOGNITÍV ENTITÁS AKTÍV (SOTA)")
    print("="*50)
    print("Írd be: 'kilép' a beszélgetés lezárásához.")
    
    while True:
        try:
            user_input = input("\nTe: ")
            if user_input.lower() in ["kilép", "exit", "quit"]:
                print("[Entitás] Beszélgetés lezárva. Offline állapotba lépés...")
                break
            
            if not user_input.strip():
                continue
                
            response = bot.talk(user_input)
            print(f"Entitás: {response}")
            
        except KeyboardInterrupt:
            print("\n[AI] Megszakítás érzékelve. Kilépés...")
            break
        except Exception as e:
            print(f"\n[AI Hiba]: {e}")

if __name__ == "__main__":
    start_interactive_session()
