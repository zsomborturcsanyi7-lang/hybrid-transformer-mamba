import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LogicProver:
    """
    LOGIKAI BIZONYÍTÓ: A BlenderBot-ot 'kényszerítjük' logikai érvelésre
    a válaszok prefixálásával és szigorú dekódolási szabályokkal.
    """
    def __init__(self, model_path="./deep_refined_model"):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        # Betöltjük a finomhangolt modellt
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prove(self, premise):
        # Kényszerített logikai érvelés
        prompt = f"Problem: {premise}. Provide a formal logical proof without politeness. Reasoning:"
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
        
        # Szigorú generálás
        generated = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            do_sample=False, # Determinisztikus, logikus válasz
            temperature=0.1,
            num_beams=3
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    solver = LogicProver()
    # A nehéz teszt: Önhivatkozási paradoxon
    task = "If an AI generates a proof that it can never be wrong, does that proof itself contain a flaw by definition of the Halting Problem?"
    
    print(f"\n[TESZTELÉS: LOGIKAI BIZONYÍTÓ]\nFeladat: {task}\n")
    result = solver.prove(task)
    print(f"AI Válasz: {result}")
