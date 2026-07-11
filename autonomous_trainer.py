import torch
import torch.nn as nn
from model import CognitiveBilingualModel
from config import ModelConfig
from tokenizer import BilingualTokenizer
import time
import math

class SOTATrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-4, 
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(self, text_batch):
        self.model.train()
        input_ids_list = [self.tokenizer.encode(t, max_length=self.config.max_seq_len) for t in text_batch]
        input_ids = torch.tensor(input_ids_list).to(self.device)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.criterion(logits.view(-1, self.config.vocab_size), targets.view(-1))
        if torch.isnan(loss): return float('nan')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

def start_autonomous_training():
    print("\n--- AUTONÓM MAGYAR B2 + KITERJESZTETT TANÍTÁS ---")
    config = ModelConfig()
    tokenizer = BilingualTokenizer(config.vocab_size)
    model = CognitiveBilingualModel(config)
    model._init_weights()
    
    trainer = SOTATrainer(model, tokenizer, config)
    
    # Korpuszok egyesítése
    from b2_hungarian_corpus import B2_CORPUS
    from expanded_hungarian_corpus import EXPANDED_CORPUS
    training_data = B2_CORPUS + EXPANDED_CORPUS
    
    try:
        for epoch in range(1, 1001):
            loss = trainer.train_step(training_data)
            if math.isnan(loss): break
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Stabil Loss: {loss:.4f}")
            
            if loss < 0.1:
                print(f"\nSOTA ÁTTÖRÉS! Loss: {loss:.4f}")
                break
                
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nMegszakítva.")
        
    torch.save(model.state_dict(), "sota_model.pt")
    print("\n--- MODELL ELMENTVE (sota_model.pt) ---")

if __name__ == "__main__":
    start_autonomous_training()
