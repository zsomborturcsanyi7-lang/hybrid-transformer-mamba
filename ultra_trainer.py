import torch
import torch.nn as nn
from model import CognitiveBilingualModel
from config import ModelConfig
from tokenizer import BilingualTokenizer
from synthetic_data import get_synthetic_batch
import time
import random

class PureLogicTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        from b2_hungarian_corpus import B2_CORPUS
        from logic_instruction_corpus import INSTRUCTION_CORPUS
        self.base_data = B2_CORPUS + INSTRUCTION_CORPUS

    def train(self, steps=1000):
        print(f"Logikai alapozás indítása: {steps} lépésben.")
        self.model.train()
        for i in range(1, steps + 1):
            # 50% Alap B2 beszéd, 50% Matek/Nyelvtan
            batch = random.sample(self.base_data, k=8)
            batch += get_synthetic_batch(size=24)
            
            input_ids_list = [self.tokenizer.encode(t, max_length=self.config.max_seq_len) for t in batch]
            input_ids = torch.tensor(input_ids_list).to(self.device)
            
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            if i % 100 == 0:
                print(f"Lépés: {i}/{steps} | Loss: {loss:.4f}")
        
        torch.save(self.model.state_dict(), "sota_model.pt")
        self.tokenizer.save_vocab()
        print("\n--- TISZTA LOGIKAI TUDAT LÉTREHOZVA ---")

if __name__ == "__main__":
    config = ModelConfig()
    tokenizer = BilingualTokenizer(config.vocab_size)
    model = CognitiveBilingualModel(config)
    model._init_weights()
    trainer = PureLogicTrainer(model, tokenizer, config)
    trainer.train(steps=1000)
