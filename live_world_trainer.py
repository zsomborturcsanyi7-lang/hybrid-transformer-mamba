import torch
import torch.nn as nn
import feedparser
from model import CognitiveBilingualModel
from config import ModelConfig
from tokenizer import BilingualTokenizer
import time
import re

class LiveWorldTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # SOTA Hírforrások (Magyar + Angol)
        self.rss_feeds = [
            "https://telex.hu/rss",
            "https://24.hu/feed/",
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "https://www.reutersagency.com/feed/"
        ]

    def fetch_live_data(self):
        print("Adatgyűjtés a világhálóról (RSS)...")
        sentences = []
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Cím és leírás tisztítása
                    text = f"{entry.title}. {getattr(entry, 'summary', '')}"
                    text = re.sub(r'<[^>]+>', '', text) # HTML mentesítés
                    sentences.append(text)
            except Exception as e:
                print(f"Hiba az RSS betöltésekor ({url}): {e}")
        return sentences

    def train_online(self, iterations=100):
        print(f"Online tanulás megkezdése: {iterations} lépésben.")
        all_text = self.fetch_live_data()
        if not all_text:
            print("Nem sikerült adatot gyűjteni.")
            return

        self.model.train()
        for i in range(1, iterations + 1):
            # Batch kiválasztása a friss hírekből
            batch_indices = torch.randint(0, len(all_text), (self.config.batch_size,))
            batch_text = [all_text[idx] for idx in batch_indices]
            
            input_ids_list = [self.tokenizer.encode(t, max_length=self.config.max_seq_len) for t in batch_text]
            input_ids = torch.tensor(input_ids_list).to(self.device)
            
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            if i % 10 == 0:
                print(f"Hírek feldolgozása: {i}/{iterations} | Loss: {loss:.4f}")
            
        torch.save(self.model.state_dict(), "sota_model.pt")
        print("Világtudat frissítve, súlyok elmentve.")

if __name__ == "__main__":
    config = ModelConfig()
    # Fontos: A hírek miatt nagyobb vocab kellhet, de maradjunk a stabilitásnál
    tokenizer = BilingualTokenizer(config.vocab_size)
    model = CognitiveBilingualModel(config)
    
    # Próbáljuk betölteni az eddigi tudást
    try:
        model.load_state_dict(torch.load("sota_model.pt"))
        print("Meglévő tudás betöltve, folytatás a hírekkel...")
    except:
        print("Új tudat építése a nulláról...")
        model._init_weights()
        
    trainer = LiveWorldTrainer(model, tokenizer, config)
    trainer.train_online(iterations=100)
