import re
import json
import os
from typing import List, Dict

class BilingualTokenizer:
    """SOTA Tokenizer v3: Karakter- és Számjegy-tudatos kezelés."""
    def __init__(self, vocab_size: int = 16384, vocab_path: str = "vocab.json"):
        self.vocab_size = vocab_size
        self.vocab_path = vocab_path
        self.id_to_word = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        self.word_to_id = {v: k for k, v in self.id_to_word.items()}
        self.next_id = 100 # Hagyjunk helyet a speciális karaktereknek
        
        # Alapvető karakterek és számjegyek regisztrálása (0-9, +, -, =, ?, !, ., ,)
        chars = "0123456789+-=?!. ,:;()áéíóöőúüű"
        for c in chars:
            self._register_token(c)
            
        self.load_vocab()

    def _register_token(self, token: str):
        if token not in self.word_to_id and self.next_id < self.vocab_size:
            self.word_to_id[token] = self.next_id
            self.id_to_word[self.next_id] = token
            self.next_id += 1

    def save_vocab(self):
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump({"word_to_id": self.word_to_id, "next_id": self.next_id}, f, ensure_ascii=False)

    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.word_to_id = data["word_to_id"]
                self.next_id = data["next_id"]
                self.id_to_word = {int(k): v for v, k in self.word_to_id.items()}

    def encode(self, text: str, max_length: int = None) -> List[int]:
        if not text: return [0]
        text = text.lower()
        
        # Speciális darabolás: Számjegyekre, írásjelekre és szavakra
        # Ez a SOTA módszer a logikai feladatokhoz
        tokens = re.findall(r'\d|[a-záéíóöőúüű]+|[^\w\s]', text, re.UNICODE)
        
        ids = [1]
        for token in tokens:
            if token not in self.word_to_id:
                self._register_token(token)
            ids.append(self.word_to_id.get(token, 3))
        ids.append(2)
        
        if max_length:
            ids = ids[:max_length] + [0] * max(0, max_length - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            i_int = int(i)
            if i_int in [0, 1, 2]: continue
            word = self.id_to_word.get(i_int, "")
            # Számjegyeknél és írásjeleknél ne tegyünk szóközt
            if word and (word.isdigit() or len(word) == 1):
                tokens.append(word)
            else:
                tokens.append(" " + word)
        return "".join(tokens).strip()

def get_tokenizer(config):
    return BilingualTokenizer(config.vocab_size)
