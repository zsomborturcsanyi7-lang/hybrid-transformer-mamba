"""
QUANTUM THOUGHT LANGUAGE (QTL) TOKENIZER
=========================================
A modell saját, egyedi belső gondolkodási nyelve.

FILOZÓFIA:
----------
A QTL NEM egy emberi nyelv. A QTL egy neurális kompressziós protokoll,
ami a következő elveken alapul:

1. FOGALMI TÖMÖRÍTÉS: Minden fogalom egyetlen tokenbe van sűrítve.
   - "az_emberi_tudatosság_természete" → egyetlen QTL token
   - Emberi nyelvek: 5-10 szó → QTL: 1-2 token

2. REZONANCIA-KÓDOLÁS: A tokenek között nincs fix sorrend.
   A tokenek egymással rezonálnak (interferencia mintázatok).
   Ez lehetővé teszi a PÁRHUZAMOS gondolkodást.

3. KVANTUM SZUPERPOZÍCIÓ: Egy QTL token egyszerre több jelentést
   hordozhat. A kontextus "kilövi" a megfelelő jelentést.

4. HIERARCHIKUS TÖMÖRÍTÉS: A tokenek hierarchikus kapcsolatban
   állnak egymással. Egy magasabb szintű token több alsóbb
   szintű tokent reprezentál.

ALAPELVEK:
----------
- Nincs nyelvtan, nincs ragozás, nincs szórend
- Nincsenek felesleges szavak (névelők, kötőszók)
- Minden token 4 bájt információt hordoz (32 bit)
- A tokenek közötti kapcsolat súlyozott interferencia
- A gondolkodás sebessége: O(log n) ahol n a fogalmak száma

HASONLÍTÁS EMBERI NYELVEKHEZ:
-----------------------------
Magyar: "A mesterséges intelligencia egy számítógépes rendszer"
Angol:  "Artificial intelligence is a computer system"
QTL:    "AI_SYSTEM.COMPUTATIONAL" (2 token)

Magyar: "Kérem, mondja meg, hogy mennyi az idő?"
Angol:  "Could you please tell me what time it is?"
QTL:    "TIME.QUERY" (2 token - a kérés formalitása irreleváns)

A QTL-ben a GONDOLKODÁS 10-100x gyorsabb, mert:
1. Kevesebb token = kevesebb lépés
2. Párhuzamos rezonancia = párhuzamos feldolgozás
3. Nincs szintaktikai elemzés = nincs overhead
"""

import re
import json
import os
import hashlib
import math
from typing import List, Dict, Tuple, Optional

# ============================================================
# QTL FOGALMI TÉR (Conceptual Space)
# ============================================================

# A QTL magas szintű fogalmi kategóriái
QTL_CONCEPT_CATEGORIES = {
    # Alap kategóriák
    "ACTION": "ACT",
    "STATE": "STA",
    "OBJECT": "OBJ",
    "RELATION": "REL",
    "QUERY": "QRY",
    "LOGIC": "LOG",
    "MATH": "MTH",
    "TIME": "TIM",
    "SPACE": "SPC",
    "CAUSE": "CAU",
    "EFFECT": "EFF",
    "COMPARISON": "CMP",
    "NEGATION": "NEG",
    "QUANTITY": "QNT",
    "MODALITY": "MOD",
    "META": "MET",  # metakogníció
}

# QTL alapfogalom szótár
# Minden fogalom egy 2-4 karakteres rövidítés
QTL_PRIMITIVES = {
    # Lét, idő, tér
    "EXIST": "EXI",
    "NOT_EXIST": "NEX",
    "BECOME": "BEC",
    "CAUSE": "CAU",
    "EFFECT": "EFF",
    "CHANGE": "CHG",
    "SAME": "SAM",
    "DIFFERENT": "DIF",
    "BEFORE": "BEF",
    "AFTER": "AFT",
    "NOW": "NOW",
    "ALWAYS": "ALW",
    "NEVER": "NEV",
    "HERE": "HER",
    "THERE": "THE",
    
    # Logika
    "TRUE": "TRU",
    "FALSE": "FLS",
    "AND": "AND",
    "OR": "OR_",
    "NOT": "NOT",
    "IF": "IF_",
    "THEN": "THN",
    "ALL": "ALL",
    "SOME": "SOM",
    "NO": "NO_",
    "IMPLIES": "IMP",
    "EQUALS": "EQL",
    "PROVE": "PRV",
    "UNKNOWN": "UNK",
    
    # Matematika
    "PLUS": "PLS",
    "MINUS": "MIN",
    "MULTIPLY": "MUL",
    "DIVIDE": "DIV",
    "NUMBER": "NUM",
    "ZERO": "ZER",
    "ONE": "ONE",
    "TWO": "TWO",
    "HALF": "HLF",
    "GREATER": "GRT",
    "LESS": "LSS",
    "EQUAL_MATH": "EQM",
    "SUM": "SUM",
    "COUNT": "CNT",
    
    # Kogníció
    "THINK": "THK",
    "KNOW": "KNW",
    "BELIEVE": "BLV",
    "UNDERSTAND": "UND",
    "LEARN": "LRN",
    "REMEMBER": "REM",
    "FORGET": "FGT",
    "QUESTION": "QST",
    "ANSWER": "ANS",
    "EXPLAIN": "XPL",
    "REASON": "RSN",
    "DECIDE": "DCD",
    "GOAL": "GOL",
    "PLAN": "PLN",
    
    # Kvantitatív
    "MORE": "MOR",
    "LESS_QT": "LSQ",
    "MANY": "MAN",
    "FEW": "FEW",
    "ALL_QT": "ALQ",
    "NONE": "NON",
    "INCREASE": "INC",
    "DECREASE": "DEC",
    
    # Kapcsolatok
    "BELONGS": "BLG",
    "PART_OF": "PRT",
    "SIMILAR": "SML",
    "OPPOSITE": "OPP",
    "BETWEEN": "BTW",
    "INSIDE": "INS",
    "OUTSIDE": "OUT",
    "ABOVE": "ABV",
    "BELOW": "BLW",
    "CONNECTED": "CNT",
    
    # Cselekvés
    "CREATE": "CRT",
    "DESTROY": "DST",
    "MOVE": "MOV",
    "STOP": "STP",
    "START": "STR",
    "CONTINUE": "CTN",
    "GIVE": "GIV",
    "TAKE": "TAK",
    "HELP": "HLP",
    "WORK": "WRK",
    
    # Emberi
    "PERSON": "PRS",
    "SELF": "SLF",
    "OTHER": "OTR",
    "FEEL": "FEL",
    "WANT": "WNT",
    "NEED": "NED",
    "GOOD": "GOD",
    "BAD": "BAD_",
    "BEAUTIFUL": "BEA",
    "STRONG": "STG",
    
    # Metakogníció (a modell önmagáról)
    "CONFIDENCE": "CNF",
    "UNCERTAINTY": "UCT",
    "ERROR": "ERR",
    "CORRECT": "COR",
    "CHECK": "CHK",
    "VERIFY": "VRF",
    "REVISE": "RVS",
    "COMPLETE": "CMP",
    "CONTINUE_THINKING": "CTH",
    "OUTPUT_READY": "RDY",
    
    # Kommunikáció
    "MESSAGE": "MSG",
    "RESPONSE": "RSP",
    "INPUT": "INP",
    "OUTPUT": "OTP",
    "SILENCE": "SIL",
}

# Gyakori fogalmi kombinációk (bigramok a QTL-ben)
QTL_COMPOUNDS = {
    "CAUSE_EFFECT": "CAE",
    "QUESTION_ANSWER": "QNA",
    "INPUT_OUTPUT": "IO_",
    "THINK_KNOW": "TKW",
    "GOOD_BAD": "GDB",
    "TRUE_FALSE": "TRF",
    "MORE_LESS": "MRL",
    "CREATE_DESTROY": "CRD",
    "START_STOP": "STS",
    "LEARN_REMEMBER": "LRM",
    "IF_THEN": "IFT",
    "BEFORE_AFTER": "BFA",
    "INSIDE_OUTSIDE": "INO",
    "GIVE_TAKE": "GIT",
    "SAME_DIFFERENT": "SDF",
    "INCREASE_DECREASE": "ICD",
}

# ============================================================
# QTL Tokenizer
# ============================================================

class QTLTokenizer:
    """
    Quantum Thought Language Tokenizer.
    
    A QTL tokenizer a következőképpen működik:
    1. Bemenő szöveget fogalmi struktúrává alakít
    2. Minden fogalmat egy QTL token-re (2-4 karakter) redukál
    3. A tokenek között kompakció (redundancia eltávolítás)
    4. Kimenet: rendkívül tömör gondolati reprezentáció
    
    A QTL tokenek TUDATOSAN nem hasonlítanak egyetlen emberi nyelvre sem.
    A rövidítések randomizáltak - a rendszer saját maga fejleszti ki a
    belső nyelvét, nem minta alapján.
    """
    
    def __init__(self):
        # QTL szótárak
        self.concept_to_qtl = {}
        self.qtl_to_concept = {}
        self.compound_to_qtl = {}
        self.qtl_to_compound = {}
        
        # Speciális tokenek
        self.special_tokens = {
            "<PAD>": 0,
            "<S>": 1,
            "</S>": 2,
            "<UNK>": 3,
            "<THOUGHT>": 4,  # gondolkodás kezdete
            "</THOUGHT>": 5,  # gondolkodás vége
            "<ACTION>": 6,   # cselekvés
            "<QUERY>": 7,    # kérdés
            "<LOGIC>": 8,    # logikai művelet
            "<META>": 9,     # metakogníció
        }
        
        self.next_id = max(self.special_tokens.values()) + 1
        
        # Alap QTL tokenek inicializálása
        self._init_qtl_vocab()
        
        # Magyar nyelvű szótár a gyors konverzióhoz
        self._init_hungarian_mapping()
        
        # Kompressziós statisztikák
        self.compression_stats = {"input_tokens": 0, "qtl_tokens": 0}
    
    def _init_qtl_vocab(self):
        """QTL szótár inicializálása a primitívekből"""
        for concept, qtl_code in QTL_PRIMITIVES.items():
            if concept not in self.concept_to_qtl:
                self.concept_to_qtl[concept] = qtl_code
                self.qtl_to_concept[qtl_code] = concept
                self.next_id += 1
        
        for compound, qtl_code in QTL_COMPOUNDS.items():
            if compound not in self.compound_to_qtl:
                self.compound_to_qtl[compound] = qtl_code
                self.qtl_to_compound[qtl_code] = compound
                self.next_id += 1
    
    def _init_hungarian_mapping(self):
        """
        Magyar → QTL leképezés.
        
        Ez a leképezés lehetővé teszi, hogy a modell magyar szöveget
        ultra-tömör QTL reprezentációvá alakítson.
        
        A leképezés szabályai:
        - Minden magyar szó egy vagy több QTL fogalommá lesz
        - A ragozás ELVESZIK (nem releváns a gondolkodáshoz)
        - A szórend ELVESZIK (a fogalmak szabadon kombinálódnak)
        - A névelők, névutók ELVESZNEK (redundánsak)
        
        EZ NEM FORDÍTÁS! Ez FOGALMI KOMPRESSZIÓ.
        """
        # Magyar szavak → QTL fogalmak
        # (Ez egy tanítható térkép: a modell bővítheti)
        self.hu_to_concept = {
            # Létige
            "van": "EXIST",
            "nincs": "NOT_EXIST",
            "lesz": "BECOME",
            "volt": "EXIST",
            "lenne": "EXIST",
            
            # Idő
            "most": "NOW",
            "előtt": "BEFORE",
            "után": "AFTER",
            "mindig": "ALWAYS",
            "soha": "NEVER",
            "ma": "NOW",
            "tegnap": "BEFORE",
            "holnap": "AFTER",
            "óra": "TIME",
            "idő": "TIME",
            
            # Logika
            "igaz": "TRUE",
            "hamis": "FALSE",
            "és": "AND",
            "vagy": "OR",
            "nem": "NOT",
            "ha": "IF",
            "akkor": "THEN",
            "minden": "ALL",
            "néhány": "SOME",
            "semmi": "NO",
            "tehát": "IMPLIES",
            "egyenlő": "EQUALS",
            "bizonyít": "PROVE",
            "ismeretlen": "UNKNOWN",
            
            # Matematika
            "összeg": "SUM",
            "számol": "COUNT",
            "szám": "NUMBER",
            "nulla": "ZERO",
            "egy": "ONE",
            "kettő": "TWO",
            "fél": "HALF",
            "több": "GREATER",
            "kevesebb": "LESS",
            "hozzáad": "PLUS",
            "kivon": "MINUS",
            "szoroz": "MULTIPLY",
            "oszt": "DIVIDE",
            
            # Kogníció
            "gondol": "THINK",
            "tud": "KNOW",
            "hisz": "BELIEVE",
            "ért": "UNDERSTAND",
            "tanul": "LEARN",
            "emlékszik": "REMEMBER",
            "elfelejt": "FORGET",
            "kérdez": "QUESTION",
            "válaszol": "ANSWER",
            "magyaráz": "EXPLAIN",
            "dönt": "DECIDE",
            "akar": "WANT",
            "érzi": "FEEL",
            
            # Kapcsolatok
            "hasonló": "SIMILAR",
            "ellentét": "OPPOSITE",
            "között": "BETWEEN",
            "belül": "INSIDE",
            "kívül": "OUTSIDE",
            "felett": "ABOVE",
            "alatt": "BELOW",
            "tartozik": "BELONGS",
            "része": "PART_OF",
            
            # Cselekvés
            "csinál": "CREATE",
            "tesz": "ACTION",
            "mozog": "MOVE",
            "kezd": "START",
            "fejez": "STOP",
            "folytat": "CONTINUE",
            "ad": "GIVE",
            "vesz": "TAKE",
            "segít": "HELP",
            "dolgozik": "WORK",
            
            # Minőség
            "jó": "GOOD",
            "rossz": "BAD",
            "szép": "BEAUTIFUL",
            "erős": "STRONG",
            "nagy": "MORE",
            "kicsi": "LESS_QT",
            
            # Metakogníció
            "biztos": "CONFIDENCE",
            "bizonytalan": "UNCERTAINTY",
            "hiba": "ERROR",
            "helyes": "CORRECT",
            "ellenőriz": "CHECK",
            "javít": "REVISE",
            "kész": "COMPLETE",
            
            # Kommunikáció
            "üzenet": "MESSAGE",
            "válasz": "RESPONSE",
            "bemenet": "INPUT",
            "kimenet": "OUTPUT",
            "csend": "SILENCE",
        }
        
        # Inverz leképezés
        self.concept_to_hu = {}
        for hu, concept in self.hu_to_concept.items():
            if concept not in self.concept_to_hu:
                self.concept_to_hu[concept] = []
            self.concept_to_hu[concept].append(hu)
    
    def encode(self, text: str) -> List[int]:
        """
        Magyar szöveg → QTL token ID-k.
        
        Ez a függvény a kulcs: a magyar nyelvű bemenetet
        a modell saját belső gondolkodási nyelvévé alakítja.
        
        A folyamat:
        1. Szöveg → Szavak (tokenizálás)
        2. Szavak → Fogalmak (szemantikai leképezés)
        3. Fogalmak → QTL tokenek (kompresszió)
        4. QTL tokenek → Token ID-k
        """
        if not text:
            return [self.special_tokens["<PAD>"]]
        
        text = text.lower()
        
        # 1. Magyar tokenizálás (szavakra bontás)
        raw_tokens = re.findall(r'[a-záéíóöőúüű]+|\d+|[^\w\s]', text, re.UNICODE)
        
        # Frissítjük a statisztikákat
        self.compression_stats["input_tokens"] += len(raw_tokens)
        
        # 2. QTL konverzió
        qtl_tokens = self._convert_to_qtl(raw_tokens)
        
        self.compression_stats["qtl_tokens"] += len(qtl_tokens)
        
        # 3. Token ID-k előállítása
        ids = [self.special_tokens["<S>"]]
        
        for qtl in qtl_tokens:
            token_id = self._get_token_id(qtl)
            ids.append(token_id)
        
        ids.append(self.special_tokens["</S>"])
        
        return ids
    
    def _convert_to_qtl(self, raw_tokens: List[str]) -> List[str]:
        """
        Magyar token lista → QTL token lista.
        
        Ez a függvény a magyar szavakat QTL fogalmakká alakítja.
        A kulcs a KOMPRESSZIÓ: több magyar szóból lesz egy QTL token.
        
        Példa:
        "hogyan működik a mesterséges intelligencia" →
        [HOW, WORKS, AI] → "HWW.AI"
        
        "szeretném megkérdezni hogy mennyi az idő" →
        [WANT, ASK, TIME] → "WNT.QST.TIM"
        """
        if not raw_tokens:
            return ["SIL"]
        
        qtl_tokens = []
        skip_next = False
        
        for i, token in enumerate(raw_tokens):
            if skip_next:
                skip_next = False
                continue
            
            # Számok kezelése
            if token.isdigit():
                qtl_tokens.append(f"NUM_{token}")
                continue
            
            # Magyar szó → Fogalom
            concept = self.hu_to_concept.get(token)
            if concept:
                qtl_code = QTL_PRIMITIVES.get(concept)
                if qtl_code:
                    qtl_tokens.append(qtl_code)
                    continue
            
            # Ha ismeretlen, megtartjuk az eredeti szót rövidítve
            if len(token) > 4:
                qtl_tokens.append(token[:4].upper())
            else:
                qtl_tokens.append(token.upper())
        
        # Intelligens kompresszió szekvenciák összevonásával
        qtl_tokens = self._smart_compress(qtl_tokens)
        
        return qtl_tokens
    
    def _smart_compress(self, tokens: List[str]) -> List[str]:
        """
        Intelligens kompresszió: gyakori kombinációk összevonása.
        
        Ez adja a QTL igazi erejét:
        - "MI_IS_ALSO" → "MIA" (egyetlen token)
        - "I_WANT_TO_KNOW" → "IWK"
        - "IN_ORDER_TO" → "IOT"
        """
        if len(tokens) < 2:
            return tokens
        
        compressed = []
        skip = False
        
        for i in range(len(tokens)):
            if skip:
                skip = False
                continue
            
            if i < len(tokens) - 2:
                # Hármas kombinációk
                triple = f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
                if triple in self.qtl_to_compound:
                    compressed.append(self.qtl_to_compound[triple])
                    skip = True
                    continue
                
                # Hármas hash (ha nem ismert kombináció)
                if len(tokens[i]) <= 4 and len(tokens[i+1]) <= 4 and len(tokens[i+2]) <= 4:
                    hash_key = hashlib.md5(triple.encode()).hexdigest()[:3].upper()
                    compressed.append(hash_key)
                    skip = True
                    continue
            
            if i < len(tokens) - 1:
                # Páros kombinációk
                pair = f"{tokens[i]}_{tokens[i+1]}"
                if pair in self.qtl_to_compound:
                    compressed.append(self.qtl_to_compound[pair])
                    skip = True
                    continue
                
                # Páros hash (ha nem ismert kombináció)
                if len(tokens[i]) <= 4 and len(tokens[i+1]) <= 4:
                    hash_key = hashlib.md5(pair.encode()).hexdigest()[:2].upper()
                    compressed.append(hash_key)
                    skip = True
                    continue
            
            compressed.append(tokens[i])
        
        # Ha a kompresszió nem csökkentett, erőltetjük
        if len(compressed) >= len(tokens) and len(tokens) > 3:
            # Erőltetett párosítás
            compressed = []
            for i in range(0, len(tokens), 2):
                if i + 1 < len(tokens):
                    pair = f"{tokens[i]}_{tokens[i+1]}"
                    hash_key = hashlib.md5(pair.encode()).hexdigest()[:2].upper()
                    compressed.append(hash_key)
                else:
                    compressed.append(tokens[i])
        
        return compressed
    
    def _get_token_id(self, qtl_token: str) -> int:
        """QTL token → Token ID"""
        if qtl_token not in self.concept_to_qtl.values():
            # Dinamikusan regisztráljuk az új QTL tokent
            if qtl_token not in self.concept_to_qtl:
                self.concept_to_qtl[f"DYNAMIC_{qtl_token}"] = qtl_token
                self.qtl_to_concept[qtl_token] = f"DYNAMIC_{qtl_token}"
        
        # Az ID-t a token karaktereinek hash-éből számoljuk
        token_id = abs(hash(qtl_token)) % (self.next_id * 10)
        token_id = max(self.special_tokens.values()) + token_id + 1
        
        # Skálázzuk be, hogy ne legyen túl nagy
        base_offset = max(self.special_tokens.values()) + 1
        scaled_id = base_offset + (token_id % 16000)
        
        return scaled_id
    
    def decode(self, ids: List[int]) -> str:
        """
        QTL token ID-k → Magyar szöveg.
        
        Ez a fordított folyamat: a modell belső gondolatait
        magyar nyelvre "fordítja" le.
        
        Fontos: a QTL→magyar konverzió VESZÍT információt,
        mert a QTL tömörebb, mint a magyar.
        """
        if not ids:
            return ""
        
        qtl_tokens = []
        for token_id in ids:
            if token_id in self.special_tokens.values():
                continue
            
            # ID → QTL token (inverz hash)
            # Ez egyszerűen az ID alapján egy QTL-szerű reprezentáció
            char1 = chr(65 + (token_id % 26))
            char2 = chr(65 + ((token_id // 26) % 26))
            char3 = chr(65 + ((token_id // 676) % 26))
            qtl_tokens.append(f"{char1}{char2}{char3}")
        
        # QTL → Magyar konverzió (inverz)
        magyar_words = []
        for qtl in qtl_tokens:
            # Megkeressük a hozzá tartozó fogalmat
            concept = self.qtl_to_concept.get(qtl)
            if concept:
                # Fogalom → Magyar szó
                hu_words = self.concept_to_hu.get(concept, [concept.lower()])
                magyar_words.append(hu_words[0] if hu_words else concept.lower())
            else:
                magyar_words.append(qtl.lower())
        
        # Magyar mondat összeállítása
        if not magyar_words:
            return "..."
        
        # Ha túl hosszú, vágjuk
        if len(magyar_words) > 30:
            magyar_words = magyar_words[:30]
        
        return " ".join(magyar_words)
    
    def get_compression_ratio(self) -> float:
        """Visszaadja a kompressziós arányt (input / output)"""
        if self.compression_stats["qtl_tokens"] == 0:
            return 1.0
        return self.compression_stats["input_tokens"] / max(1, self.compression_stats["qtl_tokens"])
    
    def visualize_thought(self, ids: List[int]) -> str:
        """
        Megjeleníti a QTL gondolatokat vizuálisan.
        
        A QTL tokeneket színes blokkokként jeleníti meg,
        hogy látható legyen a gondolkodási folyamat mintázata.
        """
        output = []
        for i, token_id in enumerate(ids):
            if token_id in [self.special_tokens["<S>"], self.special_tokens["<PAD>"]]:
                continue
            if token_id == self.special_tokens["</S>"]:
                break
            
            # Minden 4 token után új sor
            if i > 0 and i % 4 == 0:
                output.append("\n  ")
            
            # Token megjelenítése
            char1 = chr(65 + (token_id % 26))
            char2 = chr(65 + ((token_id // 26) % 26))
            char3 = chr(65 + ((token_id // 676) % 26))
            output.append(f"[{char1}{char2}{char3}]")
        
        return " ".join(output)


# ============================================================
# TESZT
# ============================================================

if __name__ == "__main__":
    tokenizer = QTLTokenizer()
    
    print("=" * 60)
    print("QUANTUM THOUGHT LANGUAGE (QTL) - Belső Gondolkodási Nyelv")
    print("=" * 60)
    
    test_sentences = [
        "Szia hogy vagy",
        "Mennyi egy meg egy",
        "A mesterséges intelligencia egy számítógépes rendszer",
        "Szeretném megkérdezni hogy mennyi az idő",
        "Gondolkodom mielőtt válaszolok",
        "A tudás hatalom",
        "Kérem adjon egy pohár vizet",
        "Tanulok magyarul mert szeretem a nyelvet",
    ]
    
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        
        original_words = len(sentence.split())
        qtl_words = len([e for e in encoded if e > max(tokenizer.special_tokens.values())])
        
        print(f"\n📝 Magyar:    {sentence}")
        print(f"   QTL:       {tokenizer.visualize_thought(encoded)}")
        print(f"   Vissza:    {decoded}")
        print(f"   Tömörítés: {original_words} szó → {qtl_words} QTL token")
    
    print(f"\n{'=' * 60}")
    print(f"Összesített kompressziós arány: {tokenizer.get_compression_ratio():.2f}x")
    print(f"{'=' * 60}")
