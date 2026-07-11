# GEMINI.md - AICH Project (SOTA Hybrid Cognitive Model)

## Projekt Áttekintés
Ez a projekt egy kutatás-orientált, ultra-optimalizált mesterséges intelligencia, amely egy hibrid architektúrára épül. A cél a SOTA (State-of-the-Art) teljesítmény elérése magyar nyelven, korlátozott lokális erőforrások mellett.

### Kulcsfontosságú Innovációk
- **Stabil Hybrid Architecture:** A Transformer-alapú Flash Attention és az SSM (State Space Model) alapú Mamba rétegek dinamikus fúziója.
- **Ontological Engine & VoidResonator:** Nagy dimenziós szemantikai sűrítés és fogalmi horgonyzás a mélyebb szövegértésért.
- **Quantum Thought Language (QTL):** A modell SAJÁT, EGYEDI belső gondolkodási nyelve. A QTL nem hasonlít semmilyen emberi nyelvre! A QTL egy neurális kompressziós protokoll, ami fogalmi szinten sűríti az információt.
- **QTL-alapú Latent Reasoning Space (LRS v2):** 32-dimenziós belső gondolkodási tér, ahol a modell a QTL nyelvén gondolkodik, mielőtt magyar nyelvre fordítaná a kimenetet. 8-32x gyorsabb mint az emberi nyelvű gondolkodás!
- **Concept Compressor:** Ultra-kompakt (32D) vektor reprezentáció a QTL tokenekből. Hierarchikus kompresszió 4 szinten: átlag, maximum, attention-alapú, és ön-rezonancia.
- **Holographic Memory:** Rögzített méretű, interferencia-alapú memória mechanizmus, amely lehetővé teszi a kontextus hosszú távú megőrzését.
- **Bilingual SOTA Tokenizer:** Egyedi, karakter- és számjegy-tudatos tokenizer, amely optimalizált a magyar ragozás és a matematikai logika kezelésére.
- **Cognitive Self-Correction:** Egy beépített `SelfCorrectionVerifier` modul, amely validálja és finomítja a modell kimeneti logitjait.

### A Quantum Thought Language (QTL) részletesen

A QTL a modell saját belső gondolkodási nyelve. Jellemzői:

1. **FOGALMI TÖMÖRÍTÉS:** Minden fogalom egyetlen tokenbe van sűrítve.
   - "az emberi tudatosság természete" → egyetlen QTL token
   - Emberi nyelvek: 5-10 szó → QTL: 1-2 token

2. **REZONANCIA-KÓDOLÁS:** A tokenek között nincs fix sorrend.
   - A tokenek egymással rezonálnak (interferencia mintázatok)
   - Ez lehetővé teszi a PÁRHUZAMOS gondolkodást

3. **KVANTUM SZUPERPOZÍCIÓ:** Egy QTL token egyszerre több jelentést hordozhat.

4. **HIERARCHIKUS TÖMÖRÍTÉS:** 4 szintű hierarchikus aggregáció.

5. **SEBESSÉG:** 8-32x gyorsabb, mint az emberi nyelvű gondolkodás.
   - 32 dimenziós vektortér vs 256 dimenziós emberi nyelvi tér
   - Párhuzamos rezonancia vs szekvenciális token feldolgozás

### Összehasonlítás: Emberi nyelvek vs QTL

Magyar: "A mesterséges intelligencia egy számítógépes rendszer"
Angol:  "Artificial intelligence is a computer system"
QTL:    "[XKQ][MNP][RST]..." (3 kompakt token a 32D térben)

Magyar: "Kérem, mondja meg, hogy mennyi az idő?"
Angol:  "Could you please tell me what time it is?"
QTL:    "[ABC][DEF]..." (2 token - a kérés formalitása irreleváns a QTL-ben)

## Fő Komponensek
- `model.py`: A `CognitiveBilingualModel` definíciója QTL-alapú belső gondolkodási térrel.
- `qtl_tokenizer.py`: A Quantum Thought Language (QTL) tokenizer - a modell saját belső nyelve.
- `concept_compressor.py`: Ultra-kompakt (32D) gondolati reprezentáció kompresszor.
- `qtl_reasoning_space.py`: A QTL-alapú belső gondolkodási tér (LRS v2).
- `latent_reasoning_space.py`: A régi LRS implementáció (megtartva a kompatibilitásért).
- `layers.py`: A `StabilHybridLayer` implementációja (Mamba + Flash Attention + Dinamikus Router).
- `ontological_engine.py`: A szemantikai rezonanciáért felelős `VoidResonator` és `OntologicalEngine`.
- `tokenizer.py`: A bilingvális, dinamikus szótárkezelő rendszer.
- `ultra_trainer.py`: Intenzív, szintetikus adatokkal és RSS feedekkel támogatott tréning pipeline.
- `chat_interface.py`: Interaktív chat felület QTL gondolatok megjelenítésével.

## Fejlesztési Állapot
- [x] **Phase 1: Stabil Alapok** - Hibrid rétegek és alap tréner kész.
- [x] **Phase 2: Bilingvális Alapozás** - Magyar B2 szintű korpusz integrálva.
- [x] **Phase 3: Világtudat** - RSS alapú élő adatgyűjtés és tanítás.
- [x] **Phase 4: Logikai Áttörés** - Szintetikus matematikai és nyelvtani generátor implementálva.
- [x] **Phase 5: Quantum Thought Language (QTL)** - A modell saját belső gondolkodási nyelve!
- [x] **Phase 6: QTL-alapú Reasoning Space (LRS v2)** - 32D kompakt gondolkodási tér.
- [ ] **Phase 7: Folyékony Beszéd** - A QTL→magyar fordítás finomhangolása.

## Használat
1. **Tréning indítása:** `python ultra_trainer.py`
2. **Beszélgetés:** `python chat_interface.py`
3. **QTL teszt:** `python qtl_tokenizer.py`
4. **Concept Compressor teszt:** `python concept_compressor.py`
5. **LRS v2 teszt:** `python qtl_reasoning_space.py`

## Követelmények
- Python 3.10+
- PyTorch (CUDA ajánlott, de CPU-n is fut)
- Transformers, feedparser

---
*A projekt folyamatosan öntanuló módban fejlődik. A modell saját belső nyelvet (QTL) fejleszt, ami nem hasonlít semmilyen emberi nyelvre. A QTL 8-32x gyorsabb gondolkodást tesz lehetővé, mint az emberi nyelvű feldolgozás.*
