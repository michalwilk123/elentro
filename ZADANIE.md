# Zadanie rekrutacyjne — Python Developer (AI/RAG), etap M1

Dziękujemy za zainteresowanie ogłoszeniem. To krótkie zadanie praktyczne — sprawdzamy
podejście do problemu, nie kompletność wykonania. Nie oczekujemy gotowego produktu.

**Dane w tym katalogu są w całości fikcyjne** (wygenerowane pod to zadanie) — nie
pochodzą od żadnego realnego klienta. Struktura i "brudność" danych odzwierciedlają
realny problem, z którym pracujemy w etapie M1.

## Kontekst

Klient z branży dystrybucji technicznej ma katalog produktów napływający od wielu
producentów, w niespójnym formacie. Zanim cokolwiek da się zaindeksować i wyszukiwać,
trzeba dane znormalizować i wzbogacić.

## Dane wejściowe

- `dane/katalog_probka.csv` — ok. 230 pozycji, 5 fikcyjnych producentów, kilka
  kategorii produktowych. Dane są celowo niespójne:
  - różne zapisy tej samej wielkości opakowania (`50 rxn` / `50 reactions` / `x50` / `50szt`)
  - różne nazewnictwo tych samych kategorii, w tym mieszanie PL/EN, oraz puste kategorie
  - różne zapisy ceny (`PLN`, `zl`, z groszami, bez waluty)
  - duplikaty / near-duplikaty numerów katalogowych (literówki, powtórne wpisy)
  - brakujące pola (kategoria / cena / atrybuty)
- `dane/producent_novagen_snapshot.html` — zrzut strony jednego z producentów
  (fikcyjny, ale realistyczna, niepełna struktura — traktuj jak wynik scrapera).
  Zawiera opisy i specyfikacje części produktów z `katalog_probka.csv`, ale też:
  pozycję z literówką w numerze katalogowym, pozycję z inną nazwą niż w katalogu
  (ten sam SKU), oraz dwie pozycje spoza katalogu (nowy produkt / produkt
  wycofywany) — nie zakładaj, że każda karta ma parę w CSV.

To nie są błędy w danych testowych — to jest dokładnie to, z czym mierzy się etap M1
(normalizacja + Product Data Harvester, patrz ogłoszenie).

## Zadanie

1. **Normalizacja** — sprowadź `katalog_probka.csv` do jednego, spójnego schematu
   (np. Postgres). Rozstrzygnij duplikaty/near-duplikaty numerów katalogowych —
   pokaż jak i dlaczego.
2. **Mini-harvester** — sparsuj `producent_novagen_snapshot.html` i zmerguj z
   katalogiem: dopasuj po numerze katalogowym (nie po nazwie — nazwy się różnią),
   zdecyduj co zrobić z pozycjami bez pary po żadnej stronie, opisz krótko jak
   obsłużyłeś/aś literówkę w SKU.
3. **Wyszukiwarka hybrydowa** — dokładne dopasowanie po numerze katalogowym +
   wyszukiwanie semantyczne (dowolny embedding + pgvector/Qdrant/FAISS) + prosty
   filtr (np. po producencie/kategorii).
4. **Ocena trafności** — każdy wynik wyszukiwania ma dostać score + krótkie
   uzasadnienie (nie tylko liczbę).
5. **Interfejs** — CLI albo prosty endpoint HTTP wystarczy, nie oceniamy UI.

## Czego nie oceniamy

- wyglądu / UI
- ręcznego wzbogacania pozycji spoza dostarczonego snapshotu producenta
- "ładności" kodu ponad jego czytelność

## Co przysłać

- Repo (git — commit dyscyplina i czytelna historia zmian też się liczą, to część
  wymagań na etapie)
- `README.md` z Twoimi decyzjami projektowymi: co, dlaczego, jakie założenia
  przyjąłeś/przyjęłaś tam, gdzie dane były niejednoznaczne, co zrobił(a)byś inaczej
  z większym budżetem czasu

## Czas

Timebox: 2-3h pracy. Orientacyjny podział: normalizacja+dedup (45-60 min),
mini-harvester (45-60 min), wyszukiwarka+scoring (45-60 min), README (15-20 min).
Termin oddania: tydzień od otrzymania zadania — nie
liczymy godzin, liczy się wynik w rozsądnym czasie.

## Po oddaniu

Rozmowa techniczna wokół Twojego rozwiązania — pytamy o decyzje, nie egzaminujemy
z teorii.
