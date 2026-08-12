# Katalog M1 — normalizacja, harvester i wyszukiwarka hybrydowa

Rozwiązanie [zadania rekrutacyjnego](ZADANIE.md): 228 wierszy z niespójnego CSV
jest normalizowanych i scalanych do 214 produktów, wzbogacanych ze snapshotu
producenta, a następnie udostępnianych w wyszukiwarce PL/EN.

[Interaktywne wyjaśnienie rozwiązania](https://michalwilk123.github.io/elentro/)
jest dostępne na GitHub Pages. To statyczna prezentacja; właściwa aplikacja
wymaga serwera i działa lokalnie.

## Uruchomienie

Wymagane: Python 3.12+ i [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run katalog-web
```

Otwórz `http://127.0.0.1:8000`. Pierwszy start tworzy `dane/katalog.sqlite3`,
importuje pliki źródłowe i zapisuje embeddingi. Kolejne uruchomienia korzystają
z trwałej bazy. Testy: `uv run pytest`.

## Architektura

```text
normalize.py   CSV -> porównywalne rekordy
resolve.py     228 rekordów -> 214 produktów
harvest.py     snapshot producenta -> opisy i zastosowania
store.py       SQLite, embeddingi, wyszukiwanie i runtime CRUD
search.py      zamiana tekstu na znormalizowane wektory
web.py         HTTP i prezentacja
```

## Najważniejsze decyzje

- **Normalizacja jest jawna i deterministyczna.** Kategorie, jednostki, ceny i
  atrybuty przechodzą przez słowniki oraz zakotwiczone parsery. Nieznana wartość
  pozostaje nieznana zamiast dostać prawdopodobne, ale nieweryfikowalne
  przypisanie. Surowe dane i źródło wartości są zachowane.
- **Duplikaty wymagają zgodności danych.** Ten sam SKU jest scalany, a warianty
  typu `CH-10248A` tylko wtedy, gdy producent, nazwa, opakowanie, cena i atrybuty
  są zgodne. Wariant zostaje aliasem. Sama odległość tekstowa byłaby ryzykowna:
  `CH-10248` i `CH-10258` to różne produkty.
- **Harvester dopasowuje po SKU, nie po nazwie.** Korekta znaków `l/I/O -> 1/0`
  jest akceptowana tylko wtedy, gdy wskazuje dokładnie jeden produkt. Pozycje bez
  pary są raportowane, ale nie trafiają automatycznie do katalogu.
- **Wyszukiwanie ma dwie rozłączne ścieżki.** Znany SKU lub alias daje dokładny
  wynik `1.00`; pozostałe zapytania używają wielojęzycznych embeddingów i cosine
  similarity liczonego w SQLite przez `sqlite-vec`. Filtry działają przed
  rankingiem, a embeddingi produktów i pól są trwałe.
- **CSV jest seedem, nie bazą runtime.** Pierwszy start zapisuje wynik pipeline'u
  do SQLite. Produkty można potem dodawać i usuwać w UI bez restartu; zmiana
  produktu i jego embeddingów odbywa się w jednej transakcji.

## Ograniczenia

- Mapy normalizacji są dopasowane do tej próbki; nowe etykiety są zgłaszane jako
  nieznane i wymagają przeglądu.
- Wynik cosine to podobieństwo, nie prawdopodobieństwo. Produkty o identycznych
  nazwach mogą różnić się w rankingu głównie przez producenta.
- Literówki w SKU wpisanym przez użytkownika nie są poprawiane. Bez rozpoznanego
  identyfikatora zapytanie przechodzi do wyszukiwania semantycznego.
- Model jest ogólny, nie laboratoryjny. Fachowe zapytania wymagają osobnego
  zestawu ewaluacyjnego przed zmianą modelu lub progów.
- Runtime CRUD przyjmuje gotowy produkt. Nie uruchamia ponownie deduplikacji
  całej dostawy CSV ani nie tworzy aliasów wariantów.

## Weryfikacja i dalsze kroki

`uv run pytest` obejmuje pełny pipeline, krytyczne scalenia, aliasy, konflikty
danych, trwałość SQLite, transakcyjne add/delete, filtry i obie ścieżki
wyszukiwania.

Kolejne kroki to edycja istniejących produktów, zestaw zapytań do pomiaru
recall@k oraz osobny raport jakości danych dla dostawcy.
