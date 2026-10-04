# Jak działa model AI

Interaktywna strona edukacyjna (PL / EN): z czego składa się model AI – architektura i wagi – oraz jak jest zapisany na dysku (`config.json` + pliki wag, np. `.safetensors`).

To demonstracja małej sieci neuronowej, nie miniaturowa Llama. Implementacja definiuje operacje, konfiguracja ich ustawienia, a parametry zawierają liczby. Schemat LLM jest osobnym, konceptualnym porównaniem, nie działającym modelem językowym.

**Strona:** https://workszop.github.io/jakdzialamodel/

Jeden samodzielny plik `index.html` – bez instalacji i budowania, działa też po otwarciu z dysku.

## Eksperymenty

- Zmieniaj architekturę i wejścia. Nowa architektura losuje parametry od nowa; samo obliczenie odpowiedzi ich nie zmienia.
- Wybierz połączenie, bias lub komórkę macierzy. Ten sam parametr jest oznaczony w obliczeniu, tensorze i jego czterech bajtach float32.
- Przechodź przez obliczenie warstwa po warstwie. Skróty: `1`–`5` – sekcje, `F` – przepływ, `R` – losowanie wag.
- W lekcji owocowej uruchom trening, zatrzymaj go, wykonaj jedną epokę albo zresetuj. Wagi, błąd i granica decyzji wynikają z rzeczywistych obliczeń w przeglądarce. Powrót do eksploratora przywraca poprzednią sieć.

**Owoce:** sieć otrzymuje dwie syntetyczne cechy: długość i okrągłość, a nie piksele ilustracji. To nie jest rozpoznawanie zdjęć. Przykłady testowe nie uczestniczą w uczeniu. Wyniki modelu nie są gwarancją ani skalibrowaną pewnością.

**Precyzja:** przełącznik służy do szacowania rozmiaru i pokazywania zaokrągleń. Parametry oraz pobierany plik pozostają float32. Obliczenia demonstracji korzystają z liczb JavaScript; niższa precyzja nie jest tu uruchamiana jako osobny model.

Ilustracje truskawki i borówki oraz inspiracja końcowym sandboxem: [Neural Networks from Scratch, aegeorge42](https://github.com/aegeorge42/aegeorge42.github.io), CC0. Ilustracje są osadzone w HTML; oryginały i pochodzenie znajdują się w `assets/training/PROVENANCE.md`.

## Weryfikacja

Panel „Kontrola modelu” w aplikacji uruchamia te same inwarianty danych i DOM, które sprawdzają testy przeglądarkowe.

```bash
node --test tests/model.test.mjs tests/training.test.mjs
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs \
CHROME_BIN=/absolute/path/to/chrome \
node tests/browser-smoke.mjs
```

Playwright jest narzędziem do testów, nie zależnością aplikacji. Test przeglądarkowy uruchamia własny lokalny serwer na wolnym porcie i izolowaną przeglądarkę, sprawdza HTTP oraz `file://`, a następnie je zamyka. Opcjonalne `SMOKE_SCREENSHOT_DIR=/absolute/path` zapisuje zrzuty. Plany i materiały robocze pozostają poza historią Git.
