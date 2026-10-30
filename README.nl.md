# Een eerlijk clubklassement over leeftijden en disciplines heen

[English](README.md) | **Nederlands** | [Français](README.fr.md)

In de schietsport zorgen leeftijdscategorieën voor een eerlijke competitie. Een club heeft echter zelden genoeg schutters per categorie om dat te laten werken: een handvol pistoolschutters en een paar karabijnschutters, verspreid over zeven leeftijdsgroepen, is geen competitie. Zonder competitie is er weinig reden om mee te doen, en zonder schutters is er geen competitie.

Dit project stelt voor om de leeftijdscategorieën in de clubcompetitie los te laten **zonder dat de vergelijking oneerlijk wordt**. Het gebruikt jaren aan resultaten van het Belgisch kampioenschap om rekening te houden met de leeftijd, en zelfs met het verschil tussen luchtkarabijn en luchtpistool, zodat alle leden van een club in één klassement tegen elkaar kunnen strijden.

## Het idee: een score op 100

Elk resultaat wordt omgerekend naar een score: *stel dat er 100 schutters van jouw discipline en leeftijdsgroep op het Belgisch kampioenschap (BOA) meedoen. Hoeveel daarvan zou je verslaan?* Omdat die vraag in elke groep hetzelfde betekent, kan je een pistoolschutter van 55 vergelijken met een karabijnschutter van 25.

![Twee voorbeeldschutters op de echte BOA-verdelingen](pictures/score_explainer_nl.png?raw=true)

Schutter A schiet 590,0 met de luchtkarabijn bij senioren 1 en verslaat 57 van de 100 schutters uit die groep. Schutter B schiet 536 met het luchtpistool bij senioren 2: 54 punten minder, maar beter dan 86 van de 100 uit die groep. **In het clubklassement eindigt B voor A.**

Merk op hoe ongelijk de score-schaal is tegenover de punten: in het drukke midden van een groep steek je met een paar punten veel schutters voorbij, onderaan kost elke plaats veel punten.

Een score kan net boven 100 uitkomen als een resultaat beter is dan alles in de gegevens. Dat is de bedoeling: meer punten moet altijd een hogere score geven. Een olympische kwalificatie van 633,0 met de luchtkarabijn scoort 100,3, een kwalificatie van 587 met het luchtpistool 103,0.

## In de praktijk

Alle schutters van het Belgisch kampioenschap 2025, beide disciplines en alle leeftijden, in één klassement:

| # | Schutter | Discipline | Categorie | Resultaat (schoten) | Score |
| --: | :-- | :-- | :-- | --: | --: |
| 1 | DEBO Elise | Luchtpistool | Cadet | 375 (40) | 101,1 |
| 2 | DRUART Jean-Michel | Luchtpistool | D3 & S3 | 553 (60) | 100,0 |
| 3 | LIEBENS Bart | Luchtpistool | D1 & S1 | 568 (60) | 99,6 |
| 4 | ADRIAENSSEN Ruben | Luchtkarabijn | Cadet | 409,5 (40) | 99,4 |
| 5 | HENNES Sandra | Luchtkarabijn | Cadet | 409,4 (40) | 99,4 |
| 6 | SCHEPENS Nicolas | Luchtkarabijn | D1 & S1 | 623,1 (60) | 98,5 |
| 7 | VANDEVYVERE Emma | Luchtkarabijn | D1 & S1 | 623,0 (60) | 98,5 |
| 8 | HERBILLON Lana | Luchtkarabijn | Junior | 620,5 (60) | 98,4 |
| 9 | VANDROOGENBROECK Océane | Luchtkarabijn | Junior | 618,0 (60) | 97,4 |
| 10 | MAHAUT Claudy | Luchtkarabijn | D3 & S3 | 584,6 (60) | 97,4 |

Cadetten, junioren, senioren en veteranen van beide disciplines halen allemaal de top. Het volledige klassement staat in [`BOA 2025 Normalized.csv`](BOA%202025%20Normalized.csv). Of dat voor elke groep eerlijk is, en niet alleen aan de top, toont [de proef op de som](#de-proef-op-de-som).

## Gebruik

### Clubklassement in een rekenblad

`club/Clubklassement St. Michiel.xlsx` is een voorstel van klassement, opgesteld voor het formaat van de clubcompetitie van St. Michiel Roeselare: een seizoen van 10 wedstrijden van 30 schoten waarvan de beste 5 tellen. Het werkt in Excel, LibreOffice Calc en Google Sheets en bevat geen macro's. Gebruik het als voorbeeld: een ander formaat vraagt aanpassingen in `src/build_club_sheet.py` (en een correctietabel voor een ander aantal schoten).

- **Uitleg**: instellingen (seizoen, aantal tellende wedstrijden) en de handleiding.
- **Resultaten**: per schutter de naam, discipline, categorie en resultaten. Voor elk resultaat wordt de score ("op 100 schutters") berekend.
- **Klassement**: het afdrukbare klassement op één liggende pagina, gesorteerd op de som van de beste 5 scores. Elke schutter heeft twee lijnen: de geschoten resultaten met hun beste 5 en gemiddelde, en daaronder de scores.
- **Coëfficiënten**: de correctietabel waarin de scores opgezocht worden.

### De scripts uitvoeren

Installeer [uv](https://docs.astral.sh/uv/) en voer de scripts uit vanuit de hoofdmap van de repository. Bij de eerste keer installeert uv Python 3.11 en de nodige pakketten.

```
uv run src/process.py                    # verdelingen fitten, correctietabel en klassement van het laatste jaar schrijven
uv run src/process.py --year 2024        # klassement van een bepaald jaar
uv run src/process.py --no-plots         # de plotly-figuren niet openen
uv run src/build_club_sheet.py           # het clubrekenblad bouwen uit de correctietabel (--rows 200, --output <bestand>)
uv run src/plot_explainer.py             # de uitlegfiguur tekenen in het Engels, Nederlands en Frans (--html voor interactieve versies)
uv run src/build_poster.py               # A4-affiche in het Nederlands en Frans in poster/ (afgedrukt met Microsoft Edge)
```

| Optie van `process.py` | Standaard | Betekenis |
| :-- | :-- | :-- |
| `--use-shot-count-normalized` | 1 | Fitten op resultaten omgerekend naar `--normalized-shot-count` schoten |
| `--normalized-shot-count` | 30 | Aantal schoten om naar om te rekenen, bv. dat van een clubwedstrijd |
| `--use-unisex` | 1 | Dames en heren van dezelfde leeftijd samennemen |
| `--score-scale` | 100 | De score-schaal: "hoeveel zou je verslaan van zoveel schutters" |

| Uitvoer | Inhoud |
| :-- | :-- |
| `normalization_tables/LogGamma_CDF_30shots.csv` | Correctietabel van alle disciplines en groepen |
| `BOA <jaar> Normalized.csv` | Alle resultaten van dat jaar gerangschikt op hun score |
| `club/Clubklassement St. Michiel.xlsx` | Het clubrekenblad |
| `pictures/score_explainer_<en, nl, fr>.png` | De uitlegfiguur |

**Jaarlijkse update:** voeg de nieuwe BOA-resultaten toe aan `input_data/BE_national.csv` en voer daarna `process.py --no-plots`, `build_club_sheet.py` en `plot_explainer.py` uit.

## Hoe het werkt

### De gegevens

[`input_data/BE_national.csv`](input_data/BE_national.csv) bevat 3.096 kwalificatieresultaten van 10m luchtkarabijn (LK) en 10m luchtpistool (LP). Bijna allemaal komen ze van het Belgisch kampioenschap (BOA): karabijn 2018 - 2025, pistool 2018 - 2025 plus enkele eerdere jaren, en een paar dozijn pistoolresultaten van provinciale kampioenschappen. Resultaten die als afwijkend gemarkeerd zijn, worden weggelaten. België kent deze leeftijdscategorieën:

| Categorie | Afkorting | Leeftijd | Schoten per wedstrijd |
| :-: | :-: | :-: | :-: |
| Duiveltje | DUV | < 12 | 30 |
| Benjamin | BEN | 12 - 13 | 30 |
| Cadet | CAD | 14 - 15 | 40 |
| Junior Dame / Heer | JD / JH | 16 - 20 | 60 |
| Dame 1 / Senioren 1 | D1 / S1 | 21 - 49 | 60 |
| Dame 2 / Senioren 2 | D2 / S2 | 50 - 59 | 60 |
| Dame 3 / Senioren 3 | D3 / S3 | 60+ | 60 |

Elke discipline en leeftijdsgroep wordt apart behandeld, in vier stappen.

### Stap 1: hetzelfde aantal schoten

De categorieën schieten 30, 40 of 60 schoten, een clubwedstrijd 30. Alle resultaten worden eerst omgerekend naar het clubformaat (een resultaat van 600 op 60 schoten telt als 300), zodat de tabel rechtstreeks op clubresultaten werkt. De score hangt niet af van die omrekening: 590,0 op 60 schoten en 295,0 op 30 schoten scoren allebei 57. Daarom tonen de figuren de vertrouwde punten op 60 schoten.

### Stap 2: dames en heren samen

Vroeger schoten dames 40 schoten en heren 60, vandaag schieten beide er 60. In België scoren de dames met de luchtkarabijn consequent hoger dan de heren, maar D2 en D3 hebben te weinig resultaten om op zichzelf een betrouwbare verdeling te fitten. Aan de top van de sport presteren dames en heren gelijkwaardig, zoals de olympische finales van Tokio 2020 tonen:

![Olympische finales Tokio 2020](pictures/olympic_final.png?raw=true)

Daarom vormen dames en heren van dezelfde leeftijd één groep: DUV, BEN, CAD, JUN, D1 & S1, D2 & S2 en D3 & S3. De scripts kunnen nog altijd per geslacht werken (`--use-unisex 0`).

### Stap 3: een verdeling fitten

De resultaten van een groep worden beschreven met een statistische verdeling, zodat de score niet verspringt door het toeval van een paar resultaten. Een goede kandidaat moet de scherpe daling bovenaan vatten: er is een maximumscore en de beste schutters zitten er dicht onder.

Het eerste idee was een scheve normale verdeling, die slecht paste. De bètaverdeling ([hier gevonden](http://zoonek2.free.fr/UNIX/48_R/07.html)) kan zo'n scherpe daling maken:

![Theoretische bètafuncties](pictures/beta_function.png?raw=true)

Op 3 jaar gegevens paste ze nog altijd slecht, op 5 jaar beter, maar sommige categorieën bleven zich vreemd gedragen. De bibliotheek [distfit](https://erdogant.github.io/distfit/pages/html/index.html) testte daarna veel verdelingen op alle categorieën; dit is de rangschikking op de opgetelde score van alle categorieën (lager is beter):

![Fitresultaten voor verschillende verdelingen](pictures/dist_fitting_unisex.png?raw=true)

Zo kwamen loggamma en genextreme erbij in de vergelijking (luchtkarabijn, eerste analyse):

![Kansdichtheidsfuncties](pictures/pdf_unisex.png?raw=true)

De bètaverdeling gedraagt zich vreemd bij de cadetten (en bij D2 als de geslachten apart blijven). Een ideale fit gaat door het midden van elke staaf van het histogram:

![Ideaal histogram](pictures/ideal_histogram.jpg?raw=true)

Loggamma vat de piek en de scherpe daling het best, en distfit zet ze op de huidige gegevens van beide disciplines nog altijd op de eerste plaats. `process.py` waarschuwt als dat ooit verandert.

### Stap 4: van verdeling naar score

De cumulatieve verdelingsfunctie (CDF) van een groep geeft voor elk resultaat het aandeel van die groep dat lager scoort. De score is dat aandeel maal 100.

Om in een rekenblad te gebruiken wordt de CDF als tabel bewaard: 1.000 resultaten tussen het 1%- en het 99%-punt van elke groep, elk met hun CDF. Een resultaat wordt vermenigvuldigd met de factor `100 x CDF / punten` van de dichtstbijzijnde rij. Tussen 1% en 99% wijkt dat minder dan 0,2 af van het exacte aandeel. Daarbuiten geldt de factor van de rand, zodat de score blijft stijgen met het resultaat en uitzonderlijke resultaten boven 100 uitkomen. De tabel is [`normalization_tables/LogGamma_CDF_30shots.csv`](normalization_tables/LogGamma_CDF_30shots.csv); het clubrekenblad neemt ze over in het tabblad *Coëfficiënten*.

### Hoe we hier kwamen

De eerste versie had geen score op 100. Ze rekende elk resultaat om naar het resultaat dat het zou zijn in een **referentiecategorie** (senioren 1, 60 schoten). Twee manieren om dat te doen werden vergeleken:

- **Lineaire methode**: verdeel elke categorie tussen haar 1%- en 99%-punt in gelijke stappen en neem het resultaat bij dezelfde stap in de referentiecategorie.
- **CDF-methode**: neem het resultaat in de referentiecategorie met dezelfde CDF, dus het resultaat dat hetzelfde aandeel van zijn groep verslaat.

![Normalisatiemethodes](pictures/normalization_methods.png?raw=true)

Op de schaal van 600 punten van de referentiecategorie zagen beide er goed uit, de lineaire zelfs iets beter:

![Boxplot van de eerste versie](pictures/result_box_plot_unisex.png?raw=true)

Wiskundig klopt de lineaire methode echter niet: de steilheid van de CDF verschilt per categorie, dus gelijke stappen in punten bevatten in elke categorie een ander aandeel schutters. Twee zaken maakten de referentiecategorie daarna overbodig:

- **De gegevens van het luchtpistool kwamen erbij.** Een referentiecategorie van de karabijn kan geen pistoolschutters rangschikken, en een aparte referentie per discipline geeft nog altijd twee klassementen.
- **De score moest begrijpelijk worden.** "Je zou 57 van de 100 schutters van jouw leeftijd verslaan" zegt elk lid iets; "als senior had je 287,4 geschoten" niet.

Met het aandeel schutters als score zelf konden beide methodes vergeleken worden op de schaal waar het om draait. Daarmee was de keuze gemaakt, zoals het volgende hoofdstuk toont.

## De proef op de som

Als de scores eerlijk zijn, ziet elke leeftijdsgroep in beide disciplines er op de score-schaal hetzelfde uit: de middelste helft van de schutters tussen 25 en 75, de helft boven 50. De rode boxen, de CDF-methode die gebruikt wordt, tonen precies dat voor alle 3.086 resultaten in de gegevens.

![Score per discipline en leeftijdsgroep](pictures/score_box_plot.png?raw=true)

De blauwe boxen zijn de lineaire methode: ze scoort te hoog (medianen van 65 - 78) en verschilt per groep. Ze blijft enkel ter vergelijking.

<details>
<summary>Hoe lees je een boxplot?</summary>

![Boxplots lezen](pictures/reading_boxplots.png?raw=true)

De box bevat de middelste helft van de scores, de volle lijn in de box is de mediaan (de helft van de schutters scoort hoger) en de stippellijn het gemiddelde. De snorharen tonen het laagste en het hoogste kwart.
</details>

## Verder werk

- In 2024 veranderde de juniorcategorie van 16 - 17 naar 16 - 20 jaar, terwijl de meeste gegevens nog de oude leeftijden gebruiken. Schutters kunnen op naam gevolgd worden om hun overgang naar D1 / S1 met 3 jaar uit te stellen. Idealiter op 8 jaar gegevens, waarbij de oudste 3 jaar wegvallen zodat er genoeg gecorrigeerde gegevens overblijven.
- Een commando om een CSV met resultaten te scoren met een bestaande correctietabel, zonder opnieuw te fitten.
