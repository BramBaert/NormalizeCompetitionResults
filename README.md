# Fair club rankings across ages and disciplines

**English** | [Nederlands](README.nl.md)

Shooting sports use age categories to keep competition fair. A local club, however, rarely has enough shooters per category to make that work: a handful of pistol shooters and a couple of rifle shooters spread over seven age groups is no competition at all. Without competition there is little reason to join, and without shooters there is no competition.

This project removes the age categories from club competitions **without making the comparison unfair**. It uses years of Belgian national championship results to account for age, and even for the difference between air rifle and air pistol, so every member of the club can compete in one ranking.

## The idea: a score out of 100

Every result is converted to a score: *imagine 100 shooters of your own discipline and age group at the Belgian championship (BOA). How many of them would you beat?* Because that question means the same in every group, a 55-year-old pistol shooter can be compared with a 25-year-old rifle shooter.

![Two example shooters on the real BOA distributions](pictures/score_explainer_en.png?raw=true)

Shooter A shoots 295.0 with the air rifle in seniors 1 and beats 57 out of 100 shooters of that group. Shooter B shoots 268 with the air pistol in seniors 2: 27 points fewer, but better than 86 out of 100 of that group. **In the club ranking, B finishes above A.**

Note how uneven the score scale is compared with the points: in the crowded middle of a group a few points move you past many shooters, at the bottom it takes a lot of points to climb.

A score can go slightly above 100 when a result is better than anything in the data. That is intended: more points must always mean a higher score. An Olympic-level air rifle qualification of 633.0 scores 100.3, an air pistol qualification of 587 scores 103.0.

## Seeing it in action

All shooters of the 2025 Belgian championship, both disciplines and all ages, in one ranking:

| # | Shooter | Discipline | Category | Result (shots) | Score |
| --: | :-- | :-- | :-- | --: | --: |
| 1 | DEBO Elise | Air pistol | Cadet | 375 (40) | 101.1 |
| 2 | DRUART Jean-Michel | Air pistol | D3 & S3 | 553 (60) | 100.0 |
| 3 | LIEBENS Bart | Air pistol | D1 & S1 | 568 (60) | 99.6 |
| 4 | ADRIAENSSEN Ruben | Air rifle | Cadet | 409.5 (40) | 99.4 |
| 5 | HENNES Sandra | Air rifle | Cadet | 409.4 (40) | 99.4 |
| 6 | SCHEPENS Nicolas | Air rifle | D1 & S1 | 623.1 (60) | 98.5 |
| 7 | VANDEVYVERE Emma | Air rifle | D1 & S1 | 623.0 (60) | 98.5 |
| 8 | HERBILLON Lana | Air rifle | Junior | 620.5 (60) | 98.4 |
| 9 | VANDROOGENBROECK Océane | Air rifle | Junior | 618.0 (60) | 97.4 |
| 10 | MAHAUT Claudy | Air rifle | D3 & S3 | 584.6 (60) | 97.4 |

Cadets, juniors, seniors and veterans of both disciplines all reach the top. The full ranking is in [`BOA 2025 Normalized.csv`](BOA%202025%20Normalized.csv). Whether that is fair for every group, not just at the top, is shown in [the proof of the pudding](#the-proof-of-the-pudding).

## Using it

### Club ranking workbook

`club/Clubklassement St. Michiel.xlsx` is a ready-to-use ranking for a club season of 10 competitions of 30 shots where the best 5 count. It works in Excel, LibreOffice Calc and Google Sheets and has no macros.

- **Uitleg**: settings (season, number of counting competitions) and instructions in Dutch.
- **Resultaten**: per shooter the name, discipline, category and results. The score ("op 100 schutters") is calculated for every result.
- **Klassement**: the printable ranking on one landscape page, sorted on the sum of the best 5 scores. Each shooter has two lines: the shot results with their best 5 and average, and below them the scores.
- **Coëfficiënten**: the correction table the scores are looked up in.

### Running the scripts

Install [uv](https://docs.astral.sh/uv/) and run the scripts from the repository root. On the first run uv installs Python 3.11 and the required packages.

```
uv run src/process.py                    # fit the distributions, write the correction table and the ranking of the latest year
uv run src/process.py --year 2024        # ranking of a specific year
uv run src/process.py --no-plots         # skip opening the plotly figures
uv run src/build_club_sheet.py           # build the club workbook from the correction table (--rows 200, --output <file>)
uv run src/plot_explainer.py             # draw the explainer figure in English and Dutch (--html for interactive versions)
```

| `process.py` option | Default | Meaning |
| :-- | :-- | :-- |
| `--use-shot-count-normalized` | 1 | Fit on results scaled to `--normalized-shot-count` shots |
| `--normalized-shot-count` | 30 | Shot count to scale to, e.g. the shot count of a club competition |
| `--use-unisex` | 1 | Merge ladies and men of the same age |
| `--score-scale` | 100 | The score scale: "how many would you beat out of this many shooters" |

| Output | Content |
| :-- | :-- |
| `normalization_tables/LogGamma_CDF_30shots.csv` | Correction table of all disciplines and groups |
| `BOA <year> Normalized.csv` | All results of that year ranked on their score |
| `club/Clubklassement St. Michiel.xlsx` | The club workbook |
| `pictures/score_explainer_en.png`, `pictures/score_explainer_nl.png` | The explainer figure |

**Yearly update:** add the new BOA results to `input_data/BE_national.csv`, then run `process.py --no-plots`, `build_club_sheet.py` and `plot_explainer.py`.

## How it works

### The data

[`input_data/BE_national.csv`](input_data/BE_national.csv) holds 3,096 qualification results of 10m air rifle (LK, *luchtkarabijn*) and 10m air pistol (LP, *luchtpistool*). Almost all are from the Belgian championship (BOA): rifle 2018 - 2025, pistol 2018 - 2025 plus a few earlier years, and a few dozen pistol results from provincial championships. Results marked as an anomaly are left out. Belgium has these age categories:

| Category | Abbreviation | Age | Shots per competition |
| :-: | :-: | :-: | :-: |
| Duiveltje | DUV | < 12 | 30 |
| Benjamin | BEN | 12 - 13 | 30 |
| Cadet | CAD | 14 - 15 | 40 |
| Junior Dame / Heer | JD / JH | 16 - 20 | 60 |
| Dame 1 / Senioren 1 | D1 / S1 | 21 - 49 | 60 |
| Dame 2 / Senioren 2 | D2 / S2 | 50 - 59 | 60 |
| Dame 3 / Senioren 3 | D3 / S3 | 60+ | 60 |

Every discipline and age group is handled separately, in four steps.

### Step 1: the same number of shots

The categories shoot 30, 40 or 60 shots, a club competition 30. All results are scaled to the club format first (a 60-shot result of 600 counts as 300), so the table can be used directly on club results.

### Step 2: ladies and men together

Historically ladies shot 40 shots and men 60; nowadays both shoot 60. In Belgium the ladies consistently score higher than the men with the air rifle, but the D2 and D3 categories have too few results to fit a reliable distribution on their own. At the top of the sport, ladies and men perform equally, as the Tokyo 2020 Olympic finals show:

![2020 Tokyo Olympic finals](pictures/olympic_final.png?raw=true)

So ladies and men of the same age form one group: DUV, BEN, CAD, JUN, D1 & S1, D2 & S2 and D3 & S3. The scripts can still work per gender (`--use-unisex 0`).

### Step 3: fitting a distribution

The results of a group are described with a statistical distribution, so the score does not jump around with the coincidences of a few results. A good candidate must capture the sharp fall-off at the top: there is a maximum score and the best shooters crowd just below it.

The first idea was a skewed normal distribution, which fitted poorly. The beta distribution ([found here](http://zoonek2.free.fr/UNIX/48_R/07.html)) can produce such a sharp fall-off:

![Theoretical beta functions](pictures/beta_function.png?raw=true)

On 3 years of data it still fitted badly, on 5 years better, but some categories kept misbehaving. The [distfit](https://erdogant.github.io/distfit/pages/html/index.html) library then tested many distributions on all categories; this is the ranking on the cumulative score of all categories (lower is better):

![Fitting results for different types of distributions](pictures/dist_fitting_unisex.png?raw=true)

That added loggamma and genextreme to the comparison (air rifle, first analysis):

![Probability density functions](pictures/pdf_unisex.png?raw=true)

Beta behaves strangely for the cadets (and for D2 when genders are separate). An ideal fit crosses the center of every bar of the histogram:

![Ideal histogram](pictures/ideal_histogram.jpg?raw=true)

Loggamma captures the peak and the sharp fall-off best, and distfit still ranks it first on today's data of both disciplines. `process.py` warns if that ever changes.

### Step 4: from distribution to score

The cumulative distribution function (CDF) of a group gives, for every result, the share of that group that scores lower. The score is that share times 100.

For use in a spreadsheet the CDF is stored as a table: 1,000 results between the 1% and 99% point of each group, each with its CDF. A result is multiplied by the factor `100 x CDF / points` of the nearest row. Within 1% - 99% this matches the exact share to within 0.2. Outside that range the factor of the edge applies, so the score keeps growing with the result and exceptional results end above 100. The table is [`normalization_tables/LogGamma_CDF_30shots.csv`](normalization_tables/LogGamma_CDF_30shots.csv); the club workbook copies it into its *Coëfficiënten* sheet.

### How we got here

The first version did not have a score out of 100. It converted every result to the result it would be in a **reference category** (seniors 1, 60 shots). Two ways to do that were compared:

- **Linear method**: split every category between its 1% and 99% point into equal steps and take the result at the same step in the reference category.
- **CDF method**: take the result in the reference category with the same CDF, i.e. the result that beats the same share of its group.

![Normalization methods](pictures/normalization_methods.png?raw=true)

On the 600-point scale of the reference category both looked good, the linear one even slightly better:

![Box plot of the first version](pictures/result_box_plot_unisex.png?raw=true)

The linear method is mathematically wrong, though: the steepness of the CDF differs per category, so equal steps in points hold a different share of shooters in each category. Two things then made the reference category obsolete:

- **Air pistol data was added.** A rifle reference category cannot rank pistol shooters, and a separate reference per discipline still gives two rankings.
- **The score was made understandable.** "You would beat 57 out of 100 shooters your age" means something to every member; "you would have shot 287.4 as a senior" does not.

With the share of shooters as the score itself, both methods could be compared on the scale where it matters. That settled it, as the next chapter shows.

## The proof of the pudding

If the scores are fair, every age group in both disciplines looks the same on the score scale: the middle half of the shooters between 25 and 75, half of them above 50. The red boxes, the CDF method that is used, show exactly that for all 3,086 results in the data.

![Score per discipline and age group](pictures/score_box_plot.png?raw=true)

The blue boxes are the linear method: it scores too high (medians of 65 - 78) and differs per group. It is only kept for comparison.

<details>
<summary>How to read a box plot</summary>

![Reading box plots](pictures/reading_boxplots.png?raw=true)

The box holds the middle half of the scores, the solid line in the box is the median (half of the shooters score higher) and the dashed line the mean. The whiskers show the lowest and highest quarter.
</details>

## Future work

- In 2024 the junior category changed from 16 - 17 to 16 - 20 years, while most data still uses the old ages. Shooters could be tracked by name to postpone their move to D1 / S1 by 3 years. Ideally on 8 years of data, dropping the oldest 3 so enough corrected data remains.
- A command to score a CSV of results with an existing correction table, without fitting again.
