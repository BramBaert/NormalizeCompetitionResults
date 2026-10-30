''' Draw the explainer figure of the 0 - 100 score: two real BOA distributions (LK D1 & S1, LP D2 & S2)
    on their points scale with the score as a second scale below, and two example shooters showing how
    a lower pistol result can rank above a higher rifle result. Made in Dutch, English and French.

    The distributions are fitted like process.py does (loggamma on the results normalized to 30 shots,
    unisex groups) and the score uses the exported CDF correction table, exactly like the club workbook.
    The points are shown on <C_DISPLAY_SHOTS> shots, the national format of the shown groups: scaling a
    result and the table by the same factor leaves the score unchanged.

    Usage: uv run src/plot_explainer.py [--output pictures/score_explainer] [--html]
           writes <output>_nl.png, <output>_en.png and <output>_fr.png
'''
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats

C_SHOT_COUNT = 30           # shot count of the correction table
C_DISPLAY_SHOTS = 60        # shot count of the points shown in the figure
C_SCORE_SCALE = 100
SHOTS = {"DUV": 30, "BEN": 30, "CAD": 40}  # every other category shoots 60
GROUPS = {"D1": "D1 & S1", "S1": "D1 & S1", "D2": "D2 & S2", "S2": "D2 & S2"}

PANELS = [  # discipline, group, color, x range, bin size, example shooter, result (all in <C_DISPLAY_SHOTS> shot points)
    ("LK", "D1 & S1", "#636EFA", (456, 640), 4, "A", 590.0),
    ("LP", "D2 & S2", "#EF553B", (330, 570), 6, "B", 536),
]
TEXT = {
    "nl": dict(
        separators=",.",
        panel={"LK": "Luchtkarabijn · senioren 1 (21 - 49 jaar)", "LP": "Luchtpistool · senioren 2 (50 - 59 jaar)"},
        discipline={"LK": "LK", "LP": "LP"},
        source="BOA-resultaten op {shots} schoten, n = {n}",
        points="punten",
        scale="op 100<br>schutters",
        shooter="Schutter {who}",
        callout="<b>{shooter}: {result} punten</b><br>beter dan {score} van de 100",
        area="<b>{score} van de 100 schutters<br>schieten minder</b>",
        ranking="Clubklassement",
        rankingLine="{result} punten → <b>{score}</b> op 100"),
    "en": dict(
        separators=".,",
        panel={"LK": "Air rifle · seniors 1 (21 - 49 years)", "LP": "Air pistol · seniors 2 (50 - 59 years)"},
        discipline={"LK": "rifle", "LP": "pistol"},
        source="Belgian championship (BOA) results, {shots} shots, n = {n}",
        points="points",
        scale="out of 100<br>shooters",
        shooter="Shooter {who}",
        callout="<b>{shooter}: {result} points</b><br>beats {score} out of 100",
        area="<b>{score} out of 100 shooters<br>score lower</b>",
        ranking="Club ranking",
        rankingLine="{result} points → <b>{score}</b> out of 100"),
    "fr": dict(
        separators=", ",
        panel={"LK": "Carabine 10 m · Senior 1 (21 - 49 ans)", "LP": "Pistolet 10 m · Senior 2 (50 - 59 ans)"},
        discipline={"LK": "carabine", "LP": "pistolet"},
        source="Résultats BOA sur {shots} coups, n = {n}",
        points="points",
        scale="sur 100<br>tireurs",
        shooter="Tireur {who}",
        callout="<b>{shooter} : {result} points</b><br>bat {score} tireurs sur 100",
        area="<b>{score} tireurs sur 100<br>font moins</b>",
        ranking="Classement du club",
        rankingLine="{result} points → <b>{score}</b> sur 100"),
}
INK, MUTED = "#2a3f5f", "#506784"
X_DOMAIN = [0.06, 0.74]
Y_DOMAINS = [[0.64, 0.94], [0.14, 0.44]]
RULER_OFFSET = 0.085  # paper distance between the points axis and the score scale

def loadResults() -> pd.DataFrame:
    ''' The valid BOA results normalized to <C_SHOT_COUNT> shots with unisex groups, like process.py '''
    df = pd.read_csv('input_data/BE_national.csv', encoding='iso-8859-1')
    df = df.loc[df['Anomaly'] == "No"].copy()
    df['Result30'] = df['Result'] * C_SHOT_COUNT / df['Category'].map(SHOTS).fillna(60)
    df['Group'] = df['Category'].map(GROUPS).fillna(df['Category'])
    return df

def scoreFunction(discipline:str, group:str):
    ''' The score of results: result x (scale x CDF / points) at the nearest point of the correction table '''
    table = pd.read_csv(f'normalization_tables/LogGamma_CDF_{C_SHOT_COUNT}shots.csv')
    table = table.loc[table['Discipline'] == discipline]
    points, cdf = table[group].to_numpy(dtype=float), table[f'{group}_CDF'].to_numpy(dtype=float)
    def score(results):
        results = np.asarray(results, dtype=float)
        nearest = np.abs(points - results[..., None]).argmin(axis=-1)
        return results * C_SCORE_SCALE * cdf[nearest] / points[nearest]
    return score

def toPaper(value:float, axisRange:tuple, domain:list) -> float:
    return domain[0] + (value - axisRange[0]) / (axisRange[1] - axisRange[0]) * (domain[1] - domain[0])

def translucent(color:str, alpha:float) -> str:
    return f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})"

def formatResult(discipline:str, result:float, text:dict) -> str:
    ''' LK results with one decimal, LP results without, with the decimal mark of the language '''
    value = f"{result:.1f}" if discipline == "LK" else f"{result:.0f}"
    return value.replace(".", text["separators"][0])

def buildFigure(df:pd.DataFrame, text:dict) -> go.Figure:
    fig = go.Figure()
    layout, shapes, annotations, ranking = {}, [], [], []
    k = C_SHOT_COUNT / C_DISPLAY_SHOTS  # from shown points to the points of the fit and the table
    for n, (dis, group, color, (lo, hi), binSize, who, result) in enumerate(PANELS):
        ax, rulerAx = ("" if n == 0 else str(n + 1)), str(n + 3)
        fitData = df.loc[(df['Discipline'] == dis) & (df['Group'] == group), 'Result30'].to_numpy(dtype=float)
        data = fitData / k
        params = scipy.stats.loggamma.fit(fitData)
        tableScore = scoreFunction(dis, group)
        score = lambda r: tableScore(np.asarray(r) * k)
        density = lambda x: scipy.stats.loggamma.pdf(np.asarray(x) * k, *params) * k
        xs = np.linspace(lo, hi, 600)
        pdf = density(xs)
        yMax = max(pdf.max(), np.histogram(data, bins=np.arange(lo, hi + binSize, binSize), density=True)[0].max()) * 1.12
        resultScore = f"{float(score(result)):.0f}"
        resultPdf = float(density(result))
        shown, shooter = formatResult(dis, result, text), text["shooter"].format(who=who)
        ranking.append((float(score(result)), shooter, dis, shown, resultScore, color))

        # Real results, the fitted curve and the share of the group below the example shooter
        fig.add_trace(go.Histogram(x=data, xbins=dict(start=lo, end=hi, size=binSize), histnorm='probability density',
                                   marker=dict(color=color, line=dict(color="white", width=1)), opacity=0.22,
                                   xaxis=f"x{ax}", yaxis=f"y{ax}", hoverinfo="skip"))
        below = xs <= result
        fig.add_trace(go.Scatter(x=np.append(xs[below], result), y=np.append(pdf[below], resultPdf), fill='tozeroy',
                                 mode='none', fillcolor=translucent(color, 0.35), xaxis=f"x{ax}", yaxis=f"y{ax}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=pdf, mode='lines', line=dict(color=color, width=3.5),
                                 xaxis=f"x{ax}", yaxis=f"y{ax}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[result], y=[resultPdf], mode='markers', xaxis=f"x{ax}", yaxis=f"y{ax}",
                                 marker=dict(size=16, color="white", line=dict(color=INK, width=3.5)), hoverinfo="skip"))
        # Invisible trace so the score scale (an overlaying axis) is drawn
        fig.add_trace(go.Scatter(x=[lo], y=[0], mode='markers', marker=dict(opacity=0), xaxis=f"x{rulerAx}",
                                 yaxis=f"y{ax}", hoverinfo="skip"))

        # Score scale: a tick where the score reaches 10, 20, ... 100 on this group's points scale
        grid = np.arange(lo, hi, 0.05)
        scores = score(grid)
        ticks = [(float(grid[np.argmax(scores >= s)]), str(s)) for s in range(10, C_SCORE_SCALE + 1, 10) if scores.max() >= s]
        bottom = Y_DOMAINS[n][0]
        layout[f"xaxis{ax}"] = dict(domain=X_DOMAIN, range=[lo, hi], anchor=f"y{ax}", dtick=20, showgrid=False,
                                    ticks="outside", showline=True, linecolor="#444", tickfont=dict(size=15, color=INK))
        layout[f"yaxis{ax}"] = dict(domain=Y_DOMAINS[n], range=[0, yMax], anchor=f"x{ax}", showticklabels=False,
                                    gridcolor="#ebf0f8", zeroline=False)
        layout[f"xaxis{rulerAx}"] = dict(overlaying=f"x{ax}", anchor="free", position=bottom - RULER_OFFSET, range=[lo, hi],
                                         tickvals=[t for t, _ in ticks], ticktext=[s for _, s in ticks], showgrid=False,
                                         showline=True, linecolor=color, linewidth=5, ticks="outside", tickcolor=color,
                                         tickwidth=2.5, ticklen=8, tickfont=dict(size=15, color=color))
        annotations += [
            dict(text=f"<b>{text['panel'][dis]}</b>", x=X_DOMAIN[0], y=Y_DOMAINS[n][1] + 0.03, xref="paper", yref="paper",
                 xanchor="left", showarrow=False, font=dict(size=19, color=color)),
            dict(text=text["source"].format(shots=C_DISPLAY_SHOTS, n=len(data)), x=X_DOMAIN[1], y=Y_DOMAINS[n][1] + 0.03,
                 xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=13, color=MUTED)),
            dict(text=text["points"], x=X_DOMAIN[0] - 0.008, y=bottom - 0.022, xref="paper", yref="paper",
                 xanchor="right", showarrow=False, font=dict(size=14, color=MUTED)),
            dict(text=f"<b>{text['scale']}</b>", x=X_DOMAIN[0] - 0.008, y=bottom - RULER_OFFSET - 0.025, xref="paper",
                 yref="paper", xanchor="right", showarrow=False, font=dict(size=14, color=color), align="right"),
            # Call-out pointing at the example shooter on the curve
            dict(text=text["callout"].format(shooter=shooter, result=shown, score=resultScore), x=result, y=resultPdf,
                 xref=f"x{ax}", yref=f"y{ax}", ax=-190, ay=45, xanchor="right", align="left", showarrow=True,
                 arrowhead=0, arrowwidth=1.5, arrowcolor=INK, bgcolor="white", bordercolor=INK, borderwidth=1.5,
                 borderpad=8, font=dict(size=16, color=INK)),
            dict(text=text["area"].format(score=resultScore), x=result - (hi - lo) * 0.12, y=yMax * 0.1,
                 xref=f"x{ax}", yref=f"y{ax}", showarrow=False, font=dict(size=14, color=color)),
        ]
        # Dashed line from the shooter on the curve down to the score scale, with a dot on the scale
        xPaper = toPaper(result, (lo, hi), X_DOMAIN)
        shapes += [
            dict(type="line", xref="paper", yref="paper", x0=xPaper, x1=xPaper, y0=bottom - RULER_OFFSET,
                 y1=toPaper(resultPdf, (0, yMax), Y_DOMAINS[n]), line=dict(color=INK, width=2, dash="dash"), layer="below"),
            dict(type="circle", xref="paper", yref="paper", x0=xPaper - 0.006, x1=xPaper + 0.006,
                 y0=bottom - RULER_OFFSET - 0.0095, y1=bottom - RULER_OFFSET + 0.0095, fillcolor=INK, line=dict(width=0)),
        ]

    # The club ranking of the two example shooters
    lines = [f"<b>{text['ranking']}</b>", ""]
    for place, (_, shooter, dis, shown, resultScore, color) in enumerate(sorted(ranking, reverse=True), start=1):
        lines += [f"<b>{place}.</b>  <b><span style='color:{color}'>{shooter} ({text['discipline'][dis]})</span></b>",
                  f"<span style='color:{MUTED}'>      {text['rankingLine'].format(result=shown, score=resultScore)}</span>", ""]
    annotations.append(dict(text="<br>".join(lines[:-1]), x=0.985, y=0.5, xref="paper", yref="paper", xanchor="right",
                            yanchor="middle", align="left", showarrow=False, bgcolor="#f5f7fb", bordercolor="#c8d4e3",
                            borderwidth=1.5, borderpad=18, font=dict(size=18, color=INK)))
    # No title: the text around the figure introduces it
    fig.update_layout(
        template="plotly_white", width=1500, height=880, showlegend=False, separators=text["separators"],
        font=dict(family="Arial", color=INK), margin=dict(l=40, r=30, t=30, b=30), barmode="overlay",
        shapes=shapes, annotations=annotations, **layout)
    return fig

parser = argparse.ArgumentParser(description="Draw the explainer figure of the 0 - 100 score in Dutch, English and French")
parser.add_argument('--output', default='pictures/score_explainer', help="Output path without extension (default: pictures/score_explainer)")
parser.add_argument('--html', action='store_true', help="Also write an interactive html per language")

if __name__ == "__main__":
    args = parser.parse_args()
    df = loadResults()
    for lang, text in TEXT.items():
        fig = buildFigure(df, text)
        fig.write_image(f"{args.output}_{lang}.png", scale=1.5)
        print(f"Saved {args.output}_{lang}.png")
        if args.html:
            fig.write_html(f"{args.output}_{lang}.html")
