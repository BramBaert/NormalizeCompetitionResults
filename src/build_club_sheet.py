''' Build the club ranking workbook (xlsx, no macros) from the exported correction tables.

    The workbook normalizes club results (30 shots) of LK and LP shooters of all age categories
    to a 0 - 100 score (where the result falls within its own discipline and age group if that group
    had 100 shooters) using the CDF method of process.py, and ranks everyone on the sum of their best
    results. All calculations are spreadsheet formulas so the file works in Excel, LibreOffice Calc
    and Google Sheets. Towards the club the score is called 'op 100 schutters'.

    Usage: uv run src/build_club_sheet.py [--rows 60] [--output <file.xlsx>]
'''
import argparse
import datetime
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

C_SHOT_COUNT = 30
C_COMPETITIONS = 10
C_DEFAULT_COUNTING = 5
C_SCORE_SCALE = 100
C_SCORE_DECIMALS = 1
C_LOGO = 'pictures/stmichiel_logo.png'
C_RESULT_LABEL = "Geschoten"
C_NORMALIZED_LABEL = "Op 100 schutters"

DISCIPLINES = {"LK": "Luchtkarabijn", "LP": "Luchtpistool"}
GROUPS = ["DUV", "BEN", "CAD", "JUN", "D1 & S1", "D2 & S2", "D3 & S3"]
CATEGORIES = [  # (category, group, description)
    ("DUV", "DUV", "Duiveltje, < 12 jaar"),
    ("BEN", "BEN", "Benjamin, 12 - 13 jaar"),
    ("CAD", "CAD", "Cadet, 14 - 15 jaar"),
    ("JD", "JUN", "Junior dame, 16 - 20 jaar"),
    ("JH", "JUN", "Junior heer, 16 - 20 jaar"),
    ("D1", "D1 & S1", "Dame 1, 21 - 49 jaar"),
    ("S1", "D1 & S1", "Senioren 1, 21 - 49 jaar"),
    ("D2", "D2 & S2", "Dame 2, 50 - 59 jaar"),
    ("S2", "D2 & S2", "Senioren 2, 50 - 59 jaar"),
    ("D3", "D3 & S3", "Dame 3, 60+ jaar"),
    ("S3", "D3 & S3", "Senioren 3, 60+ jaar"),
]
MAX_RESULT = {"LK": 327, "LP": 300}  # 30 shots of 10.9 resp. 10
SCORE_FORMAT = "0." + "0" * C_SCORE_DECIMALS

FONT = "Arial"
BLACK, GOLD, RED, GREY = "000000", "F5C55E", "C8322A", "7F7F7F"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
HEADER_FILL = PatternFill("solid", fgColor=BLACK)
SUB_FILL = PatternFill("solid", fgColor="EDEDED")
LINE_FILL = PatternFill("solid", fgColor="F2F2F2")
COUNTING_FILL = PatternFill("solid", fgColor="FBE3A6")
MEDAL_FILLS = [PatternFill("solid", fgColor=c) for c in ("F5C55E", "D9D9D9", "E6B48A")]
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def font(**kwargs):
    return Font(name=FONT, **kwargs)

def coefficientTable(discipline:str) -> dict:
    ''' This function converts the exported CDF correction table into lookup tables per group
        Input:
            discipline:     The discipline of the results to normalize
        Return:
            A dict {group: (bounds, factors)}. A result r of the group is normalized as
            r * factors[i] where i is the last index with bounds[i] <= r and
            factors[i] = <C_SCORE_SCALE> x CDF[i] / points[i]. The bounds are the midpoints between
            the points of the table, so this picks the nearest point exactly like normalizeResultCDF
            in process.py does.
    '''
    table = pd.read_csv(f'normalization_tables/LogGamma_CDF_{C_SHOT_COUNT}shots.csv')
    table = table.loc[table['Discipline'] == discipline]
    result = {}
    for group in GROUPS:
        points = table[group].to_numpy(dtype=float)
        cdf = table[f'{group}_CDF'].to_numpy(dtype=float)
        factors = C_SCORE_SCALE * cdf / points
        bounds = np.concatenate([[0.0], (points[:-1] + points[1:]) / 2])
        result[group] = (bounds, factors)
    return result

def normalize(result:float, discipline:str, group:str, tables:dict) -> float:
    ''' Python equivalent of the spreadsheet formula, used to verify the workbook '''
    bounds, factors = tables[discipline][group]
    return round(result * factors[np.searchsorted(bounds, result, side='right') - 1], C_SCORE_DECIMALS)

def styleHeader(cell, fill=HEADER_FILL, color=GOLD):
    cell.font = font(bold=True, color=color)
    cell.fill = fill
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER

def addLogo(ws, anchor:str, size:int):
    img = Image(C_LOGO)
    img.width, img.height = size, int(size * img.height / img.width)
    ws.add_image(img, anchor)

def setPrint(ws, area:str, orientation:str, titleRows:str = None):
    ''' Print the area on A4, fitted to the page width '''
    ws.print_area = area
    if titleRows:
        ws.print_title_rows = titleRows
    ws.page_setup.orientation = orientation
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True

def countingRule(cell:str, row:str, counting:str) -> str:
    ''' Conditional format formula: the cell holds one of the <counting> best results of the row '''
    return f'AND(ISNUMBER({cell}),{cell}>=LARGE({row},MIN({counting},COUNT({row}))))'

def bestSum(row:str, counting:str) -> str:
    ''' Formula: the sum of the <counting> best results of the row '''
    return "+".join(f'IFERROR(LARGE({row},{k}),0)*({counting}>={k})' for k in range(1, C_COMPETITIONS + 1))

def buildInfoSheet(wb):
    ''' The 'Uitleg' sheet with the settings, instructions and category table. Returns the cell addresses '''
    ws = wb.active
    ws.title = "Uitleg"
    ws.sheet_view.showGridLines = False
    for col, width in zip("ABCD", (14, 30, 14, 62)):
        ws.column_dimensions[col].width = width
    addLogo(ws, "A1", 90)
    ws["B1"] = "Schuttersvereniging St. Michiel Roeselare"
    ws["B1"].font = font(bold=True, size=16)
    ws["B2"] = "Clubklassement over leeftijdscategorieën en disciplines heen"
    ws["B2"].font = font(size=12, color=RED)
    ws.row_dimensions[1].height = 36
    ws.row_dimensions[2].height = 36

    ws["B6"] = "Instellingen"
    ws["B6"].font = font(bold=True, size=12)
    settings = [("Seizoen", datetime.date.today().year, True, "Verschijnt in de titels van het klassement"),
                ("Aantal tellende wedstrijden", C_DEFAULT_COUNTING, True, f"De beste resultaten die meetellen (1 - {C_COMPETITIONS})"),
                ("Schoten per wedstrijd", C_SHOT_COUNT, False, f"Vast: de coëfficiënten zijn berekend voor {C_SHOT_COUNT} schoten")]
    for row, (label, value, editable, note) in enumerate(settings, start=7):
        ws.cell(row, 2, label).font = font()
        c = ws.cell(row, 3, value)
        c.font = font(bold=True)
        c.border = BORDER
        c.alignment = Alignment(horizontal="center")
        if editable:
            c.fill = INPUT_FILL
        ws.cell(row, 4, note).font = font(italic=True, color=GREY)
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2=str(C_COMPETITIONS),
                        showErrorMessage=True, errorTitle="Ongeldige waarde",
                        error=f"Geef een geheel getal tussen 1 en {C_COMPETITIONS}.")
    ws.add_data_validation(dv)
    dv.add("C8")

    lines = [
        ("Zo gebruik je dit bestand", True),
        ("1. Vul op het tabblad 'Resultaten' per schutter de naam, de discipline (LK of LP) en de categorie in.", False),
        ("   Een schutter die beide disciplines schiet, krijgt een rij per discipline.", False),
        (f"   Vul na elke wedstrijd het resultaat ({C_SHOT_COUNT} schoten) in: LK met decimalen, LP in gehele getallen.", False),
        ("   Laat de cel leeg als de schutter niet meedeed.", False),
        ("2. Het tabblad 'Klassement' sorteert zichzelf en is klaar om af te drukken. De tellende resultaten zijn gemarkeerd.", False),
        ("Alleen de gele cellen invullen, de rest wordt berekend. Rij 1 op 'Resultaten' is een voorbeeld: overschrijf of wis die.", False),
    ]
    for row, (text, bold) in enumerate(lines, start=12):
        ws.cell(row, 2, text).font = font(bold=bold, size=12 if bold else 10)

    catRow = 21
    ws.cell(catRow, 2, "Categorieën").font = font(bold=True, size=12)
    for col, title in enumerate(("Categorie", "Groep", "Omschrijving"), start=2):
        styleHeader(ws.cell(catRow + 1, col, title))
    for row, (cat, group, desc) in enumerate(CATEGORIES, start=catRow + 2):
        for col, value in enumerate((cat, group, desc), start=2):
            c = ws.cell(row, col, value)
            c.font = font()
            c.border = BORDER
    catRange = f"Uitleg!$B${catRow + 2}:$C${catRow + 1 + len(CATEGORIES)}"

    methodRow = catRow + len(CATEGORIES) + 4
    method = [
        (f"Wat betekent '{C_NORMALIZED_LABEL.lower()}'?", True),
        ("Jongeren, senioren, luchtkarabijn en luchtpistool halen gemiddeld andere scores. Daarom wordt elk resultaat omgerekend", False),
        ("naar een score op 100: stel dat er 100 schutters van jouw leeftijdsgroep en discipline op het Belgisch kampioenschap (BOA)", False),
        ("meedoen, hoeveel daarvan zou je dan verslaan? Een score van 60 betekent dat je beter schiet dan 60 van de 100.", False),
        ("Dames en heren van dezelfde leeftijd zitten in dezelfde groep: op topniveau scoren ze gelijkwaardig.", False),
        ("De coëfficiënten staan op het tabblad 'Coëfficiënten' en worden jaarlijks bijgewerkt met de nieuwe BOA-resultaten.", False),
        (f"Gegenereerd op {datetime.date.today():%d/%m/%Y} met github.com/BramBaert/NormalizeCompetitionResults (src/build_club_sheet.py).", False),
    ]
    for row, (text, bold) in enumerate(method, start=methodRow):
        ws.cell(row, 2, text).font = font(bold=bold, size=12 if bold else 10)
    ws.cell(methodRow + len(method) - 1, 2).font = font(italic=True, color=GREY)
    setPrint(ws, f"A1:D{methodRow + len(method) - 1}", "portrait")
    return {"season": "Uitleg!$C$7", "counting": "Uitleg!$C$8", "categories": catRange}

def buildCoefficientSheet(wb, tables:dict):
    ''' The 'Coëfficiënten' sheet: per discipline and group a column with bounds and one with factors.
        Row 5 holds the lookup key '<discipline>|<group>' above each bounds column. Returns the ranges.
    '''
    ws = wb.create_sheet("Coëfficiënten")
    ws["A1"] = f"Coëfficiënten naar een score op {C_SCORE_SCALE} ({C_SHOT_COUNT} schoten)"
    ws["A1"].font = font(bold=True, size=14)
    ws["A2"] = ("Score = resultaat x factor van de rij met de grootste grens <= resultaat. "
                "Buiten 1% - 99% van de spreiding geldt de factor van de rand.")
    ws["A3"] = "Niet wijzigen: deze tabel wordt gegenereerd door src/build_club_sheet.py (blad beveiligd zonder wachtwoord)."
    for cell in ("A2", "A3"):
        ws[cell].font = font(italic=True, color=GREY)
    firstRow = 7
    col = 1
    for discipline in DISCIPLINES:
        for group in GROUPS:
            bounds, factors = tables[discipline][group]
            ws.merge_cells(start_row=4, start_column=col, end_row=4, end_column=col + 1)
            styleHeader(ws.cell(4, col, f"{discipline} {group}"))
            ws.cell(5, col, f"{discipline}|{group}").font = font(size=8, color=GREY)
            styleHeader(ws.cell(6, col, "Grens"), fill=SUB_FILL, color=BLACK)
            styleHeader(ws.cell(6, col + 1, "Factor"), fill=SUB_FILL, color=BLACK)
            for i, (b, f) in enumerate(zip(bounds, factors)):
                ws.cell(firstRow + i, col, float(b)).number_format = "0.000"
                ws.cell(firstRow + i, col + 1, float(f)).number_format = "0.00000"
            ws.column_dimensions[get_column_letter(col)].width = 10
            ws.column_dimensions[get_column_letter(col + 1)].width = 10
            col += 2
    lastCol = get_column_letter(col - 1)
    lastRow = firstRow + len(bounds) - 1
    ws.freeze_panes = f"A{firstRow}"
    ws.protection.sheet = True
    setPrint(ws, f"A1:{lastCol}6", "landscape")  # printing the whole workbook should not print 1000 rows of coefficients
    return {"keys": f"'Coëfficiënten'!$A$5:${lastCol}$5",
            "data": f"'Coëfficiënten'!$A${firstRow}:${lastCol}${lastRow}"}

def buildResultSheet(wb, rows:int, refs:dict):
    ''' The 'Resultaten' sheet: input per shooter and the normalized results. Returns the ranges '''
    ws = wb.create_sheet("Resultaten")
    first, last = 7, 6 + rows
    nComp = C_COMPETITIONS
    # Column layout
    cNr, cName, cDisc, cCat = 1, 2, 3, 4
    cRes = 5                       # results W1..W10
    cNorm = cRes + nComp           # normalized W1..W10
    cBest = cNorm + nComp          # sum of the best <counting> normalized results
    cAvg = cBest + 1               # average of those results
    cRank = cAvg + 1
    cIdx = cRank + 1               # hidden: column index in the coefficient table
    cKey = cIdx + 1                # hidden: unique sort key for the ranking sheet
    L = get_column_letter
    counting = f"${L(cIdx)}$5"     # local copy of the setting, conditional formats can't refer to other sheets in every app

    addLogo(ws, "A1", 60)
    ws["C1"] = f'="Resultaten clubcompetitie "&{refs["season"]}'
    ws["C1"].font = font(bold=True, size=16)
    ws["C2"] = (f"Gele cellen invullen: naam, discipline, categorie en resultaat ({C_SHOT_COUNT} schoten). "
                "Leeg laten als de schutter niet meedeed. De rest wordt berekend.")
    ws["C2"].font = font(italic=True, color=GREY)
    ws.row_dimensions[1].height = 30
    ws.row_dimensions[2].height = 20
    ws[counting.replace("$", "")] = f'={refs["counting"]}'

    ws.merge_cells(start_row=4, start_column=cRes, end_row=4, end_column=cRes + nComp - 1)
    styleHeader(ws.cell(4, cRes, f"{C_RESULT_LABEL} ({C_SHOT_COUNT} schoten)"))
    ws.merge_cells(start_row=4, start_column=cNorm, end_row=4, end_column=cNorm + nComp - 1)
    styleHeader(ws.cell(4, cNorm, C_NORMALIZED_LABEL))

    headers = {cNr: "Nr", cName: "Naam", cDisc: "Disc.", cCat: "Cat.", cBest: f'="Beste "&{refs["counting"]}',
               cAvg: "Gemiddelde", cRank: "Plaats", cIdx: "kol", cKey: "sorteer"}
    headers.update({cRes + k: f"W{k + 1}" for k in range(nComp)})
    headers.update({cNorm + k: f"W{k + 1}" for k in range(nComp)})
    for col, title in headers.items():
        styleHeader(ws.cell(6, col, title))
    ws.row_dimensions[6].height = 22

    normRange = lambda r: f"${L(cNorm)}{r}:${L(cNorm + nComp - 1)}{r}"
    for r in range(first, last + 1):
        name, disc, cat, idx = (f"${L(c)}{r}" for c in (cName, cDisc, cCat, cIdx))
        best, norms = f"${L(cBest)}{r}", normRange(r)
        ws.cell(r, cNr, f"=ROW()-{first - 1}")
        ws.cell(r, cIdx, f'=IF(OR({disc}="",{cat}=""),"",IFERROR(MATCH({disc}&"|"&VLOOKUP({cat},{refs["categories"]},2,FALSE),{refs["keys"]},0),""))')
        for k in range(nComp):
            res = f"{L(cRes + k)}{r}"
            ws.cell(r, cNorm + k,
                    f'=IF(OR({idx}="",N({res})<=0),"",ROUND({res}*INDEX({refs["data"]},'
                    f'MATCH({res},INDEX({refs["data"]},0,{idx}),1),{idx}+1),{C_SCORE_DECIMALS}))')
        ws.cell(r, cBest, f'=IF(OR({name}="",COUNT({norms})=0),"",{bestSum(norms, counting)})')
        ws.cell(r, cAvg, f'=IF(N({best})=0,"",{best}/MIN({counting},COUNT({norms})))')
        ws.cell(r, cRank, f'=IF(N({best})=0,"",RANK({best},${L(cBest)}${first}:${L(cBest)}${last}))')
        ws.cell(r, cKey, f'=IF(N({best})=0,"",{best}+({last + 1}-ROW())/1000000000)')

        for col in range(1, cKey + 1):
            c = ws.cell(r, col)
            c.font = font(color=GREY) if col in (cNr, cIdx, cKey) else font(bold=(col in (cBest, cRank)))
            c.border = BORDER
            if col in (cName, cDisc, cCat) or cRes <= col < cRes + nComp:
                c.fill = INPUT_FILL
                c.protection = Protection(locked=False)
            if col != cName:
                c.alignment = Alignment(horizontal="center")
            if cNorm <= col < cNorm + nComp or col in (cBest, cAvg):
                c.number_format = SCORE_FORMAT
    normBlock = f"{L(cNorm)}{first}:{L(cNorm + nComp - 1)}{last}"
    ws.conditional_formatting.add(normBlock, FormulaRule(
        formula=[countingRule(f"{L(cNorm)}{first}", normRange(first), counting)], font=font(bold=True), fill=COUNTING_FILL))

    dvDisc = DataValidation(type="list", formula1=f'"{",".join(DISCIPLINES)}"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Ongeldige discipline", error="Kies LK of LP.")
    dvCat = DataValidation(type="list", formula1=f'"{",".join(c for c, _, _ in CATEGORIES)}"', allow_blank=True,
                           showErrorMessage=True, errorTitle="Ongeldige categorie", error="Kies een categorie uit de lijst.")
    firstRes = f"{L(cRes)}{first}"
    dvRes = DataValidation(type="custom", allow_blank=True, showErrorMessage=True, errorTitle="Ongeldig resultaat",
                           formula1=f'AND(ISNUMBER({firstRes}),{firstRes}>=0,{firstRes}<=IF(${L(cDisc)}{first}="LP",{MAX_RESULT["LP"]},{MAX_RESULT["LK"]}))',
                           error=f"Resultaat voor {C_SHOT_COUNT} schoten: LK 0 - {MAX_RESULT['LK']}, LP 0 - {MAX_RESULT['LP']}.")
    for dv, rng in ((dvDisc, f"{L(cDisc)}{first}:{L(cDisc)}{last}"),
                    (dvCat, f"{L(cCat)}{first}:{L(cCat)}{last}"),
                    (dvRes, f"{L(cRes)}{first}:{L(cRes + nComp - 1)}{last}")):
        ws.add_data_validation(dv)
        dv.add(rng)

    # Example row
    ws.cell(first, cName, "Voorbeeld Schutter")
    ws.cell(first, cDisc, "LK")
    ws.cell(first, cCat, "S2")
    for k, v in enumerate((287.4, 291.0, 289.6, 284.2, 290.3, 292.1)):
        ws.cell(first, cRes + k, v)

    widths = {cNr: 5, cName: 30, cDisc: 6, cCat: 6, cBest: 10, cAvg: 14, cRank: 7}
    for col in range(1, cKey + 1):
        ws.column_dimensions[L(col)].width = widths.get(col, 7.5)
    ws.column_dimensions[L(cIdx)].hidden = True
    ws.column_dimensions[L(cKey)].hidden = True
    ws.freeze_panes = f"{L(cRes)}{first}"
    setPrint(ws, f"A1:{L(cRank)}{last}", "landscape", "4:6")
    ws.protection.sheet = True
    col = lambda c: f"Resultaten!${L(c)}${first}:${L(c)}${last}"
    block = lambda c: f"Resultaten!${L(c)}${first}:${L(c + nComp - 1)}${last}"
    return {"name": col(cName), "disc": col(cDisc), "cat": col(cCat), "best": col(cBest), "avg": col(cAvg),
            "rank": col(cRank), "key": col(cKey), "results": block(cRes), "normalized": block(cNorm), "rows": rows}

def buildRankingSheet(wb, refs:dict, res:dict):
    ''' The printable 'Klassement' sheet, sorted with formulas only. Two lines per shooter:
        the shot results and below them the results at equal age, each with the sum and
        average of its best results.
    '''
    ws = wb.create_sheet("Klassement")
    nComp = C_COMPETITIONS
    first, last = 7, 6 + 2 * res["rows"]
    cRank, cName, cDisc, cCat, cLabel = 1, 2, 3, 4, 5
    cRes = 6
    cBest = cRes + nComp
    cAvg = cBest + 1
    cKey = cAvg + 1                # hidden: sort key of the n-th shooter
    cIdx = cKey + 1                # hidden: row of that shooter on 'Resultaten'
    L = get_column_letter
    counting = f"${L(cKey)}$5"

    ws.sheet_view.showGridLines = False
    addLogo(ws, "A1", 108)
    # The title centered over the full width of the page, the logo floats on the left
    ws.merge_cells(start_row=2, start_column=1, end_row=3, end_column=cAvg)
    ws["A2"] = f'="Luchtdruk Clubklassement "&{refs["season"]}'
    ws["A2"].font = font(bold=True, size=22)
    ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
    for row, height in ((1, 15), (2, 24), (3, 24), (4, 15), (5, 15)):
        ws.row_dimensions[row].height = height
    ws[counting.replace("$", "")] = f'={refs["counting"]}'

    headers = {cRank: "Pl.", cName: "Naam", cDisc: "Disc.", cCat: "Cat.", cLabel: "",
               cBest: f'="Beste "&{refs["counting"]}', cAvg: "Gemiddelde", cKey: "sorteer", cIdx: "rij"}
    headers.update({cRes + k: f"W{k + 1}" for k in range(nComp)})
    for col, title in headers.items():
        styleHeader(ws.cell(6, col, title))
    ws.row_dimensions[6].height = 22

    for n, r in enumerate(range(first, last + 1, 2), start=1):
        r2 = r + 1
        ws.cell(r, cKey, f'=IFERROR(LARGE({res["key"]},{n}),"")')
        ws.cell(r, cIdx, f'=IF(${L(cKey)}{r}="","",MATCH(${L(cKey)}{r},{res["key"]},0))')
        ws.cell(r2, cIdx, f'=${L(cIdx)}{r}')
        idx = f"${L(cIdx)}{r}"
        for col, src in ((cRank, "rank"), (cName, "name"), (cDisc, "disc"), (cCat, "cat")):
            ws.cell(r, col, f'=IF({idx}="","",INDEX({res[src]},{idx}))')
            ws.merge_cells(start_row=r, start_column=col, end_row=r2, end_column=col)
        ws.cell(r, cLabel, f'=IF({idx}="","","{C_RESULT_LABEL}")')
        ws.cell(r2, cLabel, f'=IF({idx}="","","{C_NORMALIZED_LABEL}")')
        for k in range(nComp):
            shot = f'INDEX({res["results"]},{idx},{k + 1})'
            ws.cell(r, cRes + k, f'=IF({idx}="","",IF({shot}="","",{shot}))')
            ws.cell(r2, cRes + k, f'=IF({idx}="","",INDEX({res["normalized"]},{idx},{k + 1}))')
        # The best shot results and their average, rounded to remove floating point noise of the decimal LK results
        shots, best = f"${L(cRes)}{r}:${L(cRes + nComp - 1)}{r}", f"${L(cBest)}{r}"
        ws.cell(r, cBest, f'=IF(OR({idx}="",COUNT({shots})=0),"",ROUND({bestSum(shots, counting)},1))')
        ws.cell(r, cAvg, f'=IF(N({best})=0,"",{best}/MIN({counting},COUNT({shots})))')
        for col, src in ((cBest, "best"), (cAvg, "avg")):
            ws.cell(r2, col, f'=IF({idx}="","",INDEX({res[src]},{idx}))')
        for row in (r, r2):
            for col in range(1, cAvg + 1):
                c = ws.cell(row, col)
                c.font = font(bold=(col == cRank or row == r2 and col == cBest), color=GREY if col == cLabel else None)
                c.alignment = Alignment(horizontal="left" if col in (cName, cLabel) else "center", vertical="center")
                if row == r2 and cRes <= col < cRes + nComp or row == r2 and col == cBest or col == cAvg:
                    c.number_format = SCORE_FORMAT  # the shot results and their sum keep the general format: LP has no decimals

    resBlock = f"{L(cRes)}{first}:{L(cRes + nComp - 1)}{last}"
    row1 = f"${L(cRes)}{first}:${L(cRes + nComp - 1)}{first}"
    row2 = f"${L(cRes)}{first + 1}:${L(cRes + nComp - 1)}{first + 1}"
    isLine2 = f"MOD(ROW({L(cRes)}{first}),2)={(first + 1) % 2}"
    cf = ws.conditional_formatting
    # The counting results: highlighted on the 'gelijke leeftijd' line, bold on the shot line above
    cf.add(resBlock, FormulaRule(formula=[f'AND({isLine2},{countingRule(f"{L(cRes)}{first}", row1, counting)})'],
                                 font=font(bold=True), fill=COUNTING_FILL, border=BORDER))
    cf.add(resBlock, FormulaRule(formula=[f'AND(NOT({isLine2}),{countingRule(f"{L(cRes)}{first + 1}", row2, counting)})'],
                                 font=font(bold=True), border=BORDER))
    for i, fill in enumerate(MEDAL_FILLS, start=1):
        cf.add(f"A{first}:A{last}", FormulaRule(formula=[f"$A{first}={i}"], fill=fill, border=BORDER))
    # Shade the 'gelijke leeftijd' line and draw borders only for filled rows so the empty part stays blank
    cf.add(f"{L(cLabel)}{first}:{L(cAvg)}{last}", FormulaRule(
        formula=[f'AND(MOD(ROW({L(cLabel)}{first}),2)={(first + 1) % 2},${L(cIdx)}{first}<>"")'], fill=LINE_FILL, border=BORDER))
    cf.add(f"A{first}:{L(cAvg)}{last}", FormulaRule(formula=[f'${L(cIdx)}{first}<>""'], border=BORDER))

    widths = {cRank: 5, cName: 26, cDisc: 6, cCat: 6, cLabel: 17, cBest: 10, cAvg: 14, cKey: 10, cIdx: 6}
    for col in range(1, cIdx + 1):
        ws.column_dimensions[L(col)].width = widths.get(col, 8)
    ws.column_dimensions[L(cKey)].hidden = True
    ws.column_dimensions[L(cIdx)].hidden = True
    ws.freeze_panes = f"A{first}"
    setPrint(ws, f"A1:{L(cAvg)}{last}", "landscape", "6:6")
    ws.protection.sheet = True

parser = argparse.ArgumentParser(description="Build the club ranking workbook")
parser.add_argument('--rows', type=int, default=60, help="Number of shooter rows (default: 60)")
parser.add_argument('--output', default='club/Clubklassement St. Michiel.xlsx', help="Output file")

if __name__ == "__main__":
    args = parser.parse_args()
    tables = {d: coefficientTable(d) for d in DISCIPLINES}
    wb = Workbook()
    refs = buildInfoSheet(wb)
    refs.update(buildCoefficientSheet(wb, tables))
    res = buildResultSheet(wb, args.rows, refs)
    buildRankingSheet(wb, refs, res)
    wb._sheets = [wb[name] for name in ("Uitleg", "Resultaten", "Klassement", "Coëfficiënten")]
    wb.calculation.fullCalcOnLoad = True  # openpyxl stores no cached values, let the spreadsheet calculate on open
    wb.properties.language = "nl-BE"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    wb.save(args.output)
    print(f"Saved {args.output} ({args.rows} rows, score on {C_SCORE_SCALE})")
