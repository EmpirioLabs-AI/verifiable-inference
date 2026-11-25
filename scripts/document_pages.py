"""The pages of the OCR fingerprints, rendered from text written for this repository (CC0 1.0).

Nothing on them was published before, so a model cannot fill a gap from memory. They are clean renders at 1448 x
2048 pixels (an A4 page at the size the model card's own recipe gives a page) whose content is hard: display
mathematics, dense ruled tables with group headers, two and three columns with footnotes, accents and other scripts,
a form, a code listing and chemistry. Pillow draws the text and matplotlib's mathtext the mathematics, with the fonts
matplotlib ships (DejaVu, Bitstream Vera licence; Computer Modern and STIX for mathematics, OFL). Each fingerprint
records the SHA-256 of every page it uses: those files, not a new render, are what a check sends.

    pip install pillow matplotlib
    python scripts/document_pages.py <out dir>
"""
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import PIL  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
W, H = 1448, 2048
FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
SANS, SANSB, SANSI = "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"
SERIF, SERIFB, SERIFI = "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf"
MONO, MONOB = "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"
PAGES = []
LICENSE = "CC0 1.0 (written for this repository)"
SOURCE = "https://github.com/EmpirioLabs-AI/verifiable-inference/blob/main/scripts/document_pages.py"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def new_page():
    image = Image.new("L", (W, H), 255)
    return image, ImageDraw.Draw(image)


def wrap(draw, text, f, width):
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paragraph(draw, x, y, text, f, width, leading, justify=True, indent=0):
    lines = wrap(draw, text, f, width - indent)
    for i, line in enumerate(lines):
        lx = x + (indent if i == 0 else 0)
        lw = width - (indent if i == 0 else 0)
        words = line.split()
        if justify and i < len(lines) - 1 and len(words) > 1:
            gap = (lw - sum(draw.textlength(w, font=f) for w in words)) / (len(words) - 1)
            xx = lx
            for w in words:
                draw.text((xx, y), w, font=f, fill=0)
                xx += draw.textlength(w, font=f) + gap
        else:
            draw.text((lx, y), line, font=f, fill=0)
        y += leading
    return y


def save(image, name, title, contents, tool="Pillow %s" % PIL.__version__, fonts="DejaVu"):
    path = os.path.join(OUT, name)
    image.convert("L").save(path, optimize=True)
    PAGES.append({"file": name, "title": title, "source": SOURCE, "license": LICENSE,
                  "processing": "rendered with %s in %s, %d x %d px, 8-bit grey PNG: %s" % (tool, fonts, W, H,
                                                                                            contents)})


# ── Mathematics, with matplotlib's mathtext ─────────────────────────────────────────────────────────────────
def math_page(name, title, heading, items, fontset, contents, intro):
    plt.rcParams["mathtext.fontset"] = fontset
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    y = 120
    fig.text(110 / W, 1 - y / H, heading, fontsize=30, family="DejaVu Serif", weight="bold", va="top")
    y += 90
    for line in intro:
        fig.text(110 / W, 1 - y / H, line, fontsize=17, family="DejaVu Serif", va="top")
        y += 38
    y += 40
    for number, (text, formula) in enumerate(items, 1):
        fig.text(110 / W, 1 - y / H, text, fontsize=17, family="DejaVu Serif", va="top")
        y += 54
        fig.text(W / 2 / W, 1 - y / H, formula, fontsize=24, ha="center", va="top")
        fig.text((W - 110) / W, 1 - (y + 12) / H, "(%d)" % number, fontsize=18, family="DejaVu Serif", ha="right",
                 va="top")
        y += 100
    assert y < 1900, y
    fig.text(W / 2 / W, 1 - 1960 / H, "– 1 –", fontsize=15, family="DejaVu Serif", ha="center", va="top")
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    image = Image.open(path).convert("L")
    assert image.size == (W, H), image.size
    save(image, name, title, contents, tool="matplotlib %s (mathtext, %s)" % (matplotlib.__version__, fontset),
         fonts="DejaVu Serif")


math_page(
    "analysis-identities.png", "Ten identities of analysis, numbered", "Ten identities worth knowing",
    [("The Gaussian integral over the whole real line:", r"$\int_{-\infty}^{\infty} e^{-x^{2}}\,dx=\sqrt{\pi}$"),
     ("The sum of the reciprocal squares:", r"$\sum_{n=1}^{\infty}\frac{1}{n^{2}}=\frac{\pi^{2}}{6}$"),
     ("Stirling's approximation for large n:", r"$n!\sim\sqrt{2\pi n}\,\left(\frac{n}{e}\right)^{n}$"),
     ("Cauchy's integral formula, for a point a inside the contour:",
      r"$f(a)=\frac{1}{2\pi i}\oint_{\gamma}\frac{f(z)}{z-a}\,dz$"),
     ("The gamma function, for Re z > 0:", r"$\Gamma(z)=\int_{0}^{\infty}t^{z-1}e^{-t}\,dt$"),
     ("The Euler product, for Re s > 1:", r"$\zeta(s)=\sum_{n=1}^{\infty}n^{-s}=\prod_{p}\frac{1}{1-p^{-s}}$"),
     ("A Fourier series on the interval from minus pi to pi:",
      r"$f(x)=\frac{a_{0}}{2}+\sum_{n=1}^{\infty}\left(a_{n}\cos nx+b_{n}\sin nx\right)$"),
     ("Differentiation under the integral sign:",
      r"$\frac{d}{dx}\int_{a}^{b}f(x,t)\,dt=\int_{a}^{b}\frac{\partial f}{\partial x}(x,t)\,dt$"),
     ("The binomial theorem:", r"$(x+y)^{n}=\sum_{k=0}^{n}\binom{n}{k}x^{n-k}y^{k}$"),
     ("A limit that defines e:", r"$e=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n}\approx 2.71828$")],
    "cm", "a heading, two lines of text, ten numbered display equations (integrals, sums, products, limits, "
    "fractions, a binomial coefficient) each with a line of text, a page number",
    ["Each identity below is stated without proof. The numbers on the right are", "used for reference in the "
     "exercises that follow on the next page."])

math_page(
    "physics-equations.png", "Eight equations of physics, numbered", "Field equations and their friends",
    [("Gauss's law, in differential form:", r"$\nabla\cdot\mathbf{E}=\frac{\rho}{\varepsilon_{0}}$"),
     ("No magnetic monopoles:", r"$\nabla\cdot\mathbf{B}=0$"),
     ("Faraday's law of induction:", r"$\nabla\times\mathbf{E}=-\frac{\partial\mathbf{B}}{\partial t}$"),
     ("Ampère's law with Maxwell's correction:",
      r"$\nabla\times\mathbf{B}=\mu_{0}\mathbf{J}+\mu_{0}\varepsilon_{0}\frac{\partial\mathbf{E}}{\partial t}$"),
     ("The time-dependent Schrödinger equation:",
      r"$i\hbar\frac{\partial\Psi}{\partial t}=\left[-\frac{\hbar^{2}}{2m}\nabla^{2}+V(\mathbf{r},t)\right]\Psi$"),
     ("Momentum balance for an incompressible fluid:",
      r"$\rho\left(\frac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)"
      r"=-\nabla p+\mu\nabla^{2}\mathbf{u}+\mathbf{f}$"),
     ("Planck's law for the spectral radiance of a black body:",
      r"$B_{\nu}(T)=\frac{2h\nu^{3}}{c^{2}}\,\frac{1}{e^{h\nu/k_{B}T}-1}$"),
     ("The Lorentz factor:", r"$\gamma=\frac{1}{\sqrt{1-v^{2}/c^{2}}}$")],
    "stix", "a heading, two lines of text, eight numbered display equations (vector calculus in bold, partial "
    "derivatives, Greek letters, a square root) each with a line of text, a page number",
    ["Bold letters are vectors. SI units throughout; the constants carry their", "usual meaning."])


# ── Tables ──────────────────────────────────────────────────────────────────────────────────────────────────
def ruled_table(draw, x, y, columns, rows, f, fb, row_h, groups=None, aligns=None):
    """A ruled table: optional group headers spanning columns, a header row, rows; returns the bottom y."""
    widths = [c[1] for c in columns]
    total = sum(widths)
    draw.line([(x, y), (x + total, y)], fill=0, width=3)
    if groups:
        for label, span in groups:
            gw = sum(widths[span[0]:span[1]])
            gx = x + sum(widths[:span[0]])
            if label:
                tw = draw.textlength(label, font=fb)
                draw.text((gx + (gw - tw) / 2, y + 8), label, font=fb, fill=0)
                draw.line([(gx + 10, y + row_h - 4), (gx + gw - 10, y + row_h - 4)], fill=0, width=1)
        y += row_h
    cx = x
    for (label, width), align in zip(columns, aligns or ["l"] * len(columns)):
        tw = draw.textlength(label, font=fb)
        tx = cx + 8 if align == "l" else cx + width - 8 - tw if align == "r" else cx + (width - tw) / 2
        draw.text((tx, y + 8), label, font=fb, fill=0)
        cx += width
    y += row_h
    draw.line([(x, y), (x + total, y)], fill=0, width=2)
    for row in rows:
        cx = x
        for value, width, align in zip(row, widths, aligns or ["l"] * len(columns)):
            tw = draw.textlength(value, font=f)
            tx = cx + 8 if align == "l" else cx + width - 8 - tw if align == "r" else cx + (width - tw) / 2
            draw.text((tx, y + 7), value, font=f, fill=0)
            cx += width
        y += row_h
    draw.line([(x, y), (x + total, y)], fill=0, width=3)
    return y


def quarterly():
    rng = random.Random(4101)
    image, draw = new_page()
    draw.text((90, 90), "Larkspur & Vale Instruments: net sales by product line", font=font(SERIFB, 34), fill=0)
    paragraph(draw, 90, 150, "Unaudited. In thousands of units of account; figures in parentheses are decreases. "
              "The change column compares the fourth quarter of each year.", font(SERIF, 22), 1268, 32)
    lines = ["Optical benches", "Spectrometers", "Calibration kits", "Field sensors", "Data loggers",
             "Thermal stages", "Vacuum gauges", "Laser mounts", "Fibre couplers", "Signal analysers",
             "Power supplies", "Microscope stages", "Interferometers", "Load cells", "Flow meters",
             "Humidity probes", "Torque testers", "Strain gauges", "Encoders", "Servo drives", "Light meters",
             "Gas detectors", "Seismometers", "Anemometers", "Refractometers", "Viscometers"]
    rows = []
    for name in lines:
        values = [rng.uniform(80, 2400) for _ in range(8)]
        cells = ["{:,.1f}".format(v) for v in values]
        change = (values[7] - values[3]) / values[3] * 100
        rows.append([name] + cells + [("+%.1f%%" % change) if change >= 0 else "(%.1f%%)" % -change])
    total = [sum(float(r[i].replace(",", "")) for r in rows) for i in range(1, 9)]
    rows.append(["Total"] + ["{:,.1f}".format(v) for v in total] + ["+%.1f%%" % ((total[7] - total[3]) / total[3] * 100)
                                                                     if total[7] >= total[3] else
                                                                     "(%.1f%%)" % ((total[3] - total[7]) / total[3] * 100)])
    columns = [("Product line", 230)] + [(q, 118) for q in ("Q1", "Q2", "Q3", "Q4")] * 2 + [("Change", 104)]
    y = ruled_table(draw, 70, 250, columns, rows, font(SANS, 19), font(SANSB, 19), 54,
                    groups=[("2024", (1, 5)), ("2025", (5, 9))], aligns=["l"] + ["r"] * 9)
    paragraph(draw, 90, y + 30, "Notes: (a) Interferometers include the two models withdrawn in the second quarter "
              "of 2025. (b) Totals may not add up because of rounding.", font(SERIFI, 20), 1268, 30)
    draw.text((W / 2 - 10, 1960), "7", font=font(SERIF, 20), fill=0)
    save(image, "table-quarterly.png", "A ruled table of quarterly figures with group headers",
         "a heading, a note, a ruled table of 27 rows and 10 columns under two group headers (numbers with "
         "thousands separators, decreases in parentheses, percentages), footnotes, a page number")


def measurements():
    rng = random.Random(4202)
    image, draw = new_page()
    draw.text((90, 90), "Table 3. Conductivity of the annealed samples", font=font(SERIFB, 32), fill=0)
    paragraph(draw, 90, 148, "Each sample was measured three times at the stated temperature; the uncertainty is one "
              "standard deviation. Samples marked with an asterisk were re-polished between runs.",
              font(SERIF, 22), 1268, 32)
    rows = []
    for i in range(30):
        batch = "B-%02d%s" % (rng.randint(1, 40), rng.choice("ABCDE"))
        mass = "%.2f" % rng.uniform(10, 99)
        temp = "%.1f" % rng.choice([77.4, 150.0, 223.2, 273.2, 298.2, 350.0, 412.5])
        sigma = "%.3e" % rng.uniform(1e3, 9e6)
        mantissa, exponent = sigma.split("e")
        sigma = "%s × 10^%d" % (mantissa, int(exponent))
        unc = "± %.1f %%" % rng.uniform(0.2, 6.5)
        note = rng.choice(["", "", "", "*", "cracked edge", "*", "", "re-run", "", "oxidised"])
        rows.append(["S%03d" % (101 + i * 3), batch, mass, temp, sigma, unc, note])
    columns = [("Sample", 130), ("Batch", 140), ("Mass (mg)", 160), ("T (K)", 130), ("σ (S/m)", 280),
               ("Uncertainty", 190), ("Remarks", 238)]
    y = ruled_table(draw, 90, 240, columns, rows, font(SANS, 20), font(SANSB, 20), 52,
                    aligns=["l", "l", "r", "r", "r", "r", "l"])
    draw.text((90, y + 24), "* re-polished between runs", font=font(SERIFI, 20), fill=0)
    save(image, "table-measurements.png", "A ruled table of laboratory measurements",
         "a caption, a note, a ruled table of 30 rows and 7 columns (identifiers, decimals, scientific notation "
         "with a Greek letter and a multiplication sign, plus-minus values, remarks), a footnote")


def timetable():
    image, draw = new_page()
    draw.text((90, 90), "Coastal line: weekday departures", font=font(SANSB, 36), fill=0)
    draw.text((90, 148), "Valid from the first Monday in March. Times are local.", font=font(SANS, 22), fill=0)
    stations = ["Ashford Quay", "Brindle Cross", "Cobham Sands", "Dunmere", "Eastgate Park", "Fennick",
                "Gorse Hill", "Harwick Junction", "Ivel Bridge", "Juniper Bay", "Kestle", "Larch Point",
                "Merrow Halt", "Northcliff", "Orwell Heath", "Pennard", "Quarry Lane", "Rook Harbour"]
    trains = ["1A02", "1A04", "2C11", "1A06", "2C13", "1A08", "2C15", "1A10"]
    rng = random.Random(4303)
    starts = [5 * 60 + 42, 6 * 60 + 15, 6 * 60 + 48, 7 * 60 + 5, 7 * 60 + 33, 8 * 60 + 2, 8 * 60 + 40, 9 * 60 + 18]
    grid = []
    times = list(starts)
    for s, station in enumerate(stations):
        row = [station + (" (a)" if station in ("Fennick", "Merrow Halt") else "")]
        for t in range(len(trains)):
            if s and trains[t].startswith("2C") and rng.random() < 0.35:
                row.append("|")
                times[t] += rng.randint(2, 4)
                continue
            if s:
                times[t] += rng.randint(3, 9)
            row.append("%02d:%02d" % (times[t] // 60 % 24, times[t] % 60))
        grid.append(row)
    columns = [("Station", 330)] + [(t, 128) for t in trains]
    y = ruled_table(draw, 70, 220, columns, grid, font(MONO, 21),
                    font(SANSB, 21), 56, aligns=["l"] + ["c"] * len(trains))
    for line in ["| does not stop at this station", "(a) request stop: tell the conductor before the previous "
                 "station", "Trains 2C11, 2C13 and 2C15 do not run on public holidays."]:
        draw.text((90, y + 26), line, font=font(SANS, 20), fill=0)
        y += 34
    save(image, "table-timetable.png", "A train timetable of 18 stations and 8 trains",
         "a heading, a line of text, a ruled table of 18 rows and 9 columns in a monospaced face (times, a bar for "
         "a train that does not stop, a request-stop mark), three footnotes")


# ── Columns, footnotes and running heads ────────────────────────────────────────────────────────────────────
ARTICLE = [
    ("1 Introduction",
     "Small glass resonators lose energy in three ways: through the surface, through the mount and through the air "
     "around them. Which of the three dominates depends on the size of the resonator and on how carefully it was "
     "cleaned, and the literature gives figures that differ by an order of magnitude. We set out to measure all "
     "three on the same pieces, in the same week, with the same instruments, so that the comparison would not rest "
     "on assumptions about anybody else's laboratory."),
    ("",
     "The pieces were cut from a single rod of fused silica, polished to the same finish and numbered before they "
     "were cleaned. Half were cleaned with the solvent sequence we normally use; the other half were cleaned with "
     "water alone, to see whether the solvents mattered at all.¹ The results are in Table 2 and in Figure 1."),
    ("2 Method",
     "Each resonator hung from two loops of thin wire inside a chamber that could be pumped down to a pressure "
     "below one thousandth of a pascal. A small electrode drove the fundamental mode, and an interferometer "
     "recorded the decay of the motion once the drive was switched off. The time it took the amplitude to fall to "
     "a tenth of its starting value gave the loss for that run."),
    ("",
     "We repeated every measurement at five pressures, from atmospheric down to the lowest the pump would reach, "
     "and at two wire diameters. Changing the pressure separates the loss to the air; changing the wire separates "
     "the loss to the mount; what remains is the loss in the glass and its surface.²"),
    ("3 Results",
     "The loss to the air fell with pressure exactly as the simple theory predicts, and below one pascal it was too "
     "small to measure. The loss to the mount was larger than we expected for the thicker wire and almost vanished "
     "for the thinner one. The pieces cleaned with water alone lost about a third more energy than the others, "
     "which suggests that the solvents remove a film the water leaves behind."),
    ("",
     "Two resonators broke during handling and are not included. One showed a loss four times larger than its "
     "neighbours; when we examined it under the microscope, we found a scratch near one of the wire loops, and we "
     "report it separately in the appendix."),
]


def article():
    image, draw = new_page()
    draw.text((90, 60), "Proceedings of the Workshop on Precision Measurement, Vol. 12", font=font(SERIFI, 19), fill=0)
    draw.text((W - 90 - draw.textlength("pp. 41–47", font=font(SERIFI, 19)), 60), "pp. 41–47", font=font(SERIFI, 19),
              fill=0)
    draw.line([(90, 92), (W - 90, 92)], fill=0, width=1)
    title = "Where the energy goes: three loss channels in small glass resonators"
    y = 120
    for line in wrap(draw, title, font(SERIFB, 38), 1100):
        tw = draw.textlength(line, font=font(SERIFB, 38))
        draw.text(((W - tw) / 2, y), line, font=font(SERIFB, 38), fill=0)
        y += 50
    authors = "Mira Ostrander¹, Tobiah Keane² and Iolanthe Brack¹"
    tw = draw.textlength(authors, font=font(SERIF, 24))
    draw.text(((W - tw) / 2, y + 10), authors, font=font(SERIF, 24), fill=0)
    affil = "¹ Hollin Institute, Department of Physics   ² Varrow College, School of Engineering"
    tw = draw.textlength(affil, font=font(SERIFI, 18))
    draw.text(((W - tw) / 2, y + 48), affil, font=font(SERIFI, 18), fill=0)
    y += 100
    draw.rectangle([(130, y), (W - 130, y + 170)], outline=0, width=1)
    draw.text((150, y + 14), "Abstract", font=font(SERIFB, 21), fill=0)
    paragraph(draw, 150, y + 44, "We measured the energy lost to the air, to the mount and to the material in "
              "twenty fused silica resonators cut from one rod. Below one pascal the air no longer matters; a thin "
              "wire removes most of the mount loss; and cleaning with solvents lowers the remaining loss by about a "
              "quarter.", font(SERIF, 19), W - 300, 27)
    y += 200
    col_w, gap = 600, 68
    x_cols = [90, 90 + col_w + gap]
    col, cy = 0, y
    f_body, f_head = font(SERIF, 20), font(SERIFB, 22)
    bottom = 1780
    for heading, text in ARTICLE:
        need = (40 if heading else 0) + 29 * len(wrap(draw, text, f_body, col_w - 30)) + 14
        if cy + need > bottom and col == 0:
            col, cy = 1, y
        if heading:
            draw.text((x_cols[col], cy), heading, font=f_head, fill=0)
            cy += 40
        cy = paragraph(draw, x_cols[col], cy, text, f_body, col_w, 29, indent=0 if heading else 30) + 14
    # a figure in the right column
    fy = max(cy + 10, y) if col == 1 else y
    if fy + 330 < bottom:
        draw.rectangle([(x_cols[1], fy), (x_cols[1] + col_w, fy + 240)], outline=0, width=2)
        for i in range(12):
            px = x_cols[1] + 40 + i * 45
            py = fy + 200 - int(150 * (0.92 ** (i * 2)))
            draw.ellipse([(px - 5, py - 5), (px + 5, py + 5)], fill=0)
        draw.text((x_cols[1] + 20, fy + 12), "loss", font=font(SANS, 16), fill=0)
        draw.text((x_cols[1] + col_w - 110, fy + 214), "pressure", font=font(SANS, 16), fill=0)
        paragraph(draw, x_cols[1], fy + 252, "Figure 1. Loss to the air against pressure for resonator 7, on a "
                  "logarithmic scale.", font(SERIFI, 18), col_w, 25, justify=False)
    draw.line([(90, 1810), (480, 1810)], fill=0, width=1)
    draw.text((90, 1822), "¹ Water from the same still, used within an hour.", font=font(SERIF, 17), fill=0)
    draw.text((90, 1850), "² The two diameters were 50 and 120 micrometres.", font=font(SERIF, 17), fill=0)
    draw.text((W / 2 - 12, 1960), "43", font=font(SERIF, 20), fill=0)
    save(image, "article-two-columns.png", "A two-column article page with an abstract, a figure and footnotes",
         "a running head with page range, a title, authors with affiliation marks, a boxed abstract, numbered "
         "section headings and justified paragraphs in two columns, a drawn chart with a caption, two footnotes, "
         "a page number")


BULLETIN = [
    ("Harbour wall repairs to finish in May",
     "The council expects the last section of the harbour wall to be finished by the end of May, three weeks later "
     "than planned. The delay came from the storm in February, which washed away part of the temporary platform and "
     "a load of stone that was waiting to be placed. The footpath along the wall will stay closed until the work is "
     "done, and the ferry will keep using the north landing."),
    ("New hours at the library",
     "From the first of April the library will open at nine on weekdays and close at seven, an hour later than now. "
     "On Saturdays it will open from ten until four. The reading room upstairs will be available for groups who book "
     "it at least two days ahead, and the local history collection can now be searched from the terminals by the "
     "entrance as well as at the desk."),
    ("School choir wins regional prize",
     "Forty pupils from the upper school travelled to the regional festival last weekend and came back with first "
     "prize in the youth category. They sang three pieces, one of them written for the choir by its director. The "
     "judges praised the balance between the voices and the clarity of the words, and the choir has been invited to "
     "sing at the opening of the summer fair."),
    ("Allotments: the waiting list reopens",
     "Residents who would like an allotment can add their names to the waiting list again from Monday. Twelve plots "
     "will become free in the autumn, when the new ground behind the sports field is divided. Applications are made "
     "on the form at the town hall, and the plots will be offered in the order the forms arrive."),
    ("Bus route 14 changes",
     "Route 14 will no longer run along Mill Street after the bridge there closes for inspection. Buses will use "
     "Station Road instead and stop outside the post office. The timetable stays the same, though the journey from "
     "the hospital to the market square may take two or three minutes longer at busy times."),
    ("Volunteers wanted for the beach clean",
     "The spring beach clean takes place on the second Sunday of the month, starting from the lifeboat station at "
     "ten. Gloves, bags and litter pickers are provided, and hot drinks will be served at noon. Last year more than "
     "ninety people took part and collected over four hundred kilograms of rubbish in two hours."),
]


def bulletin():
    image, draw = new_page()
    name = "THE WESTERLY BULLETIN"
    tw = draw.textlength(name, font=font(SERIFB, 64))
    draw.text(((W - tw) / 2, 60), name, font=font(SERIFB, 64), fill=0)
    draw.line([(70, 150), (W - 70, 150)], fill=0, width=3)
    draw.text((70, 160), "No. 318", font=font(SERIF, 20), fill=0)
    date = "Thursday, the twelfth of March"
    draw.text(((W - draw.textlength(date, font=font(SERIF, 20))) / 2, 160), date, font=font(SERIF, 20), fill=0)
    draw.text((W - 70 - draw.textlength("Free", font=font(SERIF, 20)), 160), "Free", font=font(SERIF, 20), fill=0)
    draw.line([(70, 192), (W - 70, 192)], fill=0, width=1)
    col_w, gap = 404, 28
    xs = [70 + i * (col_w + gap) for i in range(3)]
    tops = [220, 220, 220]
    for i, (head, text) in enumerate(BULLETIN):
        col = i % 3
        x, y = xs[col], tops[col]
        for line in wrap(draw, head, font(SERIFB, 27), col_w):
            draw.text((x, y), line, font=font(SERIFB, 27), fill=0)
            y += 34
        y = paragraph(draw, x, y + 8, text, font(SERIF, 19), col_w, 27) + 40
        tops[col] = y
    for i in range(1, 3):
        x = xs[i] - gap / 2
        draw.line([(x, 220), (x, max(tops) - 30)], fill=0, width=1)
    save(image, "bulletin-three-columns.png", "A three-column local news page",
         "a masthead, an issue line with number and date, six headlines and justified stories in three columns "
         "divided by rules")


# ── Accents and other scripts ───────────────────────────────────────────────────────────────────────────────
LATIN = [
    ("Deutsch", "Die Fähre über den Fluss fährt im Winter nur zweimal täglich; wer später ankommt, muss über die "
     "Brücke im Süden ausweichen, was fast eine Stunde länger dauert. Größere Fahrzeuge dürfen die Fähre gar nicht "
     "benutzen."),
    ("Français", "Le marché couvert ouvrira de nouveau ses portes après les travaux de l'été : les étals du "
     "rez-de-chaussée garderont leur place, mais les fromagers s'installeront à l'étage, près des fenêtres côté "
     "rivière. Où se garer ? Le parking de la gare reste gratuit le dimanche."),
    ("Español", "¿Quién se llevó la escalera del almacén? Hacía falta para cambiar las bombillas del pasillo, y "
     "ahora nadie sabe dónde está. Si alguien la encuentra, que la devuelva antes del miércoles, por favor; el "
     "electricista vendrá a las ocho."),
    ("Polski", "Pociąg do Gdańska odjeżdża z peronu trzeciego o godzinie ósmej czterdzieści. Bagaż większy niż "
     "walizka należy oddać w przechowalni, a rowery można przewozić tylko w ostatnim wagonie, który jest oznaczony "
     "żółtą naklejką."),
    ("Čeština", "Večer se nad údolím zvedla mlha a cesta k chatě byla vidět jen na několik kroků. Řidič zpomalil, "
     "zapnul světla do mlhy a za mostem odbočil doprava, přesně podle mapy, kterou mu ráno nakreslila paní "
     "Dvořáková."),
    ("Português", "A biblioteca da vila recebeu esta semana duzentos livros novos, doados por uma família que se "
     "mudou para o estrangeiro. Há romances, atlas e até uma coleção de revistas de jardinagem dos anos noventa, "
     "que já despertou a atenção de muitos leitores."),
    ("Türkçe", "Çarşamba günü sabah erkenden yola çıktık; öğleye doğru dağ köyüne vardığımızda köylüler bizi sıcak "
     "çay ve taze ekmekle karşıladı. Dönüş yolunda yağmur başladı, ama şoförümüz yolu çok iyi biliyordu."),
    ("Tiếng Việt", "Sáng nay trời mưa nhẹ, nhưng chợ vẫn đông người. Bà bán rau ở góc đường nói rằng năm nay mùa "
     "màng tốt hơn năm ngoái, và giá cà chua đã giảm gần một nửa so với tháng trước."),
]
SCRIPTS = [
    ("Ελληνικά", "Το πλοίο για τα νησιά αναχωρεί κάθε πρωί στις επτά και μισή. Οι επιβάτες με αυτοκίνητο πρέπει να "
     "βρίσκονται στο λιμάνι τουλάχιστον μία ώρα νωρίτερα, γιατί ο χώρος στο γκαράζ είναι περιορισμένος."),
    ("Русский", "Вечером над рекой поднялся туман, и рыбаки вернулись на берег раньше обычного. Старый паром ещё "
     "ходил, но капитан предупредил, что последний рейс сегодня будет в девять часов."),
    ("Українська", "Бібліотека на розі вулиці знову відкрилася після ремонту. Тепер у читальній залі є великі вікна, "
     "нові полиці та окрема кімната для дітей, де щосуботи читають казки вголос."),
    ("Српски", "Воз за Нови Сад касни двадесет минута због радова на прузи. Путници који путују даље могу да "
     "сачекају на перону број два или у чекаоници поред благајне."),
    ("Български", "Пазарът в старата част на града работи всеки ден освен понеделник. Най-рано сутрин идват "
     "градинарите от селата с плодове, мед и прясно мляко."),
    ("Հայերեն", "Գյուղի դպրոցը նոր գրադարան ունի, որտեղ երեխաները կարող են գրքեր վերցնել տուն և վերադարձնել "
     "մեկ շաբաթից։"),
    ("ქართული", "ზამთარში მთის გზა ხშირად იკეტება, ამიტომ მგზავრები ქალაქში ადრე ჩადიან და იქ ელოდებიან ამინდის "
     "გამოკეთებას."),
]


def languages(name, title, entries, contents, f_name, fb_name):
    image, draw = new_page()
    y = 90
    for language, text in entries:
        draw.text((90, y), language, font=font(fb_name, 26), fill=0)
        y = paragraph(draw, 90, y + 40, text, font(f_name, 23), 1268, 34) + 30
    save(image, name, title, contents)


# ── A form, code, chemistry and small print ─────────────────────────────────────────────────────────────────
def invoice():
    image, draw = new_page()
    draw.text((90, 80), "Brackwater Joinery", font=font(SERIFB, 40), fill=0)
    for i, line in enumerate(["Unit 4, Tannery Yard", "Hollin, HL7 2QX", "accounts@brackwater.example"]):
        draw.text((90, 136 + i * 30), line, font=font(SANS, 20), fill=0)
    draw.text((W - 90 - draw.textlength("INVOICE", font=font(SANSB, 46)), 80), "INVOICE", font=font(SANSB, 46), fill=0)
    fields = [("Invoice no.", "BJ-2026-0417"), ("Date", "14/03/2026"), ("Due", "13/04/2026"), ("Order ref.", "PO 55-118")]
    for i, (k, v) in enumerate(fields):
        draw.text((930, 150 + i * 32), k, font=font(SANSB, 20), fill=0)
        draw.text((1110, 150 + i * 32), v, font=font(SANS, 20), fill=0)
    draw.text((90, 300), "Bill to", font=font(SANSB, 22), fill=0)
    for i, line in enumerate(["Ms R. Okonkwo-Hale", "The Old Granary, Mill Lane", "Ferrish, FE3 9LP"]):
        draw.text((90, 334 + i * 30), line, font=font(SANS, 20), fill=0)
    draw.text((760, 300), "Deliver to", font=font(SANSB, 22), fill=0)
    draw.text((760, 334), "[x] Same as billing address", font=font(SANS, 20), fill=0)
    draw.text((760, 364), "[ ] Collect from workshop", font=font(SANS, 20), fill=0)
    items = [("1", "Oak window sill, 1200 x 180 x 30 mm", "2", "86.00"),
             ("2", "Sash cords, waxed cotton, 10 m", "4", "7.50"),
             ("3", "Pine skirting board, ogee, 2.4 m", "11", "14.20"),
             ("4", "Brass sash lifts (pair)", "3", "18.90"),
             ("5", "Labour: fitting and finishing (hours)", "6.5", "42.00"),
             ("6", "Delivery within 15 miles", "1", "25.00"),
             ("7", "Discount on items 1 to 3 (5%)", "1", "-15.42")]
    rows, subtotal = [], 0.0
    for n, desc, qty, price in items:
        amount = float(qty) * float(price)
        subtotal += amount
        rows.append([n, desc, qty, price, "%.2f" % amount])
    y = ruled_table(draw, 90, 470, [("#", 60), ("Description", 650), ("Qty", 120), ("Unit price", 210),
                                    ("Amount", 228)], rows, font(SANS, 20), font(SANSB, 20), 52,
                    aligns=["l", "l", "r", "r", "r"])
    vat = round(subtotal * 0.2, 2)
    for i, (k, v) in enumerate([("Subtotal", subtotal), ("VAT at 20%", vat), ("Total due", subtotal + vat)]):
        f = font(SANSB if k == "Total due" else SANS, 22)
        draw.text((900, y + 24 + i * 38), k, font=f, fill=0)
        text = "%.2f" % v
        draw.text((1358 - draw.textlength(text, font=f), y + 24 + i * 38), text, font=f, fill=0)
    y += 180
    draw.text((90, y), "Payment", font=font(SANSB, 22), fill=0)
    paragraph(draw, 90, y + 34, "Please pay by bank transfer within 30 days, quoting the invoice number. Sort code "
              "00-11-22, account 12345678 (example details). Goods remain our property until paid for in full. "
              "Questions about this invoice: call 01632 960 418 between 8:00 and 16:30, Monday to Friday.",
              font(SANS, 19), 1268, 28)
    save(image, "invoice-form.png", "An invoice with fields, check boxes, line items and totals",
         "a letterhead, an invoice title, labelled fields, two address blocks, two check boxes, a ruled table of "
         "seven line items with quantities and amounts, totals, a paragraph of payment terms")


CODE = '''import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    due: float
    name: str = field(compare=False)
    tags: set[str] = field(default_factory=set, compare=False)


def schedule(tasks: list[Task], budget: float) -> list[str]:
    """Names of the tasks done in order of due time, while the budget lasts."""
    heap = list(tasks)
    heapq.heapify(heap)
    done, spent = [], 0.0
    while heap and spent < budget:
        task = heapq.heappop(heap)
        cost = 1.5 if "slow" in task.tags else 1.0
        if spent + cost > budget:
            break
        spent += cost
        done.append(task.name)
    return done


if __name__ == "__main__":
    jobs = [Task(3.0, "wash", {"slow"}), Task(1.0, "dry"),
            Task(2.5, "fold", {"slow", "late"})]
    print(schedule(jobs, budget=2.5))   # ['dry', 'fold']
    print({t.name: sorted(t.tags) for t in jobs if t.tags})'''


def code():
    image, draw = new_page()
    draw.text((90, 80), "Listing 4.2: a small scheduler (Python 3.12)", font=font(SERIFB, 30), fill=0)
    lines = CODE.split("\n")
    draw.rectangle([(80, 140), (W - 80, 140 + len(lines) * 34 + 30)], outline=0, width=2)
    y = 160
    for number, line in enumerate(lines, 1):
        draw.text((100, y), "%2d" % number, font=font(MONO, 22), fill=110)
        draw.text((170, y), line, font=font(MONO, 22), fill=0)
        y += 34
    save(image, "code-listing.png", "A numbered Python listing",
         "a caption and a boxed listing of %d numbered lines in a monospaced face (indentation, brackets, "
         "decorators, type hints, a string, comments)" % len(lines))


def chemistry():
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    y = 110
    fig.text(110 / W, 1 - y / H, "Worked reactions", fontsize=30, family="DejaVu Serif", weight="bold", va="top")
    y += 100
    reactions = [
        ("Complete combustion of methane:", r"$\mathrm{CH_{4}+2\,O_{2}\longrightarrow CO_{2}+2\,H_{2}O}$"),
        ("Synthesis of ammonia, at equilibrium:", r"$\mathrm{N_{2}+3\,H_{2}\rightleftharpoons 2\,NH_{3}}$"),
        ("Neutralisation:", r"$\mathrm{H_{2}SO_{4}+2\,NaOH\longrightarrow Na_{2}SO_{4}+2\,H_{2}O}$"),
        ("Photosynthesis, overall:", r"$\mathrm{6\,CO_{2}+6\,H_{2}O\longrightarrow C_{6}H_{12}O_{6}+6\,O_{2}}$"),
        ("Dissolving calcium carbonate in acid:",
         r"$\mathrm{CaCO_{3}(s)+2\,HCl(aq)\longrightarrow CaCl_{2}(aq)+H_{2}O(l)+CO_{2}(g)}$"),
        ("The equilibrium constant for a general reaction:",
         r"$K_{c}=\frac{[\mathrm{C}]^{c}\,[\mathrm{D}]^{d}}{[\mathrm{A}]^{a}\,[\mathrm{B}]^{b}}$"),
        ("The Henderson–Hasselbalch equation:", r"$\mathrm{pH}=\mathrm{p}K_{a}+\log_{10}\frac{[\mathrm{A^{-}}]}"
                                                r"{[\mathrm{HA}]}$"),
        ("An ion with its charge:", r"$\mathrm{Fe^{3+}+3\,OH^{-}\longrightarrow Fe(OH)_{3}\downarrow}$")]
    for number, (text, formula) in enumerate(reactions, 1):
        fig.text(110 / W, 1 - y / H, "%d. %s" % (number, text), fontsize=17, family="DejaVu Serif", va="top")
        y += 60
        fig.text(0.5, 1 - y / H, formula, fontsize=23, ha="center", va="top")
        y += 125
    fig.text(110 / W, 1 - (y + 10) / H, "Square brackets are concentrations in moles per litre.", fontsize=16,
             family="DejaVu Serif", style="italic", va="top")
    path = os.path.join(OUT, "chemistry-reactions.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    image = Image.open(path).convert("L")
    save(image, "chemistry-reactions.png", "Eight chemical equations with subscripts, charges and states",
         "a heading, eight numbered lines each with a centred equation (subscripts, coefficients, state symbols, "
         "an equilibrium arrow, ionic charges, a fraction of concentrations), a closing note",
         tool="matplotlib %s (mathtext, dejavuserif)" % matplotlib.__version__, fonts="DejaVu Serif")


TERMS = [
    "1. These terms apply to every booking made through the hall's office, by telephone or in person. A booking is "
    "confirmed only when the deposit has been received and a written confirmation has been sent to the hirer.",
    "2. The hirer must be at least eighteen years old and must be present for the whole of the booking. The hirer "
    "is responsible for the behaviour of every guest and for leaving the hall in the condition in which it was "
    "found, with chairs and tables returned to the store room and all rubbish removed.",
    "3. The deposit is returned within fourteen days after the booking, less the cost of any cleaning, repair or "
    "replacement that the hall's caretaker considers necessary. Where that cost exceeds the deposit, the hirer will "
    "be invoiced for the difference and must pay it within thirty days.",
    "4. Bookings cancelled more than twenty-eight days before the date receive a full refund of the deposit and of "
    "any fee paid. Bookings cancelled between twenty-eight and seven days before receive half of the fee; later "
    "cancellations receive no refund of the fee, though the deposit is returned in full.",
    "5. Music must end by eleven in the evening and the hall must be empty by half past eleven. Amplified music is "
    "allowed only in the main room, at a volume that cannot be heard in the houses on the far side of the green.",
    "6. Food may be served in the main room and the kitchen only. The kitchen's ovens and dishwasher may be used by "
    "the hirer after the caretaker has shown how they work; the hirer must not use any other appliance brought in "
    "from outside without written permission.",
    "7. Decorations must not be fixed to the walls, the ceiling or the woodwork with nails, pins, tape or adhesive "
    "of any kind. Free-standing decorations are welcome. Candles and other open flames are not allowed anywhere in "
    "the building or in the garden.",
    "8. The hall holds public liability insurance for its own activities only. Hirers running an event for paying "
    "guests, or bringing in performers, caterers or equipment hire firms, must hold their own insurance and show "
    "the certificate to the office at least seven days before the booking.",
    "9. The office may cancel a booking if the hall becomes unusable through fire, flood, a failure of the heating "
    "in winter or any other cause outside its control. In that case the deposit and the fee are refunded in full, "
    "and the office has no further liability to the hirer.",
]


def small_print():
    image, draw = new_page()
    draw.text((90, 80), "Conditions of hire: Ferrish Village Hall", font=font(SERIFB, 32), fill=0)
    y = 150
    for clause in TERMS:
        y = paragraph(draw, 90, y, clause, font(SERIF, 18), 1268, 25) + 14
    draw.text((90, y + 10), "Signed for the hirer: ______________________   Date: ___ / ___ / ______",
              font=font(SERIF, 18), fill=0)
    save(image, "small-print-terms.png", "Nine numbered clauses of small print with a signature line",
         "a heading, nine numbered justified clauses in a small face, a signature and date line")


quarterly()
measurements()
timetable()
article()
bulletin()
languages("languages-latin.png", "Paragraphs in eight languages written in the Latin alphabet", LATIN,
          "eight language headings, each with a justified paragraph (German, French, Spanish, Polish, Czech, "
          "Portuguese, Turkish, Vietnamese: their accents, hooks, cedillas and stacked marks)", SERIF, SERIFB)
languages("languages-scripts.png", "Paragraphs in Greek, four Cyrillic languages, Armenian and Georgian", SCRIPTS,
          "seven language headings, each with a justified paragraph (Greek; Russian, Ukrainian, Serbian and "
          "Bulgarian in Cyrillic; Armenian; Georgian)", SANS, SANSB)
invoice()
code()
chemistry()
small_print()
with open(os.path.join(OUT, "pages.json"), "w", encoding="utf-8", newline="\n") as handle:
    json.dump(PAGES, handle, ensure_ascii=False, indent=1)
print(len(PAGES), "pages")
for page in PAGES:
    size = os.path.getsize(os.path.join(OUT, page["file"]))
    print("%-30s %7d bytes" % (page["file"], size))
