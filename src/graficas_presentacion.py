"""Gráficas de la presentación "Cada rector reescribe la universidad".

Rehace los cálculos del notebook 04 con un estilo para diapositivas: fondo del deck,
texto grande, una idea por gráfica, contexto en gris y un solo color de acento.

    python src/graficas_presentacion.py

Guarda reports/figures/slides_*.png e imprime las cifras que citan las diapositivas.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning)

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "processed" / "noticias_base.parquet"
FIGURAS = RAIZ / "reports" / "figures"

# Paleta del deck
FONDO, TINTA, GRIS, GRIS_CLARO = "#F6F3EE", "#1B2A3A", "#9AA3AB", "#D9D4CC"
ACENTO, AZUL = "#C8553D", "#2F6690"
RECTORES = {
    "Grijalva": "heriberto grijalva",
    "Velázquez": "velázquez contreras",
    "Plancarte": "rita plancarte",
    "Camarena": "dena maría camarena|dena camarena",
}

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 20, "axes.facecolor": FONDO, "figure.facecolor": FONDO,
    "axes.edgecolor": GRIS_CLARO, "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def guardar(fig, nombre):
    fig.savefig(FIGURAS / f"slides_{nombre}.png", dpi=100, facecolor=FONDO)
    plt.close(fig)


def cargar():
    columnas = ["fecha", "titulo", "texto_limpio", "lemas_frases", "ent_texto", "ent_tipo", "firma", "es_plantilla"]
    df = pd.read_parquet(ENTRADA, columns=columnas)
    df = df.loc[~df["es_plantilla"]].reset_index(drop=True)
    df["texto_min"] = df["texto_limpio"].str.lower()
    df["trimestre"] = df["fecha"].dt.to_period("Q")

    inicio = {n: df.loc[df["texto_min"].str.contains(rf"\brectora? (?:\w+ ){{0,2}}(?:{p})"), "fecha"].min()
              for n, p in RECTORES.items()}
    inicio["Grijalva"] = df["fecha"].min()
    transiciones = pd.Series(inicio).sort_values()
    df["rectorado"] = pd.cut(df["fecha"], bins=[*transiciones, pd.Timestamp.max],
                             labels=transiciones.index, right=False).astype(str)

    en_personas = Counter(w for ents, tipos in zip(df["ent_texto"], df["ent_tipo"])
                          for e, t in zip(ents, tipos) if t == "PER" for w in e.lower().split())
    en_corpus = Counter(w for doc in df["lemas_frases"] for tok in doc for w in tok.split("_"))
    nombres = {w for w, c in en_personas.items() if en_corpus.get(w, 0) and c / en_corpus[w] >= 0.3}
    df["tokens"] = [[t for t in d if not any(p in nombres for p in t.split("_"))] for d in df["lemas_frases"]]
    return df, transiciones


def lineas_transicion(ax, transiciones, ymax, etiquetas=True):
    for nombre, fecha in transiciones.iloc[1:].items():
        ax.axvline(fecha, color=GRIS, ls=(0, (4, 4)), lw=1.5, zorder=0)
        if etiquetas:
            ax.text(fecha, ymax, f" llega {nombre}", color=GRIS, fontsize=17, va="top")


# --- Lenguaje inclusivo ------------------------------------------------------------

DESDOBLAMIENTO = (r"\b(?:las y los|los y las|niñas y niños|niños y niñas|todas y todos|todos y todas|alumnas y alumnos|"
                  r"académicas y académicos|investigadoras e investigadores|maestras y maestros|profesoras y profesores|"
                  r"estudiantes mujeres y hombres|egresadas y egresados)\b")
NEUTRO = r"\b(?:estudiantado|alumnado|profesorado|personal académico|comunidad estudiantil|personal docente)\b"


def inclusivo(df, transiciones):
    df["desdoblamiento"] = df["texto_min"].str.contains(DESDOBLAMIENTO)
    df["inclusivo"] = df["desdoblamiento"] | df["texto_min"].str.contains(NEUTRO)

    anual = df.groupby(df["fecha"].dt.year)[["desdoblamiento", "inclusivo"]].mean().mul(100)
    print("Desdoblamiento por año (%):", anual["desdoblamiento"].round(1).to_dict())
    print("Inclusivo (cualquiera) por año (%):", anual["inclusivo"].round(1).to_dict())

    tri = df.groupby("trimestre")["inclusivo"].mean().mul(100)
    tri.index = tri.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.fill_between(tri.index, tri, color=ACENTO, alpha=0.12, lw=0)
    ax.plot(tri.index, tri, color=ACENTO, lw=3.5)
    lineas_transicion(ax, transiciones, 66)
    ax.set_ylim(0, 66)
    ax.set_yticks([0, 10, 20, 30, 40, 50], [f"{v}%" for v in [0, 10, 20, 30, 40, 50]])
    ax.annotate(f"2013–2020: ~5% de las notas", (pd.Timestamp("2015-06-01"), 6), xytext=(pd.Timestamp("2014-01-01"), 20),
                fontsize=19, color=TINTA, arrowprops=dict(arrowstyle="-", color=GRIS))
    ax.text(pd.Timestamp("2025-02-01"), 52, f"2026:\n{anual.loc[2026, 'inclusivo']:.0f}% de\nlas notas",
            fontsize=21, color=ACENTO, fontweight="bold", ha="right", va="top")
    ax.set_ylabel("% de notas con lenguaje inclusivo")
    fig.tight_layout()
    guardar(fig, "inclusivo")

    # Por firma: las cinco principales en cuatro periodos
    cortes = pd.to_datetime(["2013-01-01", "2021-06-01", "2025-06-01", "2026-01-01", "2027-01-01"])
    periodos = ["2013–2021", "2021–2025", "2.º sem. 2025", "2026"]
    df["periodo"] = pd.cut(df["fecha"], cortes, labels=periodos, right=False)
    g = df.dropna(subset=["firma"]).groupby(["firma", "periodo"], observed=True)["inclusivo"].agg(["mean", "size"])
    tabla, n = g["mean"].unstack().mul(100), g["size"].unstack()
    tabla = tabla.loc[(n[["2021–2025", "2026"]] >= 15).all(axis=1)].sort_values("2026", ascending=False)
    print("Inclusivo por firma (%):\n", tabla.round(0).to_string())

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(periodos))
    y_etiqueta = []
    for firma, fila in tabla.iterrows():
        ax.plot(x, fila.values, color=ACENTO, lw=3, marker="o", ms=10, alpha=0.85)
        # Separa las etiquetas que quedarían encimadas
        y = fila.iloc[-1] if not y_etiqueta else min(fila.iloc[-1], y_etiqueta[-1] - 4.5)
        y_etiqueta.append(y)
        ax.text(x[-1] + 0.08, y, f"{firma.split()[0]} {firma.split()[1]}  {fila.iloc[-1]:.0f}%",
                va="center", fontsize=18, color=TINTA)
    ax.set_xticks(x, periodos)
    ax.set_xlim(-0.2, len(periodos) + 0.6)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    ax.set_ylabel("% de sus notas con lenguaje inclusivo")
    fig.tight_layout()
    guardar(fig, "firmas")
    return tri


# --- Verbos para citar ----------------------------------------------------------------

VERBOS = {
    "neutro": ["dijo", "señaló", "comentó", "indicó", "mencionó", "agregó", "añadió", "apuntó", "refirió", "afirmó", "sostuvo", "expresó"],
    "valorativo": ["destacó", "subrayó", "resaltó", "enfatizó", "recalcó", "puntualizó", "reconoció", "aseveró", "advirtió"],
    "explicativo": ["explicó", "detalló", "precisó", "compartió", "especificó", "abundó"],
}


def verbos(df):
    tipo = {v: t for t, vs in VERBOS.items() for v in vs}
    patron = r"\b(" + "|".join(tipo) + r")\b"
    citas = df[["fecha", "texto_min"]].assign(verbo=df["texto_min"].str.findall(patron)).explode("verbo").dropna(subset=["verbo"])
    anio = citas["fecha"].dt.year
    dijo = (citas["verbo"] == "dijo").groupby(anio).mean().mul(100)
    destaco = (citas["verbo"] == "destacó").groupby(anio).mean().mul(100)
    print('"dijo" por año (%):', dijo.round(1).to_dict())
    print('"destacó" por año (%):', destaco.round(1).to_dict())

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(dijo.index, dijo, color=GRIS, lw=3.5, marker="o", ms=8)
    ax.plot(destaco.index, destaco, color=ACENTO, lw=3.5, marker="o", ms=8)
    ax.text(dijo.index[0], dijo.iloc[0] + 1.2, f"«dijo»  {dijo.iloc[0]:.0f}%", color=TINTA, fontsize=21, fontweight="bold")
    ax.text(dijo.index[-1] - 0.2, dijo.iloc[-1] - 2.3, f"{dijo.iloc[-1]:.0f}%", color=TINTA, fontsize=21, fontweight="bold", ha="right")
    ax.text(destaco.index[0], destaco.iloc[0] - 3.2, f"«destacó»  {destaco.iloc[0]:.0f}%", color=ACENTO, fontsize=21, fontweight="bold")
    ax.text(destaco.index[-1] - 0.2, destaco.iloc[-1] + 1.2, f"{destaco.iloc[-1]:.0f}%", color=ACENTO, fontsize=21, fontweight="bold", ha="right")
    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"])
    ax.set_xticks(range(2013, 2027, 2))
    ax.set_ylabel("% de todas las citas")
    fig.tight_layout()
    guardar(fig, "verbos")


# --- Protagonismo ---------------------------------------------------------------------

def protagonismo(df, orden):
    rector = r"\brectora?\b"
    t = df.groupby("rectorado")[["texto_min", "titulo"]].agg(
        cuerpo=("texto_min", lambda s: s.str.contains(rector).mean() * 100),
        titulo=("titulo", lambda s: s.str.lower().str.contains(rector).mean() * 100)).loc[orden]
    print("Protagonismo (%):\n", t.round(1).to_string())

    fig, ax = plt.subplots(figsize=(16, 6.5))
    x = np.arange(len(orden))
    b1 = ax.bar(x - 0.2, t["cuerpo"], 0.38, color=[GRIS] * 3 + [ACENTO], label="lo menciona en el texto")
    b2 = ax.bar(x + 0.2, t["titulo"], 0.38, color=GRIS_CLARO, label="lo pone en el titular")
    for barras, fmt in [(b1, "{:.1f}%"), (b2, "{:.1f}%")]:
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, fmt.format(b.get_height()), ha="center", fontsize=19)
    ax.set_xticks(x, orden, fontsize=22)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=19)
    ax.set_ylim(0, 22)
    fig.tight_layout()
    guardar(fig, "protagonismo")


# --- Temas ----------------------------------------------------------------------------

ETIQUETAS = [
    "Emprendimiento e innovación", "Música", "Economía y política nacional", "Salud",
    "Investigación y posgrado", "Congresos y foros", "Concursos y olimpiadas", "Admisión e inscripciones",
    "Gestión y vinculación", "Testimonios y vida personal", "Deporte", "Campus Navojoa (sur)",
    "Cursos y capacitación", "Ceremonias y graduaciones", "Libros, teatro y arte", "Sorteos",
    "Campus Caborca y Nogales (norte)", "Género y derechos humanos", "Evaluación y acreditación",
    "Alimentos, agua y ambiente",
]


def temas(df):
    X = TfidfVectorizer(analyzer=lambda d: d, min_df=20, max_df=0.4, sublinear_tf=True).fit_transform(df["tokens"])
    W = NMF(n_components=20, init="nndsvda", random_state=0, max_iter=1000).fit_transform(X)
    W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    prop = pd.DataFrame(W, columns=ETIQUETAS)
    relativo = prop.groupby(df["rectorado"].values).mean() / prop.mean()
    print("Peso relativo de temas:\n", relativo.T.round(2).to_string())

    # Los ocho temas que más cambian entre el primer rectorado y el actual (los mismos de la página web)
    suben = ["Congresos y foros", "Gestión y vinculación", "Salud", "Ceremonias y graduaciones", "Investigación y posgrado"]
    bajan = ["Economía y política nacional", "Cursos y capacitación", "Libros, teatro y arte"]
    filas = bajan[::-1] + suben[::-1]
    fig, ax = plt.subplots(figsize=(16, 7.2))
    for i, tema in enumerate(filas):
        a, b = relativo.loc["Grijalva", tema], relativo.loc["Camarena", tema]
        color = ACENTO if tema in suben else AZUL
        ax.plot([a, b], [i, i], color=GRIS_CLARO, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [i], s=220, color=GRIS, zorder=2)
        ax.scatter([b], [i], s=260, color=color, zorder=3)
        ax.text(b + (0.03 if b > a else -0.03), i, f"{b:.2f}", va="center", ha="left" if b > a else "right",
                fontsize=19, color=color, fontweight="bold")
    ax.axvline(1, color=GRIS, ls=(0, (4, 4)), lw=1.5)
    ax.set_yticks(range(len(filas)), filas, fontsize=21)
    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([0.5, 0.75, 1, 1.25, 1.5], ["0.5×", "0.75×", "promedio", "1.25×", "1.5×"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=200, color=GRIS, label="Grijalva (2013–2017)")
    ax.scatter([], [], s=200, color=ACENTO, label="Camarena (2025–): sube")
    ax.scatter([], [], s=200, color=AZUL, label="Camarena (2025–): baja")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncols=3, fontsize=19)
    fig.tight_layout()
    guardar(fig, "temas")


# --- Cortes de vocabulario -------------------------------------------------------------

def calcular_cortes(df, tri_inclusivo, K=90, R=20):
    """Cambio trimestral del vocabulario (JS igualada menos ruido) y del lenguaje inclusivo, en desviaciones típicas."""
    vocab =[w for w, _ in Counter(w for d in df["tokens"] for w in d).most_common(5000)]
    indice = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(0)

    def dist(docs):
        v = np.zeros(len(vocab))
        for d in docs:
            for w in d:
                j = indice.get(w)
                if j is not None:
                    v[j] += 1
        return v / v.sum()

    grupos = {}
    for q, g in df.groupby("trimestre")["tokens"]:
        arr = np.empty(len(g), dtype=object)
        arr[:] = list(g)
        grupos[q] = arr
    qs = sorted(q for q in grupos if len(grupos[q]) >= 2 * K)

    def exceso(a, b):
        js, ruido = [], []
        for _ in range(R):
            ia, ib = rng.permutation(len(a)), rng.permutation(len(b))
            da, db = dist(a[ia[:K]]), dist(b[ib[:K]])
            js.append(jensenshannon(da, db))
            ruido.append((jensenshannon(da, dist(a[ia[K:2 * K]])) + jensenshannon(db, dist(b[ib[K:2 * K]]))) / 2)
        return np.mean(js) - np.mean(ruido)

    serie = pd.Series({b.to_timestamp(): exceso(grupos[a], grupos[b]) for a, b in zip(qs, qs[1:])})
    vocab_z = (serie - serie.median()) / serie.std()
    salto_inc = tri_inclusivo.diff().dropna()
    inc_z = (salto_inc - salto_inc.median()) / salto_inc.std()
    return vocab_z, inc_z


def cortes(df, transiciones, tri_inclusivo):
    vocab_z, inc_z = calcular_cortes(df, tri_inclusivo)
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.6), sharex=True)
    for ax, z, color, titulo in [(axes[0], vocab_z, AZUL, "De qué se habla: cambio del vocabulario"),
                                 (axes[1], inc_z, ACENTO, "Cómo se habla: cambio del lenguaje inclusivo")]:
        ax.bar(z.index, z.values, width=70, color=np.where(z.values > 3, color, GRIS_CLARO))
        ax.axhline(3, color=GRIS, lw=1, ls=":")
        lineas_transicion(ax, transiciones, 7.5, etiquetas=ax is axes[0])
        ax.set_ylim(-2.5, 7.5)
        ax.set_yticks([0, 3, 6])
        ax.text(0.01, 0.92, titulo, transform=ax.transAxes, fontsize=20, color=TINTA, fontweight="bold", va="top")
    axes[1].set_ylabel("cambio trimestral\n(desviaciones típicas)", fontsize=17)
    axes[0].set_ylabel("cambio trimestral\n(desviaciones típicas)", fontsize=17)
    fig.tight_layout()
    guardar(fig, "cortes")
    print("Vocabulario: trimestres > 3 DE:", [f"{t:%Y-%m}" for t in vocab_z[vocab_z > 3].index])
    print("Inclusivo: trimestres > 3 DE:", [f"{t:%Y-%m}" for t in inc_z[inc_z > 3].index])


def main():
    df, transiciones = cargar()
    orden = list(transiciones.index)
    print("Inicio de cada rectorado:", {k: f"{v:%Y-%m-%d}" for k, v in transiciones.items()})
    print("Notas por rectorado:", df["rectorado"].value_counts().loc[orden].to_dict())
    tri = inclusivo(df, transiciones)
    verbos(df)
    protagonismo(df, orden)
    temas(df)
    cortes(df, transiciones, tri)


if __name__ == "__main__":
    main()
