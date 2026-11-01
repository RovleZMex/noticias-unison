"""Exporta los datos de la historia y arma la página web de scrollytelling.

    python src/exportar_web.py                 # recalcula los datos y arma la página
    python src/exportar_web.py --solo-armar    # solo vuelve a armar la página con web/datos.json

1. Calcula las series (las mismas del notebook 04 y de graficas_presentacion.py).
2. Las guarda en web/datos.json.
3. Inserta ese JSON en web/plantilla.html y escribe web/index.html, la página final.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from graficas_presentacion import DESDOBLAMIENTO, ETIQUETAS, NEUTRO, VERBOS, calcular_cortes, cargar

RAIZ = Path(__file__).resolve().parent.parent
WEB = RAIZ / "web"
MARCA = "/*__DATOS__*/null"

# Ejemplos reales: la misma frase de reportero, con 13 años de distancia
EJEMPLOS = [
    {"id": 14739, "anio": 2013, "frase": "Asimismo, invitó a los estudiantes a aprovechar los conocimientos y experiencias que durante cuatro días compartirán con ellos los conferencistas invitados."},
    {"id": 38578, "anio": 2026, "frase": "Asimismo, invitó a las y los estudiantes a aprovechar al máximo esta experiencia de formación e investigación."},
]


def log_odds(docs_grupo, docs_resto, min_notas=30, alfa=0.01):
    # Log-odds con prior de Dirichlet informativo (Monroe et al., 2008), sobre frecuencia de notas
    y_i = Counter(w for d in docs_grupo for w in set(d))
    y_j = Counter(w for d in docs_resto for w in set(d))
    total = y_i + y_j
    a0 = alfa * sum(total.values())
    n_i, n_j = sum(y_i.values()), sum(y_j.values())
    filas = []
    for w, c in total.items():
        if c < min_notas:
            continue
        a = alfa * c
        d = np.log((y_i[w] + a) / (n_i + a0 - y_i[w] - a)) - np.log((y_j[w] + a) / (n_j + a0 - y_j[w] - a))
        filas.append((w, d / np.sqrt(1 / (y_i[w] + a) + 1 / (y_j[w] + a))))
    return sorted(filas, key=lambda x: -x[1])


def main():
    df, transiciones = cargar()
    orden = list(transiciones.index)
    df = df.sort_values("fecha", kind="stable").reset_index(drop=True)

    # Banderas por nota: 1 = lenguaje inclusivo, 2 = menciona al rector, 4 = menciona la pandemia
    df["inclusivo"] = df["texto_min"].str.contains(DESDOBLAMIENTO) | df["texto_min"].str.contains(NEUTRO)
    df["rector"] = df["texto_min"].str.contains(r"\brectora?\b")
    df["covid"] = df["texto_min"].str.contains(r"covid|coronavirus|pandemia|sars-cov")
    df["bandera"] = df["inclusivo"].astype(int) + 2 * df["rector"].astype(int) + 4 * df["covid"].astype(int)
    df["mes"] = df["fecha"].dt.to_period("M")
    meses = pd.period_range(df["mes"].min(), df["mes"].max(), freq="M")
    por_mes = df.groupby("mes")["bandera"].agg(lambda s: "".join(map(str, s)))
    notas = [por_mes.get(m, "") for m in meses]

    # Vocabulario distintivo
    vocabulario = {}
    for r in orden:
        top = log_odds(df.loc[df["rectorado"] == r, "tokens"], df.loc[df["rectorado"] != r, "tokens"])[:14]
        vocabulario[r] = [{"t": w.replace("_", " "), "z": round(float(z), 1)} for w, z in top]

    # Temas
    X = TfidfVectorizer(analyzer=lambda d: d, min_df=20, max_df=0.4, sublinear_tf=True).fit_transform(df["tokens"])
    W = NMF(n_components=20, init="nndsvda", random_state=0, max_iter=1000).fit_transform(X)
    W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    prop = pd.DataFrame(W, columns=ETIQUETAS)
    relativo = (prop.groupby(df["rectorado"].values).mean() / prop.mean()).loc[orden]
    # Los ocho temas que más cambian entre el primer rectorado y el actual. Se excluye el de Navojoa,
    # que solo refleja el cambio de nombre del campus (URS → campus Navojoa).
    cambio = (relativo.iloc[-1] - relativo.iloc[0]).abs().drop("Campus Navojoa (sur)").sort_values(ascending=False)
    elegidos = sorted(cambio.index[:8], key=lambda t: -(relativo.iloc[-1][t] - relativo.iloc[0][t]))
    temas = [{"tema": t, "valores": [round(float(relativo.loc[r, t]), 2) for r in orden]} for t in elegidos]

    # Verbos para citar
    tipo = {v: t for t, vs in VERBOS.items() for v in vs}
    citas = df[["fecha"]].assign(verbo=df["texto_min"].str.findall(r"\b(" + "|".join(tipo) + r")\b")).explode("verbo").dropna()
    anio = citas["fecha"].dt.year
    verbos = {"anios": sorted(anio.unique().tolist()),
              "dijo": (citas["verbo"] == "dijo").groupby(anio).mean().mul(100).round(1).tolist(),
              "destaco": (citas["verbo"] == "destacó").groupby(anio).mean().mul(100).round(1).tolist(),
              "total": int(len(citas))}

    # Protagonismo
    prot = df.groupby("rectorado").agg(cuerpo=("rector", "mean"),
                                       titulo=("titulo", lambda s: s.str.lower().str.contains(r"\brectora?\b").mean()))
    prot = (prot.loc[orden] * 100).round(1)

    # Lenguaje inclusivo por trimestre y por firma
    tri = df.groupby(df["fecha"].dt.to_period("Q"))["inclusivo"].mean().mul(100)
    tri_ts = tri.copy()
    tri_ts.index = tri_ts.index.to_timestamp()
    cortes_f = pd.to_datetime(["2013-01-01", "2021-06-01", "2025-06-01", "2026-01-01", "2027-01-01"])
    periodos = ["2013–2021", "2021–2025", "2.º sem. 2025", "2026"]
    df["periodo"] = pd.cut(df["fecha"], cortes_f, labels=periodos, right=False)
    g = df.dropna(subset=["firma"]).groupby(["firma", "periodo"], observed=True)["inclusivo"].agg(["mean", "size"])
    tabla, n = g["mean"].unstack().mul(100), g["size"].unstack()
    tabla = tabla.loc[(n[["2021–2025", "2026"]] >= 15).all(axis=1)].sort_values("2026", ascending=False)
    firmas = [{"firma": f, "valores": [None if pd.isna(v) else round(float(v)) for v in fila]} for f, fila in tabla.iterrows()]

    vocab_z, inc_z = calcular_cortes(df, tri_ts)

    anual = df.groupby(df["fecha"].dt.year)["inclusivo"].mean().mul(100)
    desdob = df.groupby(df["fecha"].dt.year)["texto_min"].apply(lambda s: s.str.contains(DESDOBLAMIENTO).mean() * 100)
    anuncio = df.loc[df["texto_min"].str.contains(
        r"lenguaje (?:incluyente|inclusivo|no sexista)|guía de (?:comunicación|lenguaje)|manual de (?:estilo|lenguaje)")]

    ejemplos = []
    for e in EJEMPLOS:
        fila = pd.read_parquet(RAIZ / "data" / "processed" / "noticias_base.parquet", columns=["id", "fecha", "titulo", "url", "texto_limpio"],
                               filters=[("id", "==", e["id"])]).iloc[0]
        assert e["frase"] in fila["texto_limpio"], e["id"]
        ejemplos.append({**e, "fecha": f"{fila['fecha']:%Y-%m-%d}", "titulo": fila["titulo"], "url": fila["url"]})

    datos = {
        "inicioMes": str(meses[0]),
        "notas": notas,
        "rectorados": [{"nombre": r, "inicio": f"{transiciones[r]:%Y-%m-%d}", "notas": int((df["rectorado"] == r).sum())} for r in orden],
        "total": int(len(df)),
        "vocabulario": vocabulario,
        "temas": temas,
        "verbos": verbos,
        "protagonismo": {"cuerpo": prot["cuerpo"].tolist(), "titulo": prot["titulo"].tolist()},
        "inclusivoTrimestral": [{"q": str(q), "v": round(float(v), 1)} for q, v in tri.items()],
        "inclusivoAnual": {int(k): round(float(v), 1) for k, v in anual.items()},
        "desdoblamientoAnual": {int(k): round(float(v), 1) for k, v in desdob.items()},
        "firmas": {"periodos": periodos, "lineas": firmas},
        "cortes": {"vocab": [{"t": f"{t:%Y-%m}", "z": round(float(z), 2)} for t, z in vocab_z.items()],
                   "inclusivo": [{"t": f"{t:%Y-%m}", "z": round(float(z), 2)} for t, z in inc_z.items()]},
        "anuncios": {int(k): int(v) for k, v in anuncio.groupby(anuncio["fecha"].dt.year).size().items()},
        "ejemplos": ejemplos,
    }
    WEB.mkdir(exist_ok=True)
    texto = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    (WEB / "datos.json").write_text(texto, encoding="utf-8")
    print(f"web/datos.json: {len(texto) / 1024:.0f} KB")
    armar_pagina()


def armar_pagina():
    """Inserta web/datos.json en web/plantilla.html y escribe web/index.html, un documento HTML completo."""
    texto = (WEB / "datos.json").read_text(encoding="utf-8")
    html = (WEB / "plantilla.html").read_text(encoding="utf-8")
    assert html.count(MARCA) == 1, "La plantilla debe tener exactamente una marca de datos"
    html = html.replace(MARCA, texto.replace("</", "<\\/"))
    cabeza, cuerpo = html.split("<main", 1)
    pagina = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              f'{cabeza.strip()}\n</head>\n<body>\n<main{cuerpo.strip()}\n</body>\n</html>\n')
    (WEB / "index.html").write_text(pagina, encoding="utf-8")
    print("web/index.html escrito")


if __name__ == "__main__":
    import sys
    armar_pagina() if "--solo-armar" in sys.argv else main()
