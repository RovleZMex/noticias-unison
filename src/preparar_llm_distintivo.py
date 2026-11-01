"""Plan B: datos de entrenamiento solo con las notas que tienen marca de época.

    python src/preparar_llm_distintivo.py [--umbral 0.6]

Con todas las notas (src/preparar_llm.py) el modelo aprende sobre todo lo que comparten las épocas y el año
del prompt apenas cambia lo que escribe. Aquí:
  1. Un clasificador de época (TF-IDF + regresión logística) da, con validación cruzada, la probabilidad de
     que cada nota de entrenamiento sea de su época real. Los números se ocultan para que no gane por
     fechas escritas en el texto («2015», «ciclo 2019-2») sino por la forma de escribir.
  2. Se quedan las notas con probabilidad ≥ umbral, balanceadas entre épocas (las chicas se repiten).
  3. El prompt lleva la época, no el año. Con todas las notas la época quedaba diluida (las notas de 2025
     sin lenguaje inclusivo); con el filtro, cada época conserva solo las notas que sí la representan.

Valid y test son las mismas notas que en data/llm (misma partición por año), con el prompt de época y sin
filtrar: la evaluación se hace sobre notas reales cualquiera.
Salida: data/llm-distintivo/{train,valid,test}.jsonl, resumen.json y clasificador.joblib
(src/puntuar_generadas.py lo usa para medir si las notas generadas suenan a la época pedida)
"""

from __future__ import annotations

import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from preparar_llm import (ENTRADA, EPOCAS, MIN_PALABRAS, SEMILLA, TEST, VALID, RAIZ, anonimizar, es_nombre,
                          palabras_de_nombres, recortar)

SALIDA = RAIZ / "data" / "llm-distintivo"
ETIQUETAS = [e for e, _ in EPOCAS]
SISTEMA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
           "Escribe el cuerpo de una nota institucional con el estilo de la época {epoca}.")


def sin_numeros(texto: str) -> str:
    return re.sub(r"\d+", "0", texto.lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--umbral", type=float, default=0.6)
    p.add_argument("--objetivo", type=int, default=2500, help="Notas por época tras recortar o repetir")
    p.add_argument("--max-copias", type=int, default=3, help="Máximo de veces que se repite una nota de una época chica")
    a = p.parse_args()

    df = pd.read_parquet(ENTRADA, columns=["id", "fecha", "titulo", "cuerpo", "ent_texto", "ent_tipo", "es_plantilla", "n_palabras"])
    df = df.loc[~df["es_plantilla"] & (df["n_palabras"] >= MIN_PALABRAS)].reset_index(drop=True)
    inicios = pd.to_datetime([f for _, f in EPOCAS] + ["2100-01-01"])
    df["epoca"] = pd.cut(df["fecha"], inicios, labels=ETIQUETAS, right=False).astype(str)
    df["anio"] = df["fecha"].dt.year

    nombres = [[e for e, t in zip(es, ts) if t == "PER" and es_nombre(e)] for es, ts in zip(df["ent_texto"], df["ent_tipo"])]
    minusculas = {w for c in df["cuerpo"] for w in re.findall(r"\b[a-záéíóúñü]+\b", c)}
    palabras_nombre = palabras_de_nombres(nombres, minusculas)
    df["titulo_a"] = [anonimizar(t, n, palabras_nombre) for t, n in zip(df["titulo"], nombres)]
    df["cuerpo_a"] = [recortar(anonimizar(c, n, palabras_nombre)) for c, n in zip(df["cuerpo"], nombres)]

    # Misma partición que src/preparar_llm.py: por año, con la misma semilla
    df["parte"] = "train"
    for anio in sorted(df["anio"].unique()):
        g = df.loc[df["anio"] == anio].sample(frac=1, random_state=SEMILLA).index
        df.loc[g[:TEST], "parte"] = "test"
        df.loc[g[TEST:TEST + VALID], "parte"] = "valid"

    entreno = df.loc[df["parte"] == "train"].copy()
    vec = TfidfVectorizer(preprocessor=sin_numeros, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          sublinear_tf=True, max_features=200_000)
    X = vec.fit_transform(entreno["cuerpo_a"])
    y = entreno["epoca"].map(ETIQUETAS.index).to_numpy()
    clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced")
    prob = cross_val_predict(clf, X, y, cv=5, method="predict_proba", n_jobs=-1)
    entreno["p_epoca"] = prob[np.arange(len(y)), y]
    print(f"exactitud del clasificador (validación cruzada): {(prob.argmax(1) == y).mean():.3f}")

    # Qué palabras usa: las de mayor peso por época, para revisar que sea estilo y no tema
    clf.fit(X, y)
    SALIDA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizador": vec, "clasificador": clf, "epocas": ETIQUETAS}, SALIDA / "clasificador.joblib")
    vocab = np.array(vec.get_feature_names_out())
    pesos = {e: vocab[np.argsort(clf.coef_[i])[::-1][:25]].tolist() for i, e in enumerate(ETIQUETAS)}
    for e, w in pesos.items():
        print(f"{e}: {', '.join(w)}")

    elegidas = []
    for e in ETIQUETAS:
        g = entreno.loc[(entreno["epoca"] == e) & (entreno["p_epoca"] >= a.umbral)].sort_values("p_epoca", ascending=False)
        # Las épocas con pocas notas distintivas (2025–2026) se repiten para no quedar en minoría
        copias = min(a.max_copias, -(-a.objetivo // max(len(g), 1)))
        elegidas.append(pd.concat([g] * copias).head(a.objetivo))
    elegidas = pd.concat(elegidas)

    def fila(r, con_meta: bool) -> dict:
        m = [{"role": "system", "content": SISTEMA.format(epoca=r["epoca"])},
             {"role": "user", "content": r["titulo_a"]},
             {"role": "assistant", "content": r["cuerpo_a"]}]
        return {"id": int(r["id"]), "anio": int(r["anio"]), "epoca": r["epoca"], "messages": m} if con_meta else {"messages": m}

    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    for nombre, parte in [("train", elegidas), ("valid", df.loc[df["parte"] == "valid"]), ("test", df.loc[df["parte"] == "test"])]:
        filas = [fila(r, nombre == "test") for _, r in parte.iterrows()]
        with open(SALIDA / f"{nombre}.jsonl", "w", encoding="utf-8") as f:
            for i in rng.permutation(len(filas)):
                f.write(json.dumps(filas[i], ensure_ascii=False) + "\n")

    resumen = {
        "umbral": a.umbral,
        "exactitud_clasificador": round(float((prob.argmax(1) == y).mean()), 3),
        "entreno_original": entreno["epoca"].value_counts().reindex(ETIQUETAS).to_dict(),
        "entreno_filtrado": elegidas["epoca"].value_counts().reindex(ETIQUETAS).to_dict(),
        "entreno_filtrado_unicas": elegidas.drop_duplicates("id")["epoca"].value_counts().reindex(ETIQUETAS).to_dict(),
        "sobre_umbral": {f"{u:.1f}": entreno.loc[entreno["p_epoca"] >= u, "epoca"].value_counts().reindex(ETIQUETAS).fillna(0).astype(int).to_dict()
                         for u in (0.5, 0.6, 0.7, 0.8)},
        "palabras_por_epoca": pesos,
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumen.items() if k != "palabras_por_epoca"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
