"""Prepara los datos para el fine-tuning del modelo que escribe notas «al estilo de cada época».

    python src/preparar_llm.py

Cada ejemplo es una conversación en formato chat:
  system    -> el redactor del portal y el año de la nota (2013 … 2026)
  user      -> el titular
  assistant -> el cuerpo de la nota

Los nombres de personas se reemplazan por «[Nombre]» antes de entrenar, para que el modelo no aprenda
a poner declaraciones inventadas en boca de personas reales.

Salida: data/llm/{train,valid,test}.jsonl y data/llm/resumen.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "processed" / "noticias_base.parquet"
SALIDA = RAIZ / "data" / "llm"

# Mismas fechas de inicio que detecta el notebook 04 en el propio texto. El prompt lleva el año, no la época:
# con la época, la mitad de las notas de 2025–2026 (las de 2025) no tienen el lenguaje inclusivo de 2026
# y la etiqueta quedaba diluida. La época se guarda en test para la evaluación.
EPOCAS = [("2013–2017", "2013-01-01"), ("2017–2021", "2017-06-15"), ("2021–2025", "2021-06-23"), ("2025–2026", "2025-06-16")]
VALID, TEST = 20, 15      # notas reservadas por año; el resto (~23 mil) va a entrenamiento
MAX_PALABRAS = 380        # se recorta el cuerpo para que quepa en 1024 tokens
MIN_PALABRAS = 120
SEMILLA = 0
MARCA = "[Nombre]"

SISTEMA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
           "Escribe el cuerpo de una nota institucional con el estilo del año {anio}.")

# Palabras que aparecen en entidades PER de spaCy pero no son nombres de persona
NO_NOMBRE = {
    "unison", "universidad", "sonora", "rectoría", "rectoria", "sistemas", "vinculación", "difusión", "comunicación",
    "finanzas", "general", "académico", "académica", "clínico", "biólogo", "anuies", "culturest", "dictus",
    "odontología", "nogales", "hermosillo", "caborca", "navojoa", "campus", "departamento", "división", "facultad",
    "consejo", "colegio", "junta", "dirección", "secretaría", "programa", "centro", "instituto", "licenciatura",
    "maestría", "doctorado", "posgrado", "ciencias", "ingeniería", "derecho", "medicina", "enfermería", "física",
    "química", "biología", "historia", "letras", "artes", "música", "teatro", "danza", "radio", "televisión",
}
PALABRA_NOMBRE = re.compile(r"^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+$")
CONECTORES = {"de", "del", "la", "las", "los", "y"}


def es_nombre(ent: str) -> bool:
    """Una entidad PER es nombre de persona si tiene al menos dos palabras con mayúscula y ninguna institucional."""
    palabras = ent.split()
    propias = [p for p in palabras if p.lower() not in CONECTORES]
    if len(propias) < 2 or len(palabras) > 7:
        return False
    return all(PALABRA_NOMBRE.match(p) and p.lower() not in NO_NOMBRE for p in propias)


# Nombre y apellidos de los rectores que aparecen sueltos: se reemplazan en cualquier nota
APELLIDOS_RECTORES = {"Heriberto", "Grijalva", "Monteverde", "Velázquez", "Plancarte", "Camarena"}
# Secuencia de 2 a 6 palabras con mayúscula, con conectores en medio («Luz María de la Rosa»)
SECUENCIA = re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+(?:(?:de|del|la|las|los|y)\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,5}\b")


def palabras_de_nombres(todas: list[list[str]], corpus_min: set[str]) -> set[str]:
    """Palabras que forman parte de nombres detectados y que no se usan como palabra común en minúscula."""
    return {p for nombres in todas for n in nombres for p in n.split()
            if p.lower() not in CONECTORES and p.lower() not in NO_NOMBRE and p.lower() not in corpus_min}


def anonimizar(texto: str, nombres: list[str], palabras_nombre: set[str]) -> str:
    # 1. Nombres detectados en esta nota, del más largo al más corto
    for n in sorted(set(nombres), key=len, reverse=True):
        texto = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", MARCA, texto)
    # 2. Secuencias con mayúscula formadas por palabras de nombres vistas en el corpus («Alcántar Bojórquez»)
    def cambia(m):
        propias = [p for p in m.group(0).split() if p.lower() not in CONECTORES]
        return MARCA if sum(p in palabras_nombre for p in propias) >= 2 else m.group(0)
    texto = SECUENCIA.sub(cambia, texto)
    # 3. Apellidos sueltos: los de los nombres de esta nota y los de los rectores
    sueltos = {p for n in nombres for p in n.split()[-2:] if p.lower() not in CONECTORES and len(p) > 3} | APELLIDOS_RECTORES
    for p in sueltos:
        texto = re.sub(rf"(?<!\w){re.escape(p)}(?!\w)", MARCA, texto)
    # Une marcas repetidas («[Nombre] [Nombre]») que quedan cuando un nombre se partió
    return re.sub(rf"{re.escape(MARCA)}(?:\s+{re.escape(MARCA)})+", MARCA, texto)


def recortar(cuerpo: str) -> str:
    """Corta en el último párrafo completo antes de MAX_PALABRAS palabras."""
    salida, n = [], 0
    for parrafo in cuerpo.split("\n"):
        k = len(parrafo.split())
        if n + k > MAX_PALABRAS and salida:
            break
        salida.append(parrafo)
        n += k
    return "\n".join(salida)


def main():
    df = pd.read_parquet(ENTRADA, columns=["id", "fecha", "titulo", "cuerpo", "ent_texto", "ent_tipo", "es_plantilla", "n_palabras"])
    df = df.loc[~df["es_plantilla"] & (df["n_palabras"] >= MIN_PALABRAS)].reset_index(drop=True)
    inicios = pd.to_datetime([f for _, f in EPOCAS] + ["2100-01-01"])
    df["epoca"] = pd.cut(df["fecha"], inicios, labels=[e for e, _ in EPOCAS], right=False).astype(str)
    df["anio"] = df["fecha"].dt.year

    nombres = [[e for e, t in zip(es, ts) if t == "PER" and es_nombre(e)] for es, ts in zip(df["ent_texto"], df["ent_tipo"])]
    minusculas = {w for c in df["cuerpo"] for w in re.findall(r"\b[a-záéíóúñü]+\b", c)}
    palabras_nombre = palabras_de_nombres(nombres, minusculas)
    df["titulo_a"] = [anonimizar(t, n, palabras_nombre) for t, n in zip(df["titulo"], nombres)]
    df["cuerpo_a"] = [recortar(anonimizar(c, n, palabras_nombre)) for c, n in zip(df["cuerpo"], nombres)]

    rng = np.random.default_rng(SEMILLA)
    partes = {"train": [], "valid": [], "test": []}
    resumen = {}
    for anio in sorted(df["anio"].unique()):
        g = df.loc[df["anio"] == anio].sample(frac=1, random_state=SEMILLA)
        test, valid, train = g.iloc[:TEST], g.iloc[TEST:TEST + VALID], g.iloc[TEST + VALID:]
        resumen[int(anio)] = {"train": len(train), "valid": len(valid), "test": len(test)}
        for nombre, parte in [("train", train), ("valid", valid), ("test", test)]:
            for _, r in parte.iterrows():
                partes[nombre].append({
                    "id": int(r["id"]), "anio": int(anio), "epoca": r["epoca"],
                    "messages": [
                        {"role": "system", "content": SISTEMA.format(anio=anio)},
                        {"role": "user", "content": r["titulo_a"]},
                        {"role": "assistant", "content": r["cuerpo_a"]},
                    ],
                })

    SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre, filas in partes.items():
        orden = rng.permutation(len(filas))
        with open(SALIDA / f"{nombre}.jsonl", "w", encoding="utf-8") as f:
            for i in orden:
                # mlx_lm solo lee «messages»; id, año y época se guardan aparte para la evaluación
                fila = filas[i] if nombre == "test" else {"messages": filas[i]["messages"]}
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    con_marca = df["cuerpo_a"].str.contains(re.escape(MARCA)).mean()
    resumen["notas_con_nombres_reemplazados"] = round(float(con_marca), 3)
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
