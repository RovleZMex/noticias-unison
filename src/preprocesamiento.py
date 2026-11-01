"""Funciones de la capa base de preprocesamiento.

Viven en un módulo (y no en el notebook) porque spaCy con varios procesos necesita
poder importarlas desde los procesos hijos.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import pandas as pd

# --- Limpieza de texto --------------------------------------------------------

URL = re.compile(r"https?://\S+|www\.\S+")
CORREO = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TELEFONO = re.compile(r"(?:\(?\+?52\)?[\s-]?)?\b\d{3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")
VINETAS = re.compile(r"^[\s•●▪◦·\-–—*]+", flags=re.M)
ESPACIOS = re.compile(r"[ \t  ​]+")
SALTOS = re.compile(r"\n{2,}")

COMILLAS = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "´": "'"})


def limpiar_texto(texto: str) -> str:
    """Quita URLs, correos, teléfonos y viñetas, y normaliza Unicode, comillas y espacios.

    Conserva mayúsculas y puntuación: el NER y los modelos de lenguaje las aprovechan.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = URL.sub(" ", texto)
    texto = CORREO.sub(" ", texto)
    texto = TELEFONO.sub(" ", texto)
    texto = texto.translate(COMILLAS)
    texto = VINETAS.sub("", texto)
    texto = ESPACIOS.sub(" ", texto)
    texto = "\n".join(linea.strip() for linea in texto.splitlines())
    return SALTOS.sub("\n", texto).strip()


# --- Procesamiento lingüístico con spaCy ---------------------------------------

def procesar_spacy(textos: list[str], ids: list[int], modelo: str = "es_core_news_md",
                   n_process: int = 4, batch_size: int = 25) -> pd.DataFrame:
    """Lematiza y etiqueta cada texto, y extrae sus entidades nombradas.

    Devuelve una fila por nota con listas paralelas `lemas`/`pos` (solo tokens
    alfabéticos, lema en minúsculas) y `ent_texto`/`ent_tipo`.
    """
    import spacy
    from tqdm.auto import tqdm

    nlp = spacy.load(modelo, disable=["parser"])  # el árbol sintáctico no se usa y es lo más lento
    filas = []
    docs = nlp.pipe(textos, batch_size=batch_size, n_process=n_process)
    for id_, doc in tqdm(zip(ids, docs), total=len(textos), desc="spaCy"):
        alfa = [t for t in doc if t.is_alpha]
        filas.append({
            "id": id_,
            "lemas": [t.lemma_.lower() for t in alfa],
            "pos": [t.pos_ for t in alfa],
            "ent_texto": [e.text for e in doc.ents],
            "ent_tipo": [e.label_ for e in doc.ents],
        })
    return pd.DataFrame(filas)


# --- Detección de frases --------------------------------------------------------

def detectar_frases(documentos: pd.Series, min_conteo: int = 30, umbral_npmi: float = 0.5) -> pd.DataFrame:
    """Encuentra pares de palabras consecutivas que aparecen juntas mucho más de lo esperado.

    Usa NPMI (información mutua puntual normalizada), que va de -1 a 1: 1 significa
    que las dos palabras solo aparecen juntas.
    """
    unigramas, bigramas = Counter(), Counter()
    for doc in documentos:
        unigramas.update(doc)
        bigramas.update(zip(doc, doc[1:]))
    n = sum(unigramas.values())

    filas = []
    for (a, b), c in bigramas.items():
        if c < min_conteo:
            continue
        p_ab = c / n
        pmi = math.log(p_ab / ((unigramas[a] / n) * (unigramas[b] / n)))
        npmi = pmi / -math.log(p_ab)
        if npmi >= umbral_npmi:
            filas.append((a, b, c, npmi))
    return (pd.DataFrame(filas, columns=["palabra_1", "palabra_2", "conteo", "npmi"])
            .sort_values("conteo", ascending=False, ignore_index=True))


def unir_frases(doc: list[str], frases: set[tuple[str, str]]) -> list[str]:
    """Reemplaza cada par de `frases` por un solo token unido con "_", de izquierda a derecha."""
    salida, i = [], 0
    while i < len(doc):
        if i + 1 < len(doc) and (doc[i], doc[i + 1]) in frases:
            salida.append(f"{doc[i]}_{doc[i + 1]}")
            i += 2
        else:
            salida.append(doc[i])
            i += 1
    return salida
