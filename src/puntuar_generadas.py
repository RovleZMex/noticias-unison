"""¿Las notas generadas suenan a la época que se pidió? Las clasifica con el clasificador del plan B.

    rsync -az yuca:unison-llm/eval-distintivo/ data/llm-distintivo/eval/
    rsync -az yuca:unison-llm/eval-anio/ data/llm/eval/
    python src/puntuar_generadas.py data/llm/eval data/llm-distintivo/eval

Para cada carpeta y cada γ: porcentaje de notas que el clasificador asigna a la época pedida (25 % = azar)
y la matriz época pedida × época que reconoce. El clasificador es el mismo que filtró los datos del plan B,
así que favorece a ese modelo; sirve para comparar γ y para ver si las cuatro columnas se separan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import __main__
import joblib
import pandas as pd

from preparar_llm_distintivo import sin_numeros

# El vectorizador se guardó corriendo preparar_llm_distintivo.py como script: su preprocesador quedó como __main__.sin_numeros
__main__.sin_numeros = sin_numeros

RAIZ = Path(__file__).resolve().parent.parent


def main():
    c = joblib.load(RAIZ / "data" / "llm-distintivo" / "clasificador.joblib")
    vec, clf, epocas = c["vectorizador"], c["clasificador"], c["epocas"]
    for carpeta in map(Path, sys.argv[1:]):
        for archivo in sorted(carpeta.glob("generadas-g*.jsonl")):
            notas = pd.DataFrame([json.loads(l) for l in open(archivo, encoding="utf-8")])
            notas["reconoce"] = [epocas[i] for i in clf.predict(vec.transform(notas["texto"]))]
            acierto = (notas["reconoce"] == notas["etiqueta"]).mean()
            print(f"\n{carpeta} · {archivo.stem}: {100 * acierto:.0f} % reconocidas como la época pedida")
            print(pd.crosstab(notas["etiqueta"], notas["reconoce"]).reindex(index=epocas, columns=epocas, fill_value=0).to_string())


if __name__ == "__main__":
    main()
