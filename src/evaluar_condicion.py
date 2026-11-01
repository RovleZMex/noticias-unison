"""¿El modelo usa la época del prompt? Pérdida de cada nota de prueba con la época correcta y con las otras.

    .venv-llm/bin/python src/evaluar_condicion.py [--adaptador modelos/lora-epocas] [--n 20]

Si el modelo aprendió el estilo de cada periodo, la diagonal (época real = época del prompt) debe ser
la pérdida más baja de cada fila. Con el modelo base (--adaptador "") la tabla sirve de referencia.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from generar_llm import ANIOS, EPOCAS, SISTEMA

RAIZ = Path(__file__).resolve().parent.parent


def perdida(modelo, tokenizador, titular: str, cuerpo: str, epoca: str) -> float:
    """Entropía cruzada promedio sobre los tokens del cuerpo, dado el titular y la época."""
    mensajes = [{"role": "system", "content": SISTEMA.format(anio=ANIOS[epoca])}, {"role": "user", "content": titular}]
    prompt = tokenizador.apply_chat_template(mensajes, add_generation_prompt=True, tokenize=True)
    completo = tokenizador.apply_chat_template(mensajes + [{"role": "assistant", "content": cuerpo}], tokenize=True)
    ids = mx.array(completo)[None]
    logits = modelo(ids[:, :-1])
    ce = nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")[0]
    return ce[len(prompt) - 1:].mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adaptador", default=str(RAIZ / "modelos" / "lora-epocas"))
    p.add_argument("--n", type=int, default=20, help="Notas de prueba por época")
    a = p.parse_args()

    modelo, tokenizador = load(a.modelo, adapter_path=a.adaptador or None)
    filas = [json.loads(l) for l in open(RAIZ / "data" / "llm" / "test.jsonl", encoding="utf-8")]

    print(f"{'real / prompt':>14}" + "".join(f"{e:>12}" for e in EPOCAS) + "   aciertos")
    for real in EPOCAS:
        notas = [f for f in filas if f["epoca"] == real][: a.n]
        tabla = [[perdida(modelo, tokenizador, f["messages"][1]["content"], f["messages"][2]["content"], e)
                  for e in EPOCAS] for f in notas]
        medias = [sum(c) / len(c) for c in zip(*tabla)]
        aciertos = sum(min(range(4), key=fila.__getitem__) == EPOCAS.index(real) for fila in tabla)
        print(f"{real:>14}" + "".join(f"{m:12.3f}" for m in medias) + f"   {aciertos}/{len(notas)}")


if __name__ == "__main__":
    main()
