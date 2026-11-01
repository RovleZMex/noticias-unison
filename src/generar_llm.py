"""Genera notas con el modelo afinado: los mismos titulares escritos con el estilo de cada época.

    .venv-llm/bin/python src/generar_llm.py                     # 40 titulares de prueba × 4 épocas
    .venv-llm/bin/python src/generar_llm.py --titular "Streamer de la Unison gana premio nacional"

Usa el adaptador LoRA de modelos/lora-epocas (o el modelo fusionado con --modelo).
Salida: data/llm/generadas.jsonl, una fila por titular y época.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "llm"
# El modelo se entrena con el año de la nota; cada época se pide con un año representativo
ANIOS = {"2013–2017": 2015, "2017–2021": 2019, "2021–2025": 2023, "2025–2026": 2026}
EPOCAS = list(ANIOS)
SISTEMA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
           "Escribe el cuerpo de una nota institucional con el estilo del año {anio}.")


def cerrar(texto: str) -> str:
    """Corta en la última oración completa: la generación se detiene por longitud, a veces a media frase."""
    i = max(texto.rfind("."), texto.rfind("»"))
    return texto[: i + 1] if i > 40 else texto


def redactar(modelo, tokenizador, titular: str, epoca: str, temperatura: float, max_tokens: int) -> str:
    mensajes = [{"role": "system", "content": SISTEMA.format(anio=ANIOS[epoca])}, {"role": "user", "content": titular}]
    prompt = tokenizador.apply_chat_template(mensajes, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=temperatura, top_p=0.9)
    return generate(modelo, tokenizador, prompt=prompt, max_tokens=max_tokens, sampler=sampler).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adaptador", default=str(RAIZ / "modelos" / "lora-epocas"))
    p.add_argument("--titular", help="Genera solo este titular en las cuatro épocas y lo imprime")
    p.add_argument("--json", help="Con --titular, guarda las cuatro notas en este archivo (ejemplo para la página)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--n", type=int, default=40, help="Titulares de prueba a usar")
    p.add_argument("--temperatura", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=450)
    a = p.parse_args()

    mx.random.seed(a.semilla)
    modelo, tokenizador = load(a.modelo, adapter_path=a.adaptador or None)

    if a.titular:
        textos = {}
        for epoca in EPOCAS:
            textos[epoca] = cerrar(redactar(modelo, tokenizador, a.titular, epoca, a.temperatura, a.max_tokens))
            print(f"\n=== {epoca} ===\n{textos[epoca]}")
        if a.json:
            Path(a.json).write_text(json.dumps({"titular": a.titular, "textos": textos}, ensure_ascii=False), encoding="utf-8")
            print(f"\nGuardado en {a.json}")
        return

    # Titulares reservados (nunca vistos en el entrenamiento), repartidos entre las cuatro épocas
    prueba = [json.loads(l) for l in open(DATOS / "test.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(prueba)
    por_epoca = {e: [f for f in prueba if f["epoca"] == e][: a.n // len(EPOCAS)] for e in EPOCAS}
    titulares = [f for e in EPOCAS for f in por_epoca[e]]

    salida = DATOS / "generadas.jsonl"
    with open(salida, "w", encoding="utf-8") as f:
        for i, fila in enumerate(titulares, 1):
            titular = fila["messages"][1]["content"]
            for epoca in EPOCAS:
                texto = redactar(modelo, tokenizador, titular, epoca, a.temperatura, a.max_tokens)
                f.write(json.dumps({"id": fila["id"], "epoca_original": fila["epoca"], "epoca": epoca,
                                    "titular": titular, "texto": texto}, ensure_ascii=False) + "\n")
            print(f"{i}/{len(titulares)} {titular[:70]}", flush=True)
    print(f"Guardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
