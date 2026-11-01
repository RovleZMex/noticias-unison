"""Evalúa un modelo fusionado en la GPU de Yuca: ¿lo que escribe cambia con la época del prompt?

    python evaluar_yuca.py --modelo fusionado-qwen2.5-0.5b --modo anio  --gammas 0 1 2 4
    python evaluar_yuca.py --modelo fusionado-distintivo   --modo epoca --gammas 0 1 2 4

1. Matriz de pérdida: cada nota de prueba con el prompt de las cuatro épocas (o sus años representativos).
   La diagonal debe ser la más baja de su fila; la distancia a las otras dice qué tanto pesa la etiqueta.
2. Generación: los mismos 48 titulares en las cuatro épocas, para cada γ (src/guia_epocas.py).
   La tabla de rasgos sale aquí; el clasificador de época se aplica después en la Mac (src/puntuar_generadas.py).
Salida en --salida: matriz.csv, generadas-g<γ>.jsonl, rasgos.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from guia_epocas import ANIOS, generar, prompt, rasgos

EPOCAS = ["2013–2017", "2017–2021", "2021–2025", "2025–2026"]
SIS_ANIO = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
            "Escribe el cuerpo de una nota institucional con el estilo del año {}.")
SIS_EPOCA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
             "Escribe el cuerpo de una nota institucional con el estilo de la época {}.")


@torch.no_grad()
def perdida(modelo, tok, sistema: str, titular: str, cuerpo: str) -> float:
    a = prompt(tok, sistema, titular)
    m = [{"role": "system", "content": sistema}, {"role": "user", "content": titular}, {"role": "assistant", "content": cuerpo}]
    b = tok(tok.apply_chat_template(m, tokenize=False), add_special_tokens=False)["input_ids"][:1024]
    ids = torch.tensor([b], device=modelo.device)
    etiquetas = ids.clone()
    etiquetas[:, :len(a)] = -100
    return modelo(ids, labels=etiquetas).loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", type=Path, required=True)
    p.add_argument("--modo", choices=["anio", "epoca"], required=True)
    p.add_argument("--prueba", type=Path, required=True, help="test.jsonl con id, epoca y messages")
    p.add_argument("--titulares", type=Path, required=True, help="JSON con la lista de titulares a generar")
    p.add_argument("--gammas", type=float, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--max-tokens", type=int, default=260)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)

    sistemas = ({e: SIS_ANIO.format(anio) for e, anio in zip(EPOCAS, ANIOS)} if a.modo == "anio"
                else {e: SIS_EPOCA.format(e) for e in EPOCAS})
    if not (a.modelo / "tokenizer_config.json").exists():
        # entrenar_yuca.py fusionaba sin tokenizador (snapshot_download sin internet no copiaba nada): sale de la caché
        import shutil
        from huggingface_hub import snapshot_download
        cache = Path(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct"))
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            shutil.copy(cache / f, a.modelo / f)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
    print(torch.cuda.get_device_name(0), a.modelo, a.modo, flush=True)

    # 1. Matriz de pérdida
    filas = []
    for l in open(a.prueba, encoding="utf-8"):
        f = json.loads(l)
        _, u, c = f["messages"]
        filas.append({"real": f["epoca"], **{e: perdida(modelo, tok, s, u["content"], c["content"]) for e, s in sistemas.items()}})
    m = pd.DataFrame(filas)
    m.to_csv(a.salida / "matriz.csv", index=False)
    resumen = m.groupby("real")[EPOCAS].mean().reindex(EPOCAS).round(3)
    resumen["aciertos"] = m.assign(ok=m[EPOCAS].idxmin(axis=1) == m["real"]).groupby("real")["ok"].agg(
        lambda s: f"{s.sum()}/{len(s)}").reindex(EPOCAS)
    print("\nreal / prompt\n" + resumen.to_string(), flush=True)

    # 2. Generación con cada γ
    titulares = json.loads(a.titulares.read_text(encoding="utf-8"))
    tabla = []
    for g in a.gammas:
        notas = generar(modelo, tok, titulares, g, a.max_tokens, sistemas=sistemas, semilla=0)
        with open(a.salida / f"generadas-g{g:g}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(n, ensure_ascii=False) + "\n" for n in notas)
        por = defaultdict(list)
        for n in notas:
            por[n["etiqueta"]].append(n["texto"])
        for e in EPOCAS:
            tabla.append({"gamma": g, "epoca": e, **rasgos(por[e])})
            print(f"γ={g:g} {e} {tabla[-1]}", flush=True)
    pd.DataFrame(tabla).to_csv(a.salida / "rasgos.csv", index=False)


if __name__ == "__main__":
    main()
