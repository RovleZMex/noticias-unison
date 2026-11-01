"""Amplifica el año del prompt al generar (classifier-free guidance / contrastive decoding).

    .venv-onnx/bin/python src/guia_epocas.py --gammas 0 1 2 3          # 48 titulares × 4 años × cada γ
    .venv-onnx/bin/python src/guia_epocas.py --titular "Streamer de la Unison gana premio nacional" --gammas 2

En cada token el modelo predice dos veces: con el año pedido y con un prompt sin año. Se usa
    log p(año) + γ · (log p(año) − log p(sin año))
así lo que distingue a ese año de lo neutro pesa más. γ = 0 es la generación normal.
Solo se permiten tokens con p(año) ≥ α · max p(año), para que γ alto no meta palabras imposibles.
Salida: data/llm/guia-g<γ>.jsonl y una tabla de rasgos por año y γ.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "llm"
ANIOS = [2015, 2019, 2023, 2026]
SISTEMA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
           "Escribe el cuerpo de una nota institucional con el estilo del año {anio}.")
NEUTRO = "Eres redactor del portal de noticias de la Universidad de Sonora. Escribe el cuerpo de una nota institucional."

DESDOBLAMIENTO = re.compile(r"\b(?:las y los|los y las|niñas y niños|niños y niñas|todas y todos|todos y todas|"
                            r"alumnas y alumnos|académicas y académicos|investigadoras e investigadores|"
                            r"maestras y maestros|profesoras y profesores|egresadas y egresados)\b")
INCLUSIVO = re.compile(r"\b(?:estudiantado|alumnado|profesorado|personal académico|comunidad estudiantil|personal docente)\b")


def prompt(tok, sistema: str, titular: str) -> list[int]:
    m = [{"role": "system", "content": sistema}, {"role": "user", "content": titular}]
    return tok(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generar(modelo, tok, titulares: list[str], gamma: float, max_tokens: int, alfa=0.1, temp=0.7, top_p=0.9,
            repeticion=1.1, semilla=0, sistemas: dict | None = None) -> list[dict]:
    """Genera cada titular con cada prompt de `sistemas` (etiqueta → prompt; por omisión los 4 años) a la vez;
    con γ > 0 cada fila lleva su gemela con el prompt neutro."""
    dev = modelo.device
    sistemas = sistemas or {a: SISTEMA.format(anio=a) for a in ANIOS}
    filas = [(t, e) for t in titulares for e in sistemas]
    ids = [prompt(tok, sistemas[e], t) for t, e in filas]
    guia = gamma > 0
    if guia:
        ids += [prompt(tok, NEUTRO, t) for t, _ in filas]
    n = len(filas)
    largo = max(map(len, ids))
    entrada = torch.tensor([[tok.pad_token_id] * (largo - len(x)) + x for x in ids], device=dev)
    mascara = torch.tensor([[0] * (largo - len(x)) + [1] * len(x) for x in ids], device=dev)
    fin = torch.tensor([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")], device=dev)
    gen = torch.Generator(device="cpu").manual_seed(semilla)

    salida = [[] for _ in range(n)]
    usados = torch.zeros(n, modelo.config.vocab_size, dtype=torch.bool, device=dev)
    vivo = torch.ones(n, dtype=torch.bool, device=dev)
    pasado, siguiente = None, entrada
    for _ in range(max_tokens):
        pos = (mascara.cumsum(-1) - 1).clamp(min=0)[:, -siguiente.shape[1]:]
        r = modelo(input_ids=siguiente, attention_mask=mascara, position_ids=pos, past_key_values=pasado, use_cache=True)
        pasado = r.past_key_values
        lp = torch.log_softmax(r.logits[:, -1].float(), -1)
        cond = lp[:n]
        if guia:
            puntaje = (cond + gamma * (cond - lp[n:])).nan_to_num(nan=-float("inf"), posinf=-float("inf"))
            puntaje[cond < cond.max(-1, keepdim=True).values + torch.log(torch.tensor(alfa))] = -float("inf")
        else:
            puntaje = cond.clone()
        puntaje = puntaje[:, :usados.shape[1]] - usados * torch.log(torch.tensor(repeticion))  # penaliza lo ya generado
        prob = torch.softmax(puntaje / temp, -1)
        orden, idx = prob.sort(-1, descending=True)
        corte = orden.cumsum(-1) - orden > top_p
        orden[corte] = 0
        elegido = idx.gather(-1, torch.multinomial((orden / orden.sum(-1, keepdim=True)).cpu(), 1, generator=gen).to(dev))
        elegido[~vivo] = tok.pad_token_id
        usados[torch.arange(n, device=dev), elegido[:, 0]] |= vivo
        for i in range(n):
            if vivo[i]:
                salida[i].append(elegido[i].item())
        vivo &= ~torch.isin(elegido[:, 0], fin)
        if not vivo.any():
            break
        siguiente = torch.cat([elegido, elegido]) if guia else elegido
        mascara = torch.cat([mascara, torch.ones(len(mascara), 1, dtype=mascara.dtype, device=dev)], -1)
    return [{"titular": t, "etiqueta": a, "gamma": gamma, "texto": tok.decode(s, skip_special_tokens=True).strip()}
            for (t, a), s in zip(filas, salida)]


def rasgos(textos: list[str]) -> dict:
    t = [x.lower() for x in textos]
    dijo = sum(len(re.findall(r"\bdijo\b", x)) for x in t)
    destaco = sum(len(re.findall(r"\bdestacó\b", x)) for x in t)
    palabras = [w for x in t for w in re.findall(r"\w+", x)]
    bigramas = list(zip(palabras, palabras[1:]))
    return {"destacó_%": round(100 * destaco / max(dijo + destaco, 1)),
            "destacó/nota": round(destaco / len(t), 2),
            "inclusivo_%": round(100 * sum(bool(DESDOBLAMIENTO.search(x) or INCLUSIVO.search(x)) for x in t) / len(t)),
            "bigramas_distintos_%": round(100 * len(set(bigramas)) / max(len(bigramas), 1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default=str(RAIZ / "modelos" / "fusionado"))
    p.add_argument("--gammas", type=float, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--titular")
    p.add_argument("--max-tokens", type=int, default=260)
    p.add_argument("--lote", type=int, default=8, help="Titulares por lote")
    a = p.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()

    if a.titular:
        for g in a.gammas:
            for f in generar(modelo, tok, [a.titular], g, a.max_tokens):
                print(f"\n=== γ={g} · {f['etiqueta']} ===\n{f['texto']}")
        return

    vistos = dict.fromkeys(json.loads(l)["titular"] for l in open(DATOS / "generadas-qwen2.5-0.5b.jsonl", encoding="utf-8"))
    titulares = list(vistos)
    for g in a.gammas:
        notas = []
        for i in range(0, len(titulares), a.lote):
            notas += generar(modelo, tok, titulares[i:i + a.lote], g, a.max_tokens, semilla=i)
            print(f"γ={g} {min(i + a.lote, len(titulares))}/{len(titulares)}", flush=True)
        with open(DATOS / f"guia-g{g:g}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(n, ensure_ascii=False) + "\n" for n in notas)
        por_anio = defaultdict(list)
        for n in notas:
            por_anio[n["etiqueta"]].append(n["texto"])
        for anio in ANIOS:
            print(f"γ={g:g} {anio} {rasgos(por_anio[anio])}", flush=True)


if __name__ == "__main__":
    main()
