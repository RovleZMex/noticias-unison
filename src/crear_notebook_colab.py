"""Escribe colab/entrenar_epocas.ipynb, el notebook de Colab para la segunda ronda de entrenamiento.

    python src/crear_notebook_colab.py

El notebook se genera desde aquí para mantenerlo en un solo archivo de texto revisable.
"""

from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

CELDAS = [
("md", """# Escucha cada época · segunda ronda de entrenamiento

Afina **Qwen2.5-Instruct** con LoRA (TRL + PEFT) sobre ~23 mil notas del portal de la Universidad de Sonora,
con el **año** de la nota en el prompt de sistema. Al final fusiona el adaptador y lo guarda en Google Drive;
la exportación a ONNX q4 para el navegador se hace después en la Mac (`src/cuantizar_onnx.py`).

**Antes de correr:**
1. *Entorno de ejecución → Cambiar tipo de entorno → GPU* (T4 en la versión gratis; L4 o A100 en Pro).
2. Sube a tu Drive la carpeta `unison-llm/datos/` con `train.jsonl`, `valid.jsonl` y `test.jsonl`
   (están en `data/llm/` del proyecto).
3. *Entorno de ejecución → Ejecutar todas*. Si la sesión se cae, vuelve a ejecutar todo: el entrenamiento
   continúa desde el último punto de control guardado en Drive."""),

("code", """!pip install -q -U "transformers>=5" trl peft accelerate datasets
# Colab trae torchao 0.10, que la versión nueva de PEFT rechaza al importarse; no se usa aquí
!pip uninstall -q -y torchao"""),

("code", """from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import torch

# ---- Configuración --------------------------------------------------------------
MODELO = "Qwen/Qwen2.5-0.5B-Instruct"   # o "Qwen/Qwen2.5-1.5B-Instruct" (≈1 GB en el navegador)
PASADAS = 2                             # pasadas completas sobre las ~23 mil notas
RANGO = 32
TASA = 1e-4
MAX_TOKENS = 1024
# ----------------------------------------------------------------------------------

BASE = Path("/content/drive/MyDrive/unison-llm")
DATOS = BASE / "datos"
ETIQUETA = MODELO.split("/")[-1].replace("-Instruct", "").lower()
SALIDA = BASE / f"lora-{ETIQUETA}"
FUSIONADO = BASE / f"fusionado-{ETIQUETA}"

gpu = torch.cuda.get_device_name(0)
# is_bf16_supported() da True en la T4 por emulación; bf16 real (y la multiplicación bf16→fp32 que usa TRL)
# requiere cómputo 8.0 o más (L4, A100). En la T4 se entrena en fp16.
COMPUTO = torch.cuda.get_device_capability(0)
BF16 = COMPUTO[0] >= 8
if not BF16:
    import trl.kernels.chunked_logprob as _clp
    _clp._MM_OUT_DTYPE = False   # suma los bloques de logits en fp32 sin el addmm que la T4 no soporta
grande = "1.5B" in MODELO
memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
LOTE = 8 if memoria > 30 else (2 if grande else 4)
ACUMULA = 16 // LOTE
print(f"{gpu} · cómputo {COMPUTO[0]}.{COMPUTO[1]} · {memoria:.0f} GB · bf16={BF16} · lote {LOTE}×{ACUMULA}")
assert (DATOS / "train.jsonl").exists(), f"Falta {DATOS}/train.jsonl en Drive\""""),

("md", """## Datos

Cada nota pasa de formato chat (`messages`) a *prompt + completion*: el prompt es el sistema con el año y el
titular; la completion es el cuerpo. TRL calcula la pérdida solo sobre la completion."""),

("code", """from datasets import load_dataset

def a_prompt_completion(fila):
    m = fila["messages"]
    return {"prompt": m[:2], "completion": m[2:]}

datos = load_dataset("json", data_files={p: str(DATOS / f"{p}.jsonl") for p in ["train", "valid"]})
entreno = datos["train"].map(a_prompt_completion, remove_columns=["messages"])
valida = datos["valid"].map(a_prompt_completion, remove_columns=["messages"])
# test lleva además id, año y época para la evaluación
prueba = load_dataset("json", data_files=str(DATOS / "test.jsonl"))["train"]
print(len(entreno), len(valida), len(prueba))
print(entreno[0]["prompt"][0]["content"])"""),

("md", """## Entrenamiento"""),

("code", """from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

tokenizador = AutoTokenizer.from_pretrained(MODELO)
modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16 if BF16 else torch.float16,
                                              attn_implementation="sdpa")

lora = LoraConfig(r=RANGO, lora_alpha=2 * RANGO, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

config = SFTConfig(
    output_dir=str(SALIDA),
    num_train_epochs=PASADAS,
    per_device_train_batch_size=LOTE,
    per_device_eval_batch_size=LOTE,
    gradient_accumulation_steps=ACUMULA,
    learning_rate=TASA,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,                  # fracción de los pasos totales
    max_length=MAX_TOKENS,
    completion_only_loss=True,
    gradient_checkpointing=True,
    bf16=BF16,
    fp16=not BF16,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=2,
    report_to="none",
    seed=0,
)
entrenador = SFTTrainer(model=modelo, args=config, train_dataset=entreno, eval_dataset=valida,
                        processing_class=tokenizador, peft_config=lora)
entrenador.model.print_trainable_parameters()"""),

("code", """# Continúa desde el último punto de control si la sesión se cayó
hay_checkpoint = SALIDA.exists() and any(SALIDA.glob("checkpoint-*"))
entrenador.train(resume_from_checkpoint=hay_checkpoint or None)
entrenador.save_model(str(SALIDA / "final"))"""),

("code", """import pandas as pd
historial = pd.DataFrame(entrenador.state.log_history)
val = historial.dropna(subset=["eval_loss"])[["step", "eval_loss"]]
print(val.to_string(index=False))
historial.to_csv(SALIDA / "historial.csv", index=False)"""),

("md", """## Fusionar y guardar

El modelo fusionado (pesos completos en fp16) queda en Drive. Descárgalo a `modelos/fusionado` en la Mac para
exportarlo a ONNX."""),

("code", """modelo_fusionado = entrenador.model.merge_and_unload()
modelo_fusionado = modelo_fusionado.to(torch.float16)
modelo_fusionado.save_pretrained(str(FUSIONADO), safe_serialization=True)
# El tokenizador no cambia. Se guardan los archivos originales de Qwen: los que escribe transformers 5
# no los lee optimum en la Mac (transformers 4.53) al exportar a ONNX.
from huggingface_hub import snapshot_download
snapshot_download(MODELO, allow_patterns=["tokenizer*", "vocab.json", "merges.txt"], local_dir=str(FUSIONADO))
modelo_fusionado.config.use_cache = True
modelo_fusionado.eval()
print(sorted(p.name for p in FUSIONADO.iterdir()))"""),

("md", """## ¿El modelo usa el año?

**1. Pérdida con el año correcto y con años equivocados.** Para cada nota de prueba se calcula la pérdida del
cuerpo poniendo en el prompt cada uno de los cuatro años representativos (uno por rectorado). Si el modelo
aprendió el estilo, la diagonal debe ser la más baja de su fila y con margen claro."""),

("code", """import torch.nn.functional as F

SISTEMA = ("Eres redactor del portal de noticias de la Universidad de Sonora. "
           "Escribe el cuerpo de una nota institucional con el estilo del año {anio}.")
EPOCAS = {"2013–2017": 2015, "2017–2021": 2019, "2021–2025": 2023, "2025–2026": 2026}

@torch.no_grad()
def perdida(titular, cuerpo, anio):
    msgs = [{"role": "system", "content": SISTEMA.format(anio=anio)}, {"role": "user", "content": titular}]
    a_ids = lambda texto: tokenizador(texto, add_special_tokens=False)["input_ids"]
    prompt = a_ids(tokenizador.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    completo = a_ids(tokenizador.apply_chat_template(msgs + [{"role": "assistant", "content": cuerpo}], tokenize=False))
    ids = torch.tensor([completo], device=modelo_fusionado.device)
    logits = modelo_fusionado(ids).logits[0, :-1].float()
    ce = F.cross_entropy(logits, ids[0, 1:], reduction="none")
    return ce[len(prompt) - 1:].mean().item()

filas = []
for f in prueba:
    if f["epoca"] not in EPOCAS:
        continue
    titular, cuerpo = f["messages"][1]["content"], f["messages"][2]["content"]
    filas.append({"real": f["epoca"], **{e: perdida(titular, cuerpo, a) for e, a in EPOCAS.items()}})
tabla = pd.DataFrame(filas)
tabla["acierto"] = tabla[list(EPOCAS)].idxmin(axis=1) == tabla["real"]
resumen = tabla.groupby("real")[list(EPOCAS) + ["acierto"]].mean().reindex(list(EPOCAS))
print(resumen.round(3))"""),

("md", """**2. Rasgos de estilo en texto generado.** Los mismos titulares de prueba, escritos con cada uno de los cuatro
años. Se mide el porcentaje de notas con lenguaje inclusivo y la proporción de «destacó» frente a «dijo», y se
compara con las notas reales de esos años."""),

("code", """import re, json, random

DESDOBLAMIENTO = re.compile(r"\\b(?:las y los|los y las|niñas y niños|niños y niñas|todas y todos|todos y todas|"
    r"alumnas y alumnos|académicas y académicos|investigadoras e investigadores|maestras y maestros|"
    r"profesoras y profesores|egresadas y egresados)\\b")
NEUTRO = re.compile(r"\\b(?:estudiantado|alumnado|profesorado|personal académico|comunidad estudiantil|personal docente)\\b")

def rasgos(textos):
    if not textos:
        return {"inclusivo_%": None, "destacó/(dijo+destacó)_%": None}
    t = [x.lower() for x in textos]
    inclusivo = sum(bool(DESDOBLAMIENTO.search(x) or NEUTRO.search(x)) for x in t) / len(t)
    dijo = sum(len(re.findall(r"\\bdijo\\b", x)) for x in t)
    destaco = sum(len(re.findall(r"\\bdestacó\\b", x)) for x in t)
    return {"inclusivo_%": round(100 * inclusivo, 1), "destacó/(dijo+destacó)_%": round(100 * destaco / max(dijo + destaco, 1), 1)}

tokenizador.padding_side = "left"

@torch.no_grad()
def redactar(titulares, anio, max_tokens=320):
    prompts = [tokenizador.apply_chat_template(
        [{"role": "system", "content": SISTEMA.format(anio=anio)}, {"role": "user", "content": t}],
        add_generation_prompt=True, tokenize=False) for t in titulares]
    x = tokenizador(prompts, return_tensors="pt", padding=True).to(modelo_fusionado.device)
    y = modelo_fusionado.generate(**x, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.1, pad_token_id=tokenizador.eos_token_id)
    return tokenizador.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)

torch.manual_seed(0)
muestra = random.Random(0).sample(list(prueba), 48)
titulares = [f["messages"][1]["content"] for f in muestra]
generadas, comparacion = [], []
for epoca, anio in EPOCAS.items():
    textos = []
    for i in range(0, len(titulares), 16):
        textos += redactar(titulares[i:i + 16], anio)
    generadas += [{"anio": anio, "titular": t, "texto": x} for t, x in zip(titulares, textos)]
    reales = [f["messages"][2]["content"] for f in prueba if f["anio"] == anio]
    comparacion.append({"año": anio, **{f"gen {k}": v for k, v in rasgos(textos).items()},
                        **{f"real {k}": v for k, v in rasgos(reales).items()}})
print(pd.DataFrame(comparacion).to_string(index=False))
with open(BASE / f"generadas-{ETIQUETA}.jsonl", "w", encoding="utf-8") as f:
    for g in generadas:
        f.write(json.dumps(g, ensure_ascii=False) + "\\n")"""),

("md", """## Ejemplo de la página

La nota del streamer en las cuatro épocas. Se guarda en Drive como `ejemplo-<modelo>.json`; va en
`web/ejemplo.json` del proyecto."""),

("code", """def cerrar(texto):
    i = max(texto.rfind("."), texto.rfind("»"))
    return texto[: i + 1] if i > 40 else texto

TITULAR = "Streamer de la Unison gana premio nacional"
torch.manual_seed(1)
ejemplo = {"titular": TITULAR, "textos": {e: cerrar(redactar([TITULAR], a)[0].strip()) for e, a in EPOCAS.items()}}
for e, t in ejemplo["textos"].items():
    print(f"=== {e} ({EPOCAS[e]}) ===\\n{t}\\n")
(BASE / f"ejemplo-{ETIQUETA}.json").write_text(json.dumps(ejemplo, ensure_ascii=False), encoding="utf-8")"""),
]


def main():
    celdas = []
    for tipo, fuente in CELDAS:
        lineas = fuente.splitlines(keepends=True)
        celda = {"cell_type": "markdown" if tipo == "md" else "code", "metadata": {}, "source": lineas}
        if tipo == "code":
            celda |= {"execution_count": None, "outputs": []}
        celdas.append(celda)
    nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    salida = RAIZ / "colab" / "entrenar_epocas.ipynb"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(salida.relative_to(RAIZ))


if __name__ == "__main__":
    main()
