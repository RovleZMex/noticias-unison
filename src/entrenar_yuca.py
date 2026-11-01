"""Segunda ronda de entrenamiento en Yuca (ACARUS): el mismo SFT con LoRA del notebook de Colab, como script.

    sbatch yuca/entrenar.slrm                       # en Yuca, desde ~/unison-llm
    python src/entrenar_yuca.py --base ~/unison-llm --max-pasos 20   # prueba corta de velocidad

Misma configuración que colab/entrenar_epocas.ipynb (src/crear_notebook_colab.py): Qwen2.5-0.5B-Instruct,
LoRA r=32, 2 pasadas, lote efectivo 16, max_length 1024. En la MI210 entrena en bf16 nativo.
Deja en --base: lora-<nombre>/ (checkpoints, final, historial.csv) y fusionado-<nombre>/.
Plan B: --datos ~/unison-llm/datos-distintivo --nombre distintivo (src/preparar_llm_distintivo.py).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class Cronometro(TrainerCallback):
    """Imprime segundos por paso para comparar con Colab (T4: 4.39 s/paso con lote 4×4)."""

    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.paso0 = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kw):
        hechos = state.global_step - self.paso0
        if hechos and "loss" in (logs or {}):
            s = (time.time() - self.t0) / hechos
            print(f"paso {state.global_step}/{state.max_steps} · {s:.2f} s/paso · "
                  f"faltan {(state.max_steps - state.global_step) * s / 3600:.1f} h", flush=True)


def a_prompt_completion(fila):
    m = fila["messages"]
    return {"prompt": m[:2], "completion": m[2:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path.home() / "unison-llm")
    p.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--datos", type=Path, help="Carpeta con train/valid.jsonl (por omisión <base>/datos)")
    p.add_argument("--nombre", help="Nombre de las carpetas de salida (por omisión, el del modelo)")
    p.add_argument("--pasadas", type=float, default=2)
    p.add_argument("--rango", type=int, default=32)
    p.add_argument("--tasa", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--max-pasos", type=int, default=-1, help="Corta antes (prueba de velocidad)")
    a = p.parse_args()

    datos = a.datos or a.base / "datos"
    etiqueta = a.nombre or a.modelo.split("/")[-1].replace("-Instruct", "").lower()
    sufijo = "-prueba" if a.max_pasos > 0 else ""
    salida = a.base / f"lora-{etiqueta}{sufijo}"
    fusionado = a.base / f"fusionado-{etiqueta}"

    print(f"{torch.cuda.get_device_name(0)} · {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB · "
          f"lote {a.lote}×{16 // a.lote}", flush=True)

    d = load_dataset("json", data_files={s: str(datos / f"{s}.jsonl") for s in ["train", "valid"]})
    entreno = d["train"].map(a_prompt_completion, remove_columns=["messages"])
    valida = d["valid"].map(a_prompt_completion, remove_columns=["messages"])

    tokenizador = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=a.rango, lora_alpha=2 * a.rango, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(
        output_dir=str(salida),
        num_train_epochs=a.pasadas,
        max_steps=a.max_pasos,
        per_device_train_batch_size=a.lote,
        per_device_eval_batch_size=a.lote,
        gradient_accumulation_steps=16 // a.lote,
        learning_rate=a.tasa,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=a.max_tokens,
        completion_only_loss=True,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5 if a.max_pasos > 0 else 25,
        eval_strategy="no" if a.max_pasos > 0 else "steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
        seed=0,
    )
    entrenador = SFTTrainer(model=modelo, args=config, train_dataset=entreno, eval_dataset=valida,
                            processing_class=tokenizador, peft_config=lora, callbacks=[Cronometro()])
    entrenador.model.print_trainable_parameters()

    hay = salida.exists() and any(salida.glob("checkpoint-*"))
    entrenador.train(resume_from_checkpoint=hay or None)
    if a.max_pasos > 0:
        return

    entrenador.save_model(str(salida / "final"))
    pd.DataFrame(entrenador.state.log_history).to_csv(salida / "historial.csv", index=False)

    unido = entrenador.model.merge_and_unload().to(torch.float16)
    unido.save_pretrained(str(fusionado), safe_serialization=True)
    # Archivos originales del tokenizador de Qwen: optimum (transformers 4.53) en la Mac no lee los de transformers 5
    # (con HF_HUB_OFFLINE, snapshot_download con local_dir no copia nada: se copian desde la caché)
    import shutil
    from huggingface_hub import snapshot_download
    cache = Path(snapshot_download(a.modelo))
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        shutil.copy(cache / f, fusionado / f)
    print("Fusionado en", fusionado)


if __name__ == "__main__":
    main()
