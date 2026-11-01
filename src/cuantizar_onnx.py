"""Cuantiza a 4 bits un modelo exportado a ONNX y lo deja con la estructura que espera transformers.js.

    .venv-onnx/bin/python src/cuantizar_onnx.py modelos/onnx modelos/web

Entrada: carpeta de `optimum-cli export onnx` (model.onnx + tokenizador y config).
Salida:  <destino>/config.json, tokenizer*.json, generation_config.json y <destino>/onnx/model_q4.onnx
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

ARCHIVOS = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"]


def main(origen: Path, destino: Path):
    (destino / "onnx").mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS:
        if (origen / nombre).exists():
            shutil.copy(origen / nombre, destino / nombre)
    # transformers.js lee la plantilla de chat desde tokenizer_config.json, no desde chat_template.jinja
    plantilla = destino / "chat_template.jinja"
    if plantilla.exists():
        cfg_path = destino / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["chat_template"] = plantilla.read_text(encoding="utf-8")
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    modelo = onnx.load(str(origen / "model.onnx"))
    opset_original = next(o.version for o in modelo.opset_import if o.domain in ("", "ai.onnx"))
    # MatMul y Gather (la tabla de embeddings) a 4 bits por bloques de 32, como los modelos q4 de onnx-community.
    # Dejar los embeddings en fp32 no acelera en WebGPU (6.3 contra 8.2 tokens/s) y sube el archivo a 855 MB.
    config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4,
                                          op_types_to_quantize=("MatMul", "Gather"), quant_axes=(("MatMul", 0), ("Gather", 1)))
    cuantizador = MatMulNBitsQuantizer(modelo, algo_config=config)
    cuantizador.process()
    # El cuantizador sube el opset estándar a 21 sin convertir los nodos existentes (ReduceMean queda inválido
    # para onnxruntime-web). Solo agrega operadores del dominio com.microsoft, así que se restaura el original.
    for o in cuantizador.model.model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = opset_original
    salida = destino / "onnx" / "model_q4.onnx"
    cuantizador.model.save_model_to_file(str(salida), use_external_data_format=False)
    tam = sum(f.stat().st_size for f in (destino / "onnx").iterdir())
    print(f"{salida}: {tam / 1e6:.0f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
