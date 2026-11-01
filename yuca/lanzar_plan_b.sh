#!/bin/bash
# Sube el plan B a Yuca y deja la cadena de trabajos encadenada en Slurm. Se corre desde la Mac,
# con la conexión maestra abierta (`ssh yuca` en otra terminal):
#   bash yuca/lanzar_plan_b.sh
#
# 1. entrenar.slrm con los datos filtrados (src/preparar_llm_distintivo.py) → fusionado-distintivo
# 2. evaluar.slrm del plan B, en cuanto termine 1
# 3. evaluar.slrm del modelo actual (año en el prompt, γ 0/1/2/4), en cuanto termine su entrenamiento
set -euo pipefail
cd "$(dirname "$0")/.."
ssh -O check yuca

ssh -o BatchMode=yes yuca 'mkdir -p ~/unison-llm/datos-distintivo'
rsync -az data/llm-distintivo/train.jsonl data/llm-distintivo/valid.jsonl data/llm-distintivo/test.jsonl yuca:unison-llm/datos-distintivo/
rsync -az data/llm/titulares-eval.json yuca:unison-llm/datos/
rsync -az src/entrenar_yuca.py src/evaluar_yuca.py src/guia_epocas.py yuca/entrenar.slrm yuca/evaluar.slrm yuca:unison-llm/

ssh -o BatchMode=yes yuca 'bash -s' <<'EOF'
set -e
cd ~/unison-llm
# El entrenamiento del modelo actual (job unison-lora) puede seguir corriendo: la evaluación espera a que acabe
ACTUAL=$(squeue -h -u $USER -n unison-lora -o %i | head -1)
B=$(sbatch --parsable --job-name=unison-lora-b entrenar.slrm --datos "$HOME/unison-llm/datos-distintivo" --nombre distintivo)
sbatch --dependency=afterok:$B --job-name=eval-b evaluar.slrm fusionado-distintivo epoca eval-distintivo
if [ -n "$ACTUAL" ]; then
  sbatch --dependency=afterok:$ACTUAL --job-name=eval-anio evaluar.slrm fusionado-qwen2.5-0.5b anio eval-anio
else
  sbatch --job-name=eval-anio evaluar.slrm fusionado-qwen2.5-0.5b anio eval-anio
fi
squeue -u $USER
EOF
