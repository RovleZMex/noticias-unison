# Actividad 1: análisis de noticias de la Universidad de Sonora

**La historia:** https://rovlezmex.github.io/noticias-unison/

El enunciado está en [docs/enunciado.md](docs/enunciado.md).

## Estructura

```
Actividad 1/
├── README.md
├── requirements.txt
├── docs/
│   └── enunciado.md            # enunciado de la actividad
├── src/
│   ├── scraper_unison.py       # pipeline de scraping
│   ├── preprocesamiento.py     # limpieza, spaCy y detección de frases
│   ├── graficas_presentacion.py  # gráficas y cifras del deck
│   ├── exportar_web.py         # datos y armado de la página web
│   ├── preparar_llm.py         # datos de entrenamiento del modelo generador
│   ├── preparar_llm_distintivo.py  # plan B: notas con marca clara de época (clasificador TF-IDF)
│   ├── entrenar_yuca.py        # entrenamiento LoRA en la supercomputadora Yuca
│   ├── evaluar_yuca.py         # evaluación del modelo en Yuca
│   ├── guia_epocas.py          # classifier-free guidance por época
│   ├── puntuar_generadas.py    # clasificador de época sobre las notas generadas
│   ├── crear_notebook_colab.py # escribe colab/entrenar_epocas.ipynb
│   ├── generar_llm.py          # genera notas con el modelo (MLX, Mac)
│   ├── evaluar_condicion.py    # ¿el modelo usa la época? (MLX, Mac)
│   └── cuantizar_onnx.py       # ONNX → 4 bits para el navegador
├── yuca/                       # trabajos de Slurm para Yuca (entrenar, evaluar, lanzar plan B)
├── colab/
│   └── entrenar_epocas.ipynb   # entrenamiento LoRA en Google Colab (TRL + PEFT)
├── web/
│   ├── plantilla.html          # diseño y texto de la página
│   ├── datos.json              # series que usa la página
│   └── index.html              # página final (se publica en GitHub Pages)
├── notebooks/
│   ├── 01_carga_datos.ipynb    # JSONL → DataFrame tipado → parquet
│   ├── 02_eda.ipynb            # análisis exploratorio
│   ├── 03_preprocesamiento.ipynb  # capa base de preprocesamiento
│   └── 04_historia_rectorados.ipynb  # historia: cada rector reescribe la universidad
├── data/
│   ├── raw/
│   │   ├── html/               # caché del HTML de cada nota (<id>.html)
│   │   ├── noticias.jsonl      # salida del scraper
│   │   └── noticias.csv        # misma salida en CSV
│   ├── interim/
│   │   └── spacy.parquet       # caché de lemas, POS y entidades (~15 min de cómputo)
│   ├── llm/                    # train/valid/test.jsonl del modelo generador
│   └── processed/
│       ├── noticias.parquet        # dataset tipado (01)
│       ├── noticias_base.parquet   # texto preprocesado (03)
│       ├── stopwords_dominio.txt   # stopwords propias del corpus (03)
│       └── frases.csv              # frases de dos palabras detectadas (03)
├── reports/
│   └── figures/                # gráficas: eda_*.png (EDA) y hist_*.png (historia)
├── modelos/                    # pesos del modelo (no van a git; se publican en Hugging Face)
└── logs/
    └── scraping.log
```

`data/raw`, `data/interim`, los parquet, los `.jsonl`, `modelos/` y `logs/` no se suben a git: se regeneran con los scripts.

Los comandos se corren desde la carpeta `Actividad 1/`.

```bash
pip install -r requirements.txt
```

## Fuente

Las notas del portal están en `https://www.unison.mx/nota/?idnoti=<ID>`. Los IDs son secuenciales: los más recientes rondan 39000 (octubre 2026) y las notas de 2013 empiezan cerca del 14150. Los IDs que no existen devuelven la página sin título. `robots.txt` solo bloquea `/wp-admin/`.

## 1. Scraping (`src/scraper_unison.py`)

1. **Descubrir**: lee la portada y toma el `idnoti` más alto.
2. **Descargar**: baja cada ID con 4 hilos y un mínimo de 0.5 s entre peticiones. El HTML se guarda en `data/raw/html/<id>.html`; los IDs vacíos se marcan para no volver a pedirlos. Si se corta (`Ctrl+C`), se vuelve a correr el mismo comando y continúa desde la caché.
3. **Parsear**: extrae título, fecha, autor, sumario (si lo hay), imagen y cuerpo.
4. **Guardar**: escribe `data/raw/noticias.jsonl` y `data/raw/noticias.csv`.

Noticias desde una fecha (recorre hacia atrás desde la más reciente):

```bash
python src/scraper_unison.py --desde 2013-01-01 > logs/scraping.log 2>&1
```

Rango de IDs:

```bash
python src/scraper_unison.py --id-inicio 30000 --id-fin 39008
```

Reprocesar la caché sin descargar (útil tras cambiar el parser):

```bash
python src/scraper_unison.py --solo-parsear --desde 2013-01-01
```

Velocidad aproximada: 2 notas/s. Todo el archivo desde 2013 son ~25,000 IDs (~3.5 h).

## 2. Carga a DataFrame (`notebooks/01_carga_datos.ipynb`)

Lee `data/raw/noticias.jsonl`, asigna tipos, agrega columnas derivadas, elimina las 26 notas publicadas dos veces y guarda `data/processed/noticias.parquet` (23,830 notas, ~52 MB, compresión zstd).

```python
import pandas as pd
df = pd.read_parquet("../data/processed/noticias.parquet")  # desde notebooks/
```

### Columnas del parquet

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | int32 | `idnoti` de la nota |
| `fecha` | datetime | Fecha de publicación |
| `anio`, `mes`, `trimestre`, `semestre` | int | Partes de la fecha |
| `dia_semana` | category | `lunes` … `domingo` (ordenada) |
| `titulo` | str | Titular |
| `sumario` | str | Línea de entrada que algunas notas ponen antes del autor |
| `cuerpo` | str | Texto de la nota, un párrafo por línea |
| `texto` | str | Título + sumario + cuerpo; entrada para NLP |
| `autor` | category | Reportero (notas desde ~2018) |
| `autor_iniciales` | category | Iniciales de la firma al final de la nota, p. ej. `JAR` (sobre todo antes de 2018) |
| `firma` | category | `autor`, o `autor_iniciales` si no hay autor |
| `tiene_firma` | bool | Si la nota tiene alguna firma |
| `n_palabras`, `n_parrafos`, `n_caracteres` | int | Tamaño del cuerpo |
| `es_corta` | bool | Menos de 20 palabras (19 notas) |
| `fecha_texto` | str | Fecha como aparece en el sitio |
| `url`, `imagen` | str | URL de la nota y de su foto principal |

Los nombres en `autor` no están normalizados: hay variantes como "Paula Trespalacios", "Por Paula Trespalacios Argain" o firmas compartidas ("Aleyda Gutiérrez/Elías Quijada").

## 3. Análisis exploratorio (`notebooks/02_eda.ipynb`)

Cada sección plantea una pregunta, la responde con tablas o gráficas y cierra con un recuadro de **Hallazgos**. Cubre:

1. Vistazo general del dataset
2. Calidad: cobertura temporal, títulos repetidos, atípicos de longitud, ruido (URLs, correos) y completitud de la firma
3. Tiempo: volumen mensual, estacionalidad (mapa año × mes y perfil normalizado), día de la semana
4. Longitud de las notas y su evolución
5. Autoría y concentración
6. Vocabulario: leyes de Zipf y Heaps, hapax, stopwords de dominio, bigramas y trigramas
7. Términos distintivos por año (TF-IDF)
8. Entidades: departamentos, divisiones vs. facultades, campus
9. Señales temáticas: COVID-19, virtualidad, IA, género, medio ambiente, etc.
10. Resumen con las decisiones para el preprocesamiento y las historias candidatas

Las figuras se guardan en `reports/figures/eda_*.png`. Tarda ~30 s en ejecutarse y requiere `nltk` con las stopwords descargadas:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## 4. Preprocesamiento base (`notebooks/03_preprocesamiento.ipynb`)

Aplica las decisiones del EDA que no dependen de la historia elegida:

1. Excluye las 19 notas sin texto y marca con `es_plantilla` los avisos recurrentes (Gaceta, sesiones de órganos colegiados).
2. Limpia URLs, correos, teléfonos y viñetas; normaliza Unicode, comillas y espacios (`texto_limpio`).
3. Lematiza, etiqueta categorías gramaticales y extrae entidades con spaCy (`es_core_news_md`).
4. Quita stopwords de NLTK, de spaCy y del dominio, y conserva solo sustantivos, nombres propios, adjetivos y verbos (`lemas`).
5. Une frases de dos palabras detectadas con NPMI, p. ej. `medio_ambiente` (`lemas_frases`).

Salida: `data/processed/noticias_base.parquet`, con las columnas de `noticias.parquet` más `texto_limpio`, `lemas`, `lemas_frases`, `n_lemas`, `ent_texto`, `ent_tipo` y `es_plantilla`.

spaCy necesita su modelo en español:

```bash
python -m spacy download es_core_news_md
```

La primera ejecución tarda ~15 min por spaCy; el resultado queda en `data/interim/spacy.parquet` y las siguientes tardan ~20 s. Para recalcularlo, pon `REPROCESAR_SPACY = True`. En la primera ejecución con `nbconvert`, el kernel se quedó colgado después de spaCy (los procesos hijos no liberaron el kernel). Si te pasa, interrumpe y vuelve a ejecutar: la caché ya quedó guardada.

Resultado: 23,811 notas (168 plantillas), 4.5 M lemas de contenido (41% de los tokens), vocabulario de 72,766 lemas y 2,630 frases. Pendientes conocidos: el NER confunde verbos al inicio de oración con entidades ("Añadió", "Indicó"), y ~0.7% de los tokens son plurales que spaCy no lematizó ("alumnos").

## 5. Historia: la Unison cambió su forma de escribir (`notebooks/04_historia_rectorados.ipynb`)

**Pregunta:** ¿cambia la voz oficial de la universidad cuando cambia quien la dirige? El corpus cubre cuatro rectorados (Grijalva, Velázquez, Plancarte y Camarena), cuyas fechas de inicio se detectan desde el propio texto.

| Capítulo | Técnica |
|---|---|
| Vocabulario de cada rectorado | Log-odds con prior de Dirichlet (Monroe et al., 2008), sin nombres de personas y con control por la pandemia |
| Temas | NMF con 20 temas sobre TF-IDF |
| Protagonismo del rector | Menciones en cuerpo y titular |
| Estilo | Lenguaje inclusivo (desdoblamiento, colectivos neutros) por rectorado y por firma; verbos para citar |
| Cortes | Divergencia de Jensen-Shannon con muestras igualadas y ruido de base; prueba de Mann-Whitney |

**Resultado:** el vocabulario y los temas cambian de forma gradual; las transiciones no son cortes de vocabulario (p = 0.48). El estilo sí cambia de golpe: el lenguaje inclusivo pasa del 5% al 37% de las notas, en todas las firmas a la vez y sin anuncio, y "dijo" cae del 18% al 6.5% de las citas. El notebook cierra con límites y un guion para la narrativa. Las figuras quedan en `reports/figures/hist_*.png`. Tarda ~40 s.

## 6. Presentación

La historia está contada en un deck de 17 diapositivas siguiendo el *Data Storytelling Arc* de Brent Dykes (contexto y gancho → puntos que van subiendo → momento «ajá» → qué sigue), con títulos-conclusión al estilo de Cole Nussbaumer Knaflic. Las gráficas del deck se generan con:

```bash
python src/graficas_presentacion.py
```

El script rehace los cálculos del notebook 04 con un estilo para diapositivas, guarda `reports/figures/slides_*.png` e imprime las cifras que citan las diapositivas.

## 7. Página web (scrollytelling)

`web/index.html` cuenta la historia como una página que avanza con el scroll: 23,643 puntos (uno por nota) que se reorganizan por rectorado, pandemia, menciones del rector y lenguaje inclusivo; vocabulario y temas por época; el paso de «dijo» a «destacó»; el giro de los cortes y las firmas.

```bash
python src/exportar_web.py              # recalcula las series y arma la página
python src/exportar_web.py --solo-armar # solo vuelve a armar web/index.html desde la plantilla
```

El script guarda las series en `web/datos.json` y las inserta en `web/plantilla.html`. Para cambiar el diseño o el texto se edita la plantilla y se vuelve a armar. La página usa D3 7.9 (cdnjs).

**Publicación.** `.github/workflows/pages.yml` publica la carpeta `web/` en GitHub Pages en cada push a `main` que la modifique. En el repositorio, *Settings → Pages → Source* debe estar en *GitHub Actions*.

## 8. Modelo generador (experimento, ya no está en la página)

La idea era que el lector escribiera un titular y un modelo pequeño redactara la nota con el estilo de cada rectorado, en el navegador (sección «Escucha cada época»). La sección se quitó de la página: aun con el plan B (notas filtradas por un clasificador de época y classifier-free guidance, 58% de notas reconocidas como su época contra 39% del primer modelo), las cuatro versiones de una misma nota se parecían demasiado a simple vista, y la generación en el navegador era lenta. El modelo sigue publicado en Hugging Face (`RovleZ/unison-epocas`) y las evaluaciones están en `data/llm/eval/` y `data/llm-distintivo/eval/`. Lo que sigue documenta cómo se hizo.

1. **Datos** (`src/preparar_llm.py`): ~23 mil notas (sin plantillas, de 120 palabras o más, recortadas a 380) en formato chat: el sistema indica el año de la nota, el usuario da el titular y el asistente responde con el cuerpo. Se reservan 20 notas por año para validación y 15 para prueba. Los nombres de personas se reemplazan por «[Nombre]» (entidades PER de spaCy, secuencias de palabras de nombres y apellidos de los rectores) para que el modelo no ponga declaraciones inventadas en boca de personas reales.
2. **Entrenamiento** (`colab/entrenar_epocas.ipynb`, en Google Colab con GPU): LoRA de rango 32 sobre Qwen2.5-0.5B-Instruct con TRL, 2 pasadas, pérdida solo sobre el cuerpo. El notebook continúa desde el último punto de control si la sesión se cae, fusiona el adaptador, mide si el modelo usa el año (pérdida con el año correcto contra los demás; lenguaje inclusivo y «destacó» en texto generado) y guarda en Drive el modelo fusionado y el ejemplo de la página. El notebook se genera con `python src/crear_notebook_colab.py`.
3. **Exportación** (Mac, entorno `.venv-onnx` con transformers 4.53, optimum 1.27 y torch 2.7):

```bash
.venv-onnx/bin/optimum-cli export onnx --model modelos/fusionado --task text-generation-with-past modelos/onnx
.venv-onnx/bin/python src/cuantizar_onnx.py modelos/onnx modelos/web
```

`cuantizar_onnx.py` cuantiza a 4 bits (bloques de 32) las multiplicaciones de matrices y la tabla de embeddings, restaura el opset original (el cuantizador lo sube a 21 y onnxruntime-web rechaza el modelo) e incluye la plantilla de chat en `tokenizer_config.json`, que es donde la busca transformers.js. Resultado: ~395 MB.

4. **Navegador**: la página carga `modelos/web` desde Hugging Face Hub con transformers.js (WebGPU si está disponible, si no WebAssembly) y pide cada época con un año representativo: 2015, 2019, 2023 y 2026.

Una primera ronda (MLX en la Mac, 8 mil notas, prompt con el rango de la época) aprendió el género pero casi no distinguía las épocas: la pérdida con la época correcta era solo 0.003–0.009 menor que con una equivocada, y las notas generadas para 2025–2026 no tenían lenguaje inclusivo. Por eso la segunda ronda usa el año (2026 queda separado de 2025, cuando el lenguaje inclusivo todavía no aparecía) y todas las notas.
