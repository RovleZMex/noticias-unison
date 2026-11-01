"""Pipeline de scraping del portal de noticias de la Universidad de Sonora.

Las notas viven en https://www.unison.mx/nota/?idnoti=<ID>, con IDs secuenciales.
El pipeline tiene cuatro etapas:

1. descubrir: obtiene el ID más reciente desde la portada.
2. descargar: baja el HTML de cada ID y lo guarda en data/raw/html/ (caché, permite reanudar).
3. parsear: extrae título, fecha, autor, imagen y cuerpo de cada HTML.
4. guardar: escribe data/raw/noticias.jsonl y data/raw/noticias.csv.

Ejemplos:
    python src/scraper_unison.py --desde 2023-01-01
    python src/scraper_unison.py --id-inicio 38500 --id-fin 39008
    python src/scraper_unison.py --solo-parsear --desde 2013-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.unison.mx"
NOTA_URL = BASE_URL + "/nota/?idnoti={id}"
USER_AGENT = "Mozilla/5.0 (proyecto academico PLN, Maestria; contacto: gsrrobles@gmail.com)"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR = DATA_DIR / "html"
VACIO = "__VACIO__"  # marcador en caché para IDs que no existen

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
INICIALES_RE = re.compile(r"\s*\(([A-ZÁÉÍÓÚÑ]{1,5}(?:/[A-ZÁÉÍÓÚÑ]{1,5})*)\)\s*$")
FECHA_RE = re.compile(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+(?:de|del)\s+(\d{4})", re.I)

log = logging.getLogger("scraper_unison")


@dataclass
class Noticia:
    id: int
    url: str
    titulo: str
    fecha: str | None  # ISO YYYY-MM-DD
    fecha_texto: str
    autor: str | None
    autor_iniciales: str | None
    sumario: str | None
    imagen: str | None
    cuerpo: str
    n_palabras: int


# --- Etapa 1: descubrir -------------------------------------------------------

def ultimo_id(session: requests.Session) -> int:
    html = session.get(BASE_URL + "/", timeout=30).text
    ids = [int(x) for x in re.findall(r"idnoti=(\d+)", html)]
    if not ids:
        raise RuntimeError("No se encontraron enlaces a notas en la portada")
    return max(ids)


# --- Etapa 2: descargar -------------------------------------------------------

class Descargador:
    """Descarga con caché en disco, reintentos y un límite global de peticiones por segundo."""

    def __init__(self, pausa: float = 0.5, reintentos: int = 3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.pausa = pausa
        self.reintentos = reintentos
        self._lock = threading.Lock()
        self._ultimo = 0.0
        self.detener = threading.Event()
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _esperar_turno(self) -> None:
        with self._lock:
            espera = self._ultimo + self.pausa - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            self._ultimo = time.monotonic()

    def html(self, id_: int) -> str | None:
        ruta = RAW_DIR / f"{id_}.html"
        if ruta.exists():
            texto = ruta.read_text(encoding="utf-8")
            return None if texto == VACIO else texto

        for intento in range(1, self.reintentos + 1):
            self._esperar_turno()
            if self.detener.is_set():
                return None
            try:
                r = self.session.get(NOTA_URL.format(id=id_), timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                log.warning("ID %s intento %s falló: %s", id_, intento, e)
                if self.detener.wait(2 ** intento):
                    return None
        else:
            return None  # no se guarda en caché: se reintentará en la próxima corrida

        r.encoding = "utf-8"
        texto = r.text
        if not _tiene_nota(texto):
            _escribir_atomico(ruta, VACIO)
            return None
        _escribir_atomico(ruta, texto)
        return texto


def _escribir_atomico(ruta: Path, texto: str) -> None:
    """Escribe a un temporal y lo renombra, para que un corte no deje archivos a medias."""
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(texto, encoding="utf-8")
    tmp.replace(ruta)


def _tiene_nota(html: str) -> bool:
    m = re.search(r'class="page-entry-title">\s*([^<]*)<', html)
    return bool(m and m.group(1).strip())


# --- Etapa 3: parsear ---------------------------------------------------------

def parsear_fecha(texto: str) -> str | None:
    m = FECHA_RE.search(texto)
    if not m:
        return None
    dia, mes, anio = m.groups()
    mes_num = MESES.get(mes.lower())
    if not mes_num:
        return None
    try:
        return date(int(anio), mes_num, int(dia)).isoformat()
    except ValueError:
        return None


def _parece_autor(linea: str) -> bool:
    """Un nombre propio corto, sin puntuación final ni dígitos."""
    palabras = linea.split()
    return (2 <= len(palabras) <= 6 and len(linea) <= 60
            and not linea.endswith((".", ":", "”", '"', ","))
            and not any(c.isdigit() for c in linea)
            and all(p[0].isupper() or p.lower() in {"de", "del", "la", "y"} for p in palabras))


def parsear(id_: int, html: str) -> Noticia | None:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1.page-entry-title")
    contenido = soup.select_one("div.entry-content")
    if not h1 or not contenido:
        return None
    titulo = h1.get_text(" ", strip=True)
    if not any(c.isalpha() for c in titulo):
        return None  # marcadores vacíos del sitio (título ".")

    p_fecha = contenido.find("p")
    fecha_texto = p_fecha.get_text(strip=True) if p_fecha else ""

    cuerpo_div = contenido.find("div", attrs={"align": "justify"})
    if cuerpo_div is None:
        cuerpo_div = contenido
        for basura in cuerpo_div.select("#compartirredessociales"):
            basura.decompose()
        if p_fecha:
            p_fecha.decompose()

    img = cuerpo_div.find("img")
    imagen = img["src"] if img and img.get("src") else None

    # Los <br> separan párrafos; se convierten en saltos de línea antes de extraer el texto.
    for br in cuerpo_div.find_all("br"):
        br.replace_with("\n")
    lineas = [l.strip() for l in cuerpo_div.get_text().splitlines()]
    lineas = [l for l in lineas if l]

    # En las notas recientes el autor va en la primera línea, o en la segunda cuando
    # la nota abre con un sumario.
    autor = sumario = None
    if len(lineas) > 1 and _parece_autor(lineas[0]):
        autor = lineas.pop(0)
    elif len(lineas) > 2 and _parece_autor(lineas[1]):
        sumario = lineas.pop(0)
        autor = lineas.pop(0)

    # Las notas antiguas firman al final con iniciales entre paréntesis, p. ej. "(JAR)".
    autor_iniciales = None
    if lineas and (m := INICIALES_RE.search(lineas[-1])):
        autor_iniciales = m.group(1)
        lineas[-1] = lineas[-1][:m.start()].rstrip()
        if not lineas[-1]:
            lineas.pop()

    cuerpo = "\n".join(lineas)
    if not cuerpo:
        return None
    return Noticia(
        id=id_,
        url=NOTA_URL.format(id=id_),
        titulo=titulo,
        fecha=parsear_fecha(fecha_texto),
        fecha_texto=fecha_texto,
        autor=autor,
        autor_iniciales=autor_iniciales,
        sumario=sumario,
        imagen=imagen,
        cuerpo=cuerpo,
        n_palabras=len(cuerpo.split()),
    )


# --- Orquestación -------------------------------------------------------------

def descargar_rango(desc: Descargador, ids: list[int], hilos: int) -> dict[int, Noticia]:
    noticias: dict[int, Noticia] = {}
    ex = ThreadPoolExecutor(max_workers=hilos)
    try:
        futuros = {ex.submit(desc.html, i): i for i in ids}
        for f in tqdm(as_completed(futuros), total=len(futuros), desc="Descargando"):
            id_ = futuros[f]
            html = f.result()
            if html and (n := parsear(id_, html)):
                noticias[id_] = n
    except KeyboardInterrupt:
        # Sin esto, Ctrl+C espera a que terminen todas las descargas pendientes del lote.
        desc.detener.set()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return noticias


def descargar_desde_fecha(desc: Descargador, id_max: int, desde: str, hilos: int,
                          lote: int = 200) -> dict[int, Noticia]:
    """Recorre IDs hacia atrás en lotes hasta que un lote completo es anterior a `desde`.

    Los IDs son aproximadamente cronológicos, así que se descarga un lote extra
    por seguridad antes de detenerse.
    """
    noticias: dict[int, Noticia] = {}
    tope = id_max
    while tope > 0:
        ids = list(range(tope, max(tope - lote, 0), -1))
        nuevas = descargar_rango(desc, ids, hilos)
        noticias.update(nuevas)
        fechas = [n.fecha for n in nuevas.values() if n.fecha]
        log.info("IDs %s-%s: %s notas, fecha más reciente %s",
                 ids[-1], ids[0], len(nuevas), max(fechas, default="-"))
        if fechas and max(fechas) < desde:
            break
        tope -= lote
    return {i: n for i, n in noticias.items() if n.fecha is None or n.fecha >= desde}


def parsear_cache() -> dict[int, Noticia]:
    noticias = {}
    for ruta in tqdm(sorted(RAW_DIR.glob("*.html")), desc="Parseando caché"):
        html = ruta.read_text(encoding="utf-8")
        if html != VACIO and (n := parsear(int(ruta.stem), html)):
            noticias[n.id] = n
    return noticias


def guardar(noticias: dict[int, Noticia]) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filas = [asdict(n) for n in sorted(noticias.values(), key=lambda n: n.id)]
    with open(DATA_DIR / "noticias.jsonl", "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    df = pd.DataFrame(filas)
    df.to_csv(DATA_DIR / "noticias.csv", index=False, encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", help="Fecha mínima YYYY-MM-DD; recorre hacia atrás desde la nota más reciente")
    ap.add_argument("--id-inicio", type=int, help="ID inicial (inclusive)")
    ap.add_argument("--id-fin", type=int, help="ID final (inclusive); por defecto el más reciente")
    ap.add_argument("--solo-parsear", action="store_true", help="No descarga; reprocesa el HTML en caché")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--pausa", type=float, default=0.5, help="Segundos mínimos entre peticiones (global)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.solo_parsear:
        noticias = parsear_cache()
        if args.desde:
            noticias = {i: n for i, n in noticias.items() if n.fecha is None or n.fecha >= args.desde}
    else:
        desc = Descargador(pausa=args.pausa)
        id_fin = args.id_fin or ultimo_id(desc.session)
        log.info("ID más reciente: %s", id_fin)
        try:
            if args.desde:
                noticias = descargar_desde_fecha(desc, id_fin, args.desde, args.hilos)
            elif args.id_inicio:
                noticias = descargar_rango(desc, list(range(id_fin, args.id_inicio - 1, -1)), args.hilos)
            else:
                ap.error("Indica --desde, --id-inicio o --solo-parsear")
        except KeyboardInterrupt:
            log.info("Pausado. Lo descargado está en %s; vuelve a correr el mismo comando para continuar.", RAW_DIR)
            raise SystemExit(130)

    df = guardar(noticias)
    log.info("Guardadas %s noticias en %s", len(df), DATA_DIR)
    if len(df):
        log.info("Rango de fechas: %s a %s", df["fecha"].min(), df["fecha"].max())


if __name__ == "__main__":
    main()
