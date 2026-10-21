"""
Laboratorio 7 - Evaluacion de LLMs con las metricas de la rubrica.

Corre cada modelo sobre las ordenes de dataset_lab7.json (usando la transcripcion correcta,
es decir, mide solo el LLM) y calcula: F1 por categoria, estructura, robustez, JSON valido,
exactitud por orden y latencia. Necesita probar_llms.py en la misma carpeta.

Ejemplos:
    py -3.12 evaluar.py --ver-dataset                                  # revisar las anotaciones
    py -3.12 evaluar.py --modelos qwen2.5:1.5b --version v1            # linea base, particion de ajuste
    py -3.12 evaluar.py --modelos qwen2.5:1.5b --version v2 --postproceso
    py -3.12 evaluar.py --particion prueba --version v2 --postproceso  # medicion final (4 modelos)

IMPORTANTE: las metricas son MI interpretacion de la guia (el profesor no entrego su script).
La definicion exacta de cada metrica puede diferir de la oficial.
"""
import argparse
import sys
import json
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime

import ollama
import probar_llms as P

CATS = ("accion", "persona", "objeto", "adjetivo", "lugar")
_CLAVES = (("objetos", "objeto"), ("personas", "persona"), ("lugares", "lugar"))


# ----------------------------------------------------------------------------
# Normalizacion y extraccion de elementos
# ----------------------------------------------------------------------------
def norm(s):
    """minusculas, sin tildes, sin articulo inicial: 'La Señora Gómez' -> 'senora gomez'."""
    s = unicodedata.normalize("NFD", str(s).strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"^(el|la|los|las|un|una|unos|unas)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def a_dict(x):
    """Devuelve el dict de acciones, o None si la prediccion no es un JSON valido."""
    if isinstance(x, dict):
        return x
    try:
        d = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return None
    return d if P.es_json_valido(x) else None


def elementos(x):
    """-> (Counter por categoria, Counter de entidades con su estructura, idem con acciones)."""
    cat = {k: Counter() for k in CATS}
    est, est_full = Counter(), Counter()
    d = a_dict(x)
    if d is None:
        return cat, est, est_full
    for acc in d.get("acciones", []):
        a = norm(acc.get("accion", ""))
        cat["accion"][a] += 1
        est_full[(a, "accion", "", ())] += 1
        for clave, nombre_cat in _CLAVES:
            for e in acc.get(clave, []):
                n = norm(e.get("nombre", ""))
                if not n:
                    continue
                adjs = tuple(sorted(a2 for a2 in (norm(v) for v in e.get("adjetivos", [])) if a2))
                cat[nombre_cat][n] += 1
                for v in adjs:
                    cat["adjetivo"][v] += 1
                est[(a, nombre_cat, n, adjs)] += 1
                est_full[(a, nombre_cat, n, adjs)] += 1
    return cat, est, est_full


def _tp_fp_fn(g, p):
    return sum((g & p).values()), sum((p - g).values()), sum((g - p).values())


def f1(tp, fp, fn):
    if tp + fp + fn == 0:
        return 1.0
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def _acumular(pares):
    acc = {k: [0, 0, 0] for k in CATS}
    est = [0, 0, 0]
    for par in pares:
        gc, ge, _ = elementos(par["gold"])
        pc, pe, _ = elementos(par["pred"])
        for k in CATS:
            for i, v in enumerate(_tp_fp_fn(gc[k], pc[k])):
                acc[k][i] += v
        for i, v in enumerate(_tp_fp_fn(ge, pe)):
            est[i] += v
    return acc, est


def metricas(pares):
    """pares: lista de dicts {nivel, gold, pred}. Devuelve las metricas de la rubrica."""
    acc, est = _acumular(pares)
    f1_cat = {k: f1(*acc[k]) for k in CATS}
    dificiles = [p for p in pares if p["nivel"] in ("4", "5")]
    acc45, _ = _acumular(dificiles)
    robustez = sum(f1(*acc45[k]) for k in CATS) / len(CATS) if dificiles else None

    exactas, por_nivel = 0, {}
    for par in pares:
        _, _, ge = elementos(par["gold"])
        _, _, pe = elementos(par["pred"])
        ok = (ge == pe) and a_dict(par["pred"]) is not None
        exactas += ok
        n_ok, n_tot = por_nivel.get(par["nivel"], (0, 0))
        por_nivel[par["nivel"]] = (n_ok + ok, n_tot + 1)

    validos = sum(a_dict(p["pred"]) is not None for p in pares)
    n = len(pares)
    return {
        "n": n,
        "f1_categorias": {k: round(v, 3) for k, v in f1_cat.items()},
        "f1_promedio": round(sum(f1_cat.values()) / len(CATS), 3),
        "estructura_f1": round(f1(*est), 3),
        "robustez_f1": None if robustez is None else round(robustez, 3),
        "json_valido_pct": round(100 * validos / n, 1),
        "exactas_pct": round(100 * exactas / n, 1),
        "exactas_por_nivel": {k: f"{a}/{b}" for k, (a, b) in sorted(por_nivel.items())},
    }


def compacto(x):
    """Texto corto para comparar una anotacion o una prediccion a simple vista."""
    d = a_dict(x)
    if d is None:
        return "(JSON invalido)"
    if not d.get("acciones"):
        return "[] (no es una orden)"
    partes = []
    for acc in d["acciones"]:
        trozos = []
        for clave, tag in (("objetos", "obj"), ("personas", "per"), ("lugares", "lug")):
            nombres = []
            for e in acc.get(clave, []):
                adj = ",".join(e.get("adjetivos", []))
                nombres.append(f"{e.get('nombre', '')}({adj})" if adj else str(e.get("nombre", "")))
            if nombres:
                trozos.append(f"{tag}: " + ", ".join(nombres))
        partes.append(str(acc.get("accion", "?")) + (" [" + "; ".join(trozos) + "]" if trozos else ""))
    return " | ".join(partes)


# ----------------------------------------------------------------------------
# Dataset y ejecucion
# ----------------------------------------------------------------------------
def cargar_dataset(ruta, particion):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    ordenes = d["ordenes"]
    if particion != "todo":
        ordenes = [o for o in ordenes if o["particion"] == particion]
    return d, ordenes


def ver_dataset(d):
    print(d.get("nota", ""))
    print("\nConvenciones de anotacion:")
    for i, c in enumerate(d.get("convenciones", []), 1):
        print(f"  {i}. {c}")
    print()
    for o in d["ordenes"]:
        print(f"{o['id']} N{o['nivel']} {o['escenario']:<10} {o['particion']:<7} | {o['orden']}")
        print(f"      -> {compacto(o['gold'])}")


def evaluar_modelo(modelo, ordenes, formato, prompt, ejemplos, solo_cpu, postproceso, repeticiones):
    print(f"\n{'=' * 70}\nModelo: {modelo}\n{'=' * 70}")
    try:
        formato = P.elegir_formato(modelo, formato, prompt, ejemplos, solo_cpu)  # tambien calienta el modelo
    except ollama.ResponseError as e:
        print(f"  [SALTADO] {e}. Descargalo con: ollama pull {modelo}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    pares, latencias = [], []
    for i, o in enumerate(ordenes, 1):
        tiempos, bruta, fallo = [], None, None
        for _ in range(repeticiones):
            try:
                s, seg = P.extraer(modelo, o["orden"], formato, prompt, ejemplos, solo_cpu)
            except Exception as e:  # timeout u otro error: se cuenta como respuesta invalida y se sigue
                fallo = type(e).__name__
                s, seg = "", 180.0  # mismo limite que el timeout del cliente
            tiempos.append(seg)
            bruta = bruta or s
        pred = P.limpiar_salida(bruta) if postproceso else bruta
        latencias.append(statistics.median(tiempos))
        pares.append({"id": o["id"], "nivel": o["nivel"], "orden": o["orden"], "gold": o["gold"],
                      "pred_bruta": bruta, "pred": pred, "latencia_s": round(latencias[-1], 3)})
        extra = f"  [FALLO: {fallo}]" if fallo else ("" if a_dict(pred) is not None else "  [JSON invalido]")
        print(f"  {i}/{len(ordenes)} {o['id']} ({latencias[-1]:.1f} s){extra}", flush=True)

    m = metricas(pares)
    m["latencia_mediana_s"] = round(statistics.median(latencias), 3)
    total_gb, vram_gb = P.memoria_modelo(modelo)
    m["ram_modelo_gb"] = round(total_gb, 2) if total_gb is not None else None
    if vram_gb:
        print(f"  AVISO: {vram_gb:.2f} GB del modelo estan en GPU. Esta medicion NO es solo CPU.")
    P.descargar_modelo(modelo)
    return {"modelo": modelo, "metricas": m, "detalle": pares}


def mostrar_errores(res, maximo):
    malas = [p for p in res["detalle"]
             if elementos(p["gold"])[2] != elementos(p["pred"])[2] or a_dict(p["pred"]) is None]
    print(f"\n  {res['modelo']}: {len(malas)} de {len(res['detalle'])} ordenes con diferencias")
    for p in malas[:maximo]:
        print(f"   [{p['id']} N{p['nivel']}] {p['orden']}")
        print(f"      esperado: {compacto(p['gold'])}")
        print(f"      modelo  : {compacto(p['pred'])}")
    if len(malas) > maximo:
        print(f"   ... y {len(malas) - maximo} mas (usa --max-errores {len(malas)} para verlas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_lab7.json")
    ap.add_argument("--particion", choices=["ajuste", "prueba", "todo"], default="ajuste")
    ap.add_argument("--modelos", nargs="+", default=P.MODELOS)
    ap.add_argument("--version", choices=sorted(P.VERSIONES), default="v2")
    ap.add_argument("--sin-fewshot", action="store_true")
    ap.add_argument("--postproceso", action="store_true")
    ap.add_argument("--repeticiones", type=int, default=1)
    ap.add_argument("--sin-esquema", action="store_true")
    ap.add_argument("--permitir-gpu", action="store_true")
    ap.add_argument("--max-errores", type=int, default=6)
    ap.add_argument("--ver-errores", action="store_true",
                    help="mostrar errores tambien en la particion de prueba (no ajustes el prompt con ellos)")
    ap.add_argument("--ver-dataset", action="store_true", help="imprimir las anotaciones y salir")
    ap.add_argument("--ids", nargs="+", metavar="ID",
                    help="evaluar solo estas ordenes, de cualquier particion (ej. --ids O018 O028)")
    ap.add_argument("--leer", default=None, metavar="ARCHIVO",
                    help="mostrar los errores de un archivo de resultados ya guardado (no corre modelos)")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    if args.leer:
        with open(args.leer, encoding="utf-8") as f:
            guardado = json.load(f)
        print("Configuracion:", guardado.get("config"))
        for r in guardado["resultados"]:
            mostrar_errores(r, args.max_errores)
        return

    d, ordenes = cargar_dataset(args.dataset, args.particion)
    if args.ids:
        ordenes = [o for o in d["ordenes"] if o["id"] in set(args.ids)]
        if not ordenes:
            sys.exit(f"Ningun id coincide: {args.ids}")
    if args.ver_dataset:
        ver_dataset(d)
        return

    prompt, ejemplos = P.VERSIONES[args.version]
    if args.sin_fewshot:
        ejemplos = []
    formato = "json" if args.sin_esquema else P.ESQUEMA
    solo_cpu = not args.permitir_gpu

    resultados = [r for r in (evaluar_modelo(m, ordenes, formato, prompt, ejemplos, solo_cpu,
                                             args.postproceso, max(1, args.repeticiones))
                              for m in args.modelos) if r]

    config = {"version": args.version, "particion": args.particion, "ordenes": len(ordenes),
              "ejemplos": len(ejemplos), "postproceso": args.postproceso,
              "repeticiones": args.repeticiones, "solo_cpu": solo_cpu}
    etiqueta = args.version + ("_sinfs" if args.sin_fewshot else "") + ("_post" if args.postproceso else "")
    salida = args.salida or f"eval_{args.particion}_{etiqueta}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump({"config": config, "resultados": resultados}, f, ensure_ascii=False, indent=1)

    print(f"\nConfiguracion: {config}")
    cab = f"{'Modelo':<14}{'F1 prom':>8}{'Estruct':>9}{'Robust':>8}{'JSON%':>7}{'Exactas%':>9}{'Lat(s)':>8}{'RAM(GB)':>9}"
    print(cab)
    for r in resultados:
        m = r["metricas"]
        rob = "n/d" if m["robustez_f1"] is None else f"{m['robustez_f1']:.3f}"
        ram = "n/d" if m["ram_modelo_gb"] is None else f"{m['ram_modelo_gb']}"
        print(f"{r['modelo']:<14}{m['f1_promedio']:>8.3f}{m['estructura_f1']:>9.3f}{rob:>8}"
              f"{m['json_valido_pct']:>7}{m['exactas_pct']:>9}{m['latencia_mediana_s']:>8}{ram:>9}")
    print("\nF1 por categoria:")
    print(f"{'Modelo':<14}" + "".join(f"{c:>10}" for c in CATS))
    for r in resultados:
        fc = r["metricas"]["f1_categorias"]
        print(f"{r['modelo']:<14}" + "".join(f"{fc[c]:>10.3f}" for c in CATS))
    print("\nOrdenes exactas por nivel:")
    for r in resultados:
        print(f"  {r['modelo']:<14}" + "  ".join(f"N{k}:{v}" for k, v in r["metricas"]["exactas_por_nivel"].items()))
    print("\nRubrica (F1 promedio): >=0.85 excelente, 0.70-0.85 aceptable, <0.70 deficiente.")

    if args.particion == "prueba" and not args.ver_errores:
        print("\nPARTICION DE PRUEBA: no muestro los errores orden por orden para que no ajustes el prompt "
              "con ellos. Usa --ver-errores si necesitas analizarlos para el informe.")
    else:
        for r in resultados:
            mostrar_errores(r, args.max_errores)
    print(f"\nResultados completos en {salida}")


if __name__ == "__main__":
    main()
