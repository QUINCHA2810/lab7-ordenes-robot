"""
Laboratorio 7 - Tablas del informe a partir de los archivos de resultados guardados (no corre modelos).

    py -3.12 tablas.py                 # usa los archivos finales por defecto
    py -3.12 tablas.py --json tablas.json

Toda cifra del informe sale de aqui. La fila "sin post-proceso" se recalcula sobre las salidas brutas
guardadas (pred_bruta) de la misma corrida, asi que aisla el efecto del post-proceso.
"""
import argparse
import glob
import json
import os
import statistics

import evaluar as E
import probar_llms as P

LLM_V2 = "eval_prueba_v2_post_4modelos_final.json"
LLM_V1 = "eval_prueba_v1_qwen15_final.json"
TAMANO_GB = {"qwen2.5:1.5b": 0.99, "qwen2.5:3b": 1.9, "gemma2:2b": 1.6, "llama3.2:3b": 2.0}  # ollama list
PARAMS = {"qwen2.5:1.5b": "1.5B", "qwen2.5:3b": "3.1B", "gemma2:2b": "2.6B", "llama3.2:3b": "3.2B"}


def ultimo(patron):
    archivos = sorted(glob.glob(patron), key=os.path.getmtime)
    return archivos[-1] if archivos else None


def fila_llm(r, etiqueta=None):
    m = r["metricas"]
    return {"modelo": etiqueta or r["modelo"], "f1": m["f1_promedio"], "estructura": m["estructura_f1"],
            "robustez": m["robustez_f1"], "json": m["json_valido_pct"], "exactas": m["exactas_pct"],
            "latencia": m.get("latencia_mediana_s"), "ram": m.get("ram_modelo_gb"),
            "tamano": TAMANO_GB.get(r["modelo"]), "params": PARAMS.get(r["modelo"]),
            "f1_cat": m["f1_categorias"], "por_nivel": m["exactas_por_nivel"]}


def sin_post(r):
    pares = [{**p, "pred": p["pred_bruta"]} for p in r["detalle"]]
    m = E.metricas(pares)
    m["latencia_mediana_s"] = r["metricas"]["latencia_mediana_s"]
    m["ram_modelo_gb"] = r["metricas"]["ram_modelo_gb"]
    return {"modelo": r["modelo"], "metricas": m}


def f1_por_nivel(r):
    out = {}
    for nivel in sorted({p["nivel"] for p in r["detalle"]}):
        m = E.metricas([p for p in r["detalle"] if p["nivel"] == nivel])
        out[nivel] = m["f1_promedio"]
    return out


def nivel_rubrica(valor, excelente, aceptable, mayor_es_mejor=True):
    """5 = excelente, 3 = aceptable, 1 = deficiente; 2 en los huecos que la rubrica no define
    (latencia 5-8 s, memoria 6-8 GB)."""
    if valor is None:
        return None
    if mayor_es_mejor:
        return 5 if valor >= excelente else 3 if valor >= aceptable else 1
    e, (a_min, a_max), d = excelente, aceptable, None
    if valor <= e:
        return 5
    if valor <= a_max:
        return 3
    return 2 if a_max < valor < a_min else 1


def puntos_rubrica(e):
    """Estimacion de los criterios medibles (sin respuesta hablada ni informe), con las cifras de audio."""
    crit = {  # (peso, nivel)
        "F1 (25)": (25, nivel_rubrica(e["f1_audio"], 0.85, 0.70)),
        "Estructura (10)": (10, nivel_rubrica(e["estr_audio"], 0.95, 0.80)),
        "Robustez (10)": (10, nivel_rubrica(e["rob_audio"], 0.75, 0.60)),
        "JSON (5)": (5, 5 if e["json_audio"] > 98 else 3 if e["json_audio"] >= 90 else 1),
        "WER (10)": (10, 5 if e["wer"] < 10 else 3 if e["wer"] <= 20 else 1),
        "Latencia (15)": (15, nivel_rubrica(e["latencia"], 2, (8, 5), False)),
        "Memoria (5)": (5, nivel_rubrica(e["ram_total"], 3, (8, 6), False)),
    }
    pts = {k: round(p * n / 5, 1) for k, (p, n) in crit.items()}
    pts["Total (de 80)"] = round(sum(pts.values()), 1)
    return pts


def imprimir(titulo, filas, cols):
    print(f"\n## {titulo}")
    print(" | ".join(c for c, _ in cols))
    for f in filas:
        print(" | ".join("n/d" if f.get(k) is None else str(f.get(k)) for _, k in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=LLM_V2)
    ap.add_argument("--v1", default=LLM_V1)
    ap.add_argument("--voz", default=None, help="eval_voz_*.json (por defecto el mas reciente de audios)")
    ap.add_argument("--e2e", nargs="*", default=None, help="e2e_*.json (por defecto todos)")
    ap.add_argument("--json", default=None, help="guardar las tablas en este archivo")
    args = ap.parse_args()
    tablas = {}

    with open(args.llm, encoding="utf-8") as f:
        v2 = json.load(f)["resultados"]
    tablas["llms"] = [fila_llm(r) for r in v2]
    tablas["f1_por_nivel"] = {r["modelo"]: f1_por_nivel(r) for r in v2}
    cols = [("Modelo", "modelo"), ("Parametros", "params"), ("F1 prom", "f1"), ("Estructura", "estructura"),
            ("Robustez N4-5", "robustez"), ("JSON %", "json"), ("Exactas %", "exactas"),
            ("Latencia med (s)", "latencia"), ("RAM modelo (GB)", "ram"), ("Tamano disco (GB)", "tamano")]
    imprimir("LLMs, particion prueba, v2 + post-proceso, texto correcto", tablas["llms"], cols)
    print("\nF1 por categoria:")
    for f in tablas["llms"]:
        print(f"  {f['modelo']:<14}", f["f1_cat"])
    print("F1 promedio por nivel:")
    for k, v in tablas["f1_por_nivel"].items():
        print(f"  {k:<14}", v)

    abl = []
    q15 = next((r for r in v2 if r["modelo"] == "qwen2.5:1.5b"), None)
    if os.path.exists(args.v1):
        with open(args.v1, encoding="utf-8") as f:
            for r in json.load(f)["resultados"]:
                abl.append(fila_llm(r, f"{r['modelo']} v1"))
    if q15:
        abl.append(fila_llm(sin_post(q15), "qwen2.5:1.5b v2 sin post"))
        abl.append(fila_llm(q15, "qwen2.5:1.5b v2 + post"))
    tablas["ablacion"] = abl
    imprimir("Ablacion (qwen2.5:1.5b, prueba)", abl, cols[:1] + cols[2:8])

    voz = args.voz or ultimo("eval_voz_audios_2*.json")  # beam 1 (los de beam 5 van aparte)
    if voz:
        with open(voz, encoding="utf-8") as f:
            vr = json.load(f)["resultados"]
        tablas["voz"] = [{"motor": r["motor"], "wer": r["wer_pct"], "latencia": r["latencia_mediana_s"],
                          "rtf": r["rtf"], "n": r["n"]} for r in vr]
        tablas["voz_archivo"] = voz
        imprimir(f"Voz a texto ({voz})", tablas["voz"],
                 [("Motor", "motor"), ("N", "n"), ("WER %", "wer"), ("Latencia med (s)", "latencia"), ("RTF", "rtf")])

    beam5 = ultimo("eval_voz_audios_beam5_*.json")
    if beam5:
        with open(beam5, encoding="utf-8") as f:
            tablas["voz_beam5"] = [{"motor": r["motor"], "wer": r["wer_pct"], "latencia": r["latencia_mediana_s"],
                                    "rtf": r["rtf"], "n": r["n"]} for r in json.load(f)["resultados"]]
        imprimir(f"Voz a texto, beam 5 ({beam5})", tablas["voz_beam5"],
                 [("Motor", "motor"), ("N", "n"), ("WER %", "wer"), ("Latencia med (s)", "latencia"), ("RTF", "rtf")])

    e2e_archivos = args.e2e if args.e2e is not None else sorted(glob.glob("e2e_*.json"), key=os.path.getmtime)
    filas = []
    for a in e2e_archivos:
        with open(a, encoding="utf-8") as f:
            d = json.load(f)
        if not str(d["config"].get("entrada", "")).rstrip("/\\").endswith("audios"):
            continue
        c, t = d["con_transcripcion"], d["con_texto_correcto"]
        filas.append({"config": f"{d['config']['llm']} + whisper-{d['config']['whisper']}", "archivo": a,
                      "wer": d["wer_pct"], "f1_audio": c["f1_promedio"], "f1_texto": t["f1_promedio"],
                      "estr_audio": c["estructura_f1"], "rob_audio": c["robustez_f1"], "rob_texto": t["robustez_f1"],
                      "json_audio": c["json_valido_pct"], "latencia": d["latencia_mediana_s"],
                      "lat_max": d["latencia_max_s"], "ram_total": d["ram_total_gb"],
                      "ram_llm": d["ram_llm_gb"], "ram_python": d["ram_python_whisper_gb"],
                      "lat_s2t": round(statistics.median(p["metricas"]["s2t_s"] for p in d["predicciones"]), 3),
                      "lat_llm": round(statistics.median(p["metricas"]["llm_s"] for p in d["predicciones"]), 3),
                      "lat_tts": round(statistics.median(p["metricas"]["tts_s"] for p in d["predicciones"]), 3)})
        filas[-1]["rubrica"] = puntos_rubrica(filas[-1])
    if filas:
        tablas["e2e"] = filas
        imprimir("Extremo a extremo (audios grabados)", filas,
                 [("Config", "config"), ("WER %", "wer"), ("F1 audio", "f1_audio"), ("F1 texto", "f1_texto"),
                  ("Estructura audio", "estr_audio"), ("Robustez audio", "rob_audio"), ("JSON %", "json_audio"),
                  ("Lat med (s)", "latencia"), ("Lat max (s)", "lat_max"), ("RAM total (GB)", "ram_total")])
        print("Mediana por etapa (s):")
        for f in filas:
            print(f"  {f['config']:<32} S2T {f['lat_s2t']}  LLM {f['lat_llm']}  TTS {f['lat_tts']}")
        print("Estimacion de puntos de la rubrica (criterios medibles):")
        for f in filas:
            print(f"  {f['config']:<32}", f["rubrica"])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(tablas, f, ensure_ascii=False, indent=1)
        print(f"\nTablas en {args.json}")


if __name__ == "__main__":
    main()
