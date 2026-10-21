"""
Laboratorio 7 - Sistema completo de comprension de ordenes en español.

    audio --(faster-whisper)--> texto --(LLM ligero via Ollama)--> JSON --> respuesta --(Piper, opcional)--> voz

Comando unico (entregable de la guia):
    python predecir.py --entrada audios/ --salida predicciones.json

Otros usos:
    python predecir.py --texto "Trae el vaso"                  # prueba rapida sin audio (solo LLM)
    python predecir.py --entrada audios/ --whisper base        # S2T mas rapido
    python predecir.py --entrada audios/ --llm qwen2.5:3b      # LLM mas grande
    python predecir.py --entrada audios/ --hablar              # tambien genera la respuesta hablada (Piper)
    python predecir.py --entrada audios/ --anotaciones audios/anotaciones.json
        # extremo a extremo: F1 con la transcripcion real vs con el texto correcto, y WER

Requiere: Ollama abierto con el modelo descargado (ollama pull qwen2.5:1.5b) y probar_llms.py en la misma carpeta.
Todo corre en CPU y sin internet durante la inferencia (los modelos de voz se leen de modelos/;
si no estan, faster-whisper descarga el suyo la primera vez).
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

import probar_llms as P

CARPETA_MODELOS = Path(__file__).parent / "modelos"
# Configuracion final (elegida con los datos de eval_* y e2e_*; ver README e informe)
LLM_POR_DEFECTO = "qwen2.5:3b"  # 45/80 pts estimados vs 40 (llama3.2:3b) y 34 (qwen2.5:1.5b); ver tablas.py
WHISPER_POR_DEFECTO = "small"
EXT_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


# ----------------------------------------------------------------------------
# Respuesta hablada (plantillas): confirma lo entendido o pregunta lo que falta
# ----------------------------------------------------------------------------
_FEMENINAS = {"luz", "mano", "llave", "gente", "calle", "noche", "carne", "leche", "clase", "nube",
              "torre", "parte", "fuente", "sede", "vez", "pared", "flor", "sal", "tarde"}
_MASCULINAS_EN_A = {"día", "mapa", "planeta", "problema", "sistema", "tema", "idioma", "programa", "clima"}
_SIN_ARTICULO = {"papá", "mamá"}
_TERM_FEM = ("ción", "sión", "dad", "tad", "tud", "umbre")
_DESTINO = {"ir", "llevar", "traer", "acompañar", "guiar", "entregar", "dar", "enviar", "mandar"}

_PREGUNTAS = {
    "traer": "¿Qué quieres que te traiga?",
    "llevar": "¿Qué quieres que lleve y adónde?",
    "dar": "¿Qué quieres que te dé?",
    "entregar": "¿Qué quieres que entregue y a quién?",
    "buscar": "¿Qué o a quién quieres que busque?",
    "poner": "¿Qué quieres que ponga y dónde?",
    "guardar": "¿Qué quieres que guarde y dónde?",
    "abrir": "¿Qué quieres que abra?",
    "cerrar": "¿Qué quieres que cierre?",
    "apagar": "¿Qué quieres que apague?",
    "encender": "¿Qué quieres que encienda?",
    "avisar": "¿A quién quieres que avise?",
    "saludar": "¿A quién quieres que salude?",
    "llamar": "¿A quién quieres que llame?",
    "ir": "¿A dónde quieres que vaya?",
}


def _articulo(nombre):
    """Articulo definido segun una heuristica simple. Vacio para nombres propios (Carlos, Laura)."""
    primera = nombre.split()[0]
    if primera[0].isupper() or primera.lower() in _SIN_ARTICULO:
        return ""
    p = primera.lower()
    plural = len(p) > 3 and p.endswith("s")
    base = p[:-1] if plural else p
    femenino = (base in _FEMENINAS or base.endswith(_TERM_FEM)
                or (base.endswith("a") and base not in _MASCULINAS_EN_A))
    if plural:
        return "las" if femenino else "los"
    return "la" if femenino else "el"


def _sintagma(entidad):
    """{'nombre': 'taza', 'adjetivos': ['roja']} -> 'la taza roja'."""
    nombre = str(entidad.get("nombre", "")).strip()
    if not nombre:
        return ""
    partes = [_articulo(nombre), nombre] + [str(a).strip() for a in entidad.get("adjetivos", [])]
    return " ".join(x for x in partes if x)


def _unir(textos):
    textos = [t for t in textos if t]
    if len(textos) <= 1:
        return "".join(textos)
    return ", ".join(textos[:-1]) + " y " + textos[-1]


def _frase_accion(acc):
    verbo = str(acc.get("accion", "")).strip()
    partes = [verbo]
    objetos = _unir([_sintagma(e) for e in acc.get("objetos", [])])
    personas = _unir(["a " + _sintagma(e) for e in acc.get("personas", [])
                      if _sintagma(e)])
    lugares = _unir([_sintagma(e) for e in acc.get("lugares", [])])
    if objetos:
        partes.append(objetos)
    if personas:
        partes.append(personas)
    if lugares:
        partes.append(("a " if verbo in _DESTINO else "en ") + lugares)
    return " ".join(partes)


def generar_respuesta(acciones, json_valido=True):
    """Texto de la respuesta hablada. Pregunta cuando una accion no tiene nada que la complete."""
    if not json_valido:
        return "No pude entender la orden. ¿Puedes repetirla?"
    if not acciones:
        return "No entendí ninguna orden."
    for acc in acciones:
        if not (acc.get("objetos") or acc.get("personas") or acc.get("lugares")):
            verbo = str(acc.get("accion", "")).strip()
            return _PREGUNTAS.get(verbo, f"Me falta información para {verbo}. ¿Puedes darme más detalles?")
    frase = "Voy a " + _unir([_frase_accion(a) for a in acciones])
    frase = re.sub(r"\ba el\b", "al", frase)
    return re.sub(r"\bde el\b", "del", frase)


# ----------------------------------------------------------------------------
# Componentes de voz
# ----------------------------------------------------------------------------
def cargar_whisper(tamano):
    from faster_whisper import WhisperModel  # import tardio: --texto no lo necesita
    local = CARPETA_MODELOS / f"whisper-{tamano}"  # copia local: no necesita internet
    return WhisperModel(str(local) if (local / "model.bin").exists() else tamano,
                        device="cpu", compute_type="int8")


def calentar_whisper(modelo):
    """Primera transcripcion con 1 s de silencio, para no medir la inicializacion en la primera orden."""
    try:
        import numpy as np
        segs, _ = modelo.transcribe(np.zeros(16000, dtype=np.float32), language="es")
        list(segs)
    except Exception:
        pass


def leer_audio(ruta):
    """Archivo -> float32 mono 16 kHz. WAV/FLAC/OGG se leen con soundfile: faster-whisper 1.2.1 falla al
    decodificar archivos con PyAV >= 15 ("unexpected keyword argument 'metadata_errors'").
    Otros formatos (mp3, m4a...) se dejan a faster-whisper."""
    import numpy as np
    import soundfile as sf
    try:
        datos, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
    except Exception:
        return str(ruta)
    datos = datos.mean(axis=1)
    if sr != 16000:  # remuestreo lineal (las grabaciones de grabar.py ya vienen a 16 kHz)
        n = int(round(len(datos) * 16000 / sr))
        datos = np.interp(np.linspace(0, len(datos) - 1, n), np.arange(len(datos)), datos).astype(np.float32)
    return datos


def transcribir(modelo, ruta, beam):
    """ruta puede ser un archivo o un arreglo float32 de 16 kHz (microfono)."""
    audio = leer_audio(ruta) if isinstance(ruta, (str, Path)) else ruta
    segmentos, _ = modelo.transcribe(audio, language="es", beam_size=beam)
    return " ".join(s.text.strip() for s in segmentos).strip()


class Voz:
    """Piper con la API de Python: el modelo se carga una vez y cada respuesta tarda poco."""

    def __init__(self, modelo_onnx):
        from piper import PiperVoice
        self.voz = PiperVoice.load(str(modelo_onnx))

    def sintetizar(self, texto):
        """-> (muestras int16, frecuencia de muestreo)."""
        import numpy as np
        trozos = list(self.voz.synthesize(texto))
        return np.concatenate([t.audio_int16_array for t in trozos]), trozos[0].sample_rate

    def guardar(self, ondas, salida):
        import soundfile as sf
        sf.write(str(salida), ondas[0], ondas[1], subtype="PCM_16")

    def reproducir(self, ondas):
        import sounddevice as sd
        sd.play(ondas[0], ondas[1])
        sd.wait()

    def decir(self, texto):
        self.reproducir(self.sintetizar(texto))


def sintetizar(texto, modelo_onnx, salida):
    """Alternativa: Piper por linea de comandos (lee el texto por stdin)."""
    exe = shutil.which("piper")
    base = [exe] if exe else [sys.executable, "-m", "piper"]
    # Las opciones de Piper cambiaron entre versiones: se prueban las variantes conocidas.
    variantes = (["--model", str(modelo_onnx), "--output_file", str(salida)],
                 ["--model", str(modelo_onnx), "--output-file", str(salida)],
                 ["-m", str(modelo_onnx), "-f", str(salida)])
    ultimo_error = ""
    for opciones in variantes:
        r = subprocess.run(base + opciones, input=texto.encode("utf-8"), capture_output=True, timeout=120)
        if r.returncode == 0:
            return
        ultimo_error = r.stderr.decode("utf-8", errors="ignore")[-300:]
    raise RuntimeError(ultimo_error)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def entender(transcripcion, llm, formato, prompt, ejemplos, postproceso):
    """Texto -> (acciones, json_valido, error). Nunca lanza excepcion: un fallo del LLM no tumba el lote."""
    try:
        crudo, _ = P.extraer(llm, transcripcion, formato, prompt, ejemplos, solo_cpu=True)
    except Exception as e:
        return [], False, f"{type(e).__name__}: {e}"
    salida = P.limpiar_salida(crudo) if postproceso else crudo
    if not P.es_json_valido(salida):
        return [], False, "JSON invalido"
    return json.loads(salida)["acciones"], True, None


def ram_pico_proceso(proc):
    """Pico de memoria de este proceso (Python + Whisper). En Windows, peak_wset es el pico real."""
    m = proc.memory_info()
    return getattr(m, "peak_wset", None) or m.rss


def evaluar_extremo(resultados, anotaciones, llm, formato, prompt, ejemplos, postproceso):
    """Compara contra el JSON anotado: F1 con la transcripcion real y con el texto correcto (mismo LLM).
    La diferencia es el error que propaga la voz a texto."""
    import jiwer
    import evaluar as E
    from evaluar_voz import normalizar
    real, perfecto, refs, hips = [], [], [], []
    for r in resultados:
        clave = Path(r["archivo"]).stem if r["archivo"] else None
        if clave not in anotaciones:
            continue
        a = anotaciones[clave]
        acc_txt, valido_txt, _ = entender(a["orden"], llm, formato, prompt, ejemplos, postproceso)
        pred_real = {"acciones": r["acciones"]} if r["metricas"]["json_valido"] else "invalido"
        pred_txt = {"acciones": acc_txt} if valido_txt else "invalido"
        real.append({"id": clave, "nivel": a["nivel"], "gold": a["gold"], "pred": pred_real})
        perfecto.append({"id": clave, "nivel": a["nivel"], "gold": a["gold"], "pred": pred_txt})
        refs.append(normalizar(a["orden"]))
        hips.append(normalizar(r["transcripcion"]) or "<vacio>")
        r["referencia"] = a["orden"]
    if not real:
        return None
    return {"n": len(real), "wer_pct": round(100 * jiwer.wer(refs, hips), 1),
            "con_transcripcion": E.metricas(real), "con_texto_correcto": E.metricas(perfecto),
            "detalle_texto_correcto": perfecto}


def listar_audios(carpeta):
    ruta = Path(carpeta)
    if not ruta.is_dir():
        sys.exit(f"No existe la carpeta de entrada: {carpeta}")
    archivos = sorted(p for p in ruta.iterdir() if p.suffix.lower() in EXT_AUDIO)
    if not archivos:
        sys.exit(f"No encontre audios ({', '.join(sorted(EXT_AUDIO))}) en {carpeta}")
    return archivos


def main():
    ap = argparse.ArgumentParser(description="Comprension de ordenes en español: audio -> JSON -> respuesta")
    ap.add_argument("--entrada", help="carpeta con los audios de las ordenes")
    ap.add_argument("--texto", help="una orden escrita (salta la transcripcion; sirve para probar el LLM)")
    ap.add_argument("--salida", default="predicciones.json")
    ap.add_argument("--llm", default=LLM_POR_DEFECTO, help="modelo de Ollama")
    ap.add_argument("--whisper", default=WHISPER_POR_DEFECTO, help="tiny, base, small o medium")
    ap.add_argument("--beam", type=int, default=5, help="beam de Whisper (5 = WER 12.7% vs 13.2% con 1, misma latencia)")
    ap.add_argument("--version", choices=sorted(P.VERSIONES), default="v2", help="prompt y ejemplos")
    ap.add_argument("--sin-postproceso", action="store_true", help="no limpiar la salida del LLM")
    ap.add_argument("--hablar", action="store_true", help="sintetizar la respuesta con Piper")
    ap.add_argument("--piper-modelo", default=str(CARPETA_MODELOS / "es_ES-davefx-medium.onnx"))
    ap.add_argument("--anotaciones", default=None,
                    help="JSON id -> {orden, nivel, gold} (lo crea grabar.py en audios/anotaciones.json) "
                         "para medir F1 de extremo a extremo y WER")
    ap.add_argument("--carpeta-voz", default="respuestas_voz")
    args = ap.parse_args()

    if not args.entrada and not args.texto:
        ap.error("indica --entrada <carpeta con audios> o --texto \"<orden>\"")

    prompt, ejemplos = P.VERSIONES[args.version]
    formato = P.ESQUEMA
    postproceso = not args.sin_postproceso

    if args.texto:
        items = [(None, args.texto)]
        whisper = None
    else:
        items = [(p, None) for p in listar_audios(args.entrada)]
        print(f"Cargando Whisper '{args.whisper}' (CPU, int8)...", flush=True)
        whisper = cargar_whisper(args.whisper)
        calentar_whisper(whisper)

    print(f"Calentando el LLM '{args.llm}'...", flush=True)
    try:
        formato = P.elegir_formato(args.llm, formato, prompt, ejemplos, solo_cpu=True)
    except Exception as e:
        sys.exit(f"No pude usar el LLM '{args.llm}': {e}\n"
                 f"Revisa que Ollama este abierto y que el modelo exista (ollama pull {args.llm}).")

    voz = None
    if args.hablar:
        Path(args.carpeta_voz).mkdir(exist_ok=True)
        try:
            voz = Voz(args.piper_modelo)
            voz.sintetizar("Listo.")  # calentar
        except Exception as e:
            print(f"  [Piper (API de Python) no disponible: {e}; uso la linea de comandos]")

    proc = psutil.Process()
    resultados = []
    for i, (ruta, texto) in enumerate(items, 1):
        t0 = time.perf_counter()
        transcripcion = texto if ruta is None else transcribir(whisper, ruta, args.beam)
        t_s2t = time.perf_counter() - t0

        t1 = time.perf_counter()
        acciones, valido, error = entender(transcripcion, args.llm, formato, prompt, ejemplos, postproceso)
        t_llm = time.perf_counter() - t1

        respuesta = generar_respuesta(acciones, valido)

        audio_resp, t_tts = None, 0.0
        if args.hablar:
            t2 = time.perf_counter()
            nombre = (ruta.stem if ruta else "orden") + ".wav"
            destino = Path(args.carpeta_voz) / nombre
            try:
                if voz:
                    ondas = voz.sintetizar(respuesta)
                    t_tts = time.perf_counter() - t2  # aqui ya podria empezar a sonar
                    voz.guardar(ondas, destino)
                else:
                    sintetizar(respuesta, args.piper_modelo, destino)
                    t_tts = time.perf_counter() - t2
                audio_resp = str(destino)
            except Exception as e:
                print(f"  [Piper fallo: {e}]")

        latencia = t_s2t + t_llm + t_tts  # fin del audio -> inicio de la respuesta (incluye voz si --hablar)
        etiqueta = ruta.name if ruta else "texto"
        print(f"  {i}/{len(items)} {etiqueta}: {latencia:.1f} s | \"{transcripcion}\" -> {respuesta}"
              + ("" if valido else f"  [{error}]"), flush=True)

        resultados.append({
            "archivo": ruta.name if ruta else None,
            "transcripcion": transcripcion,
            "acciones": acciones,
            "respuesta": respuesta,
            "metricas": {"latencia_s": round(latencia, 3), "s2t_s": round(t_s2t, 3),
                         "llm_s": round(t_llm, 3), "tts_s": round(t_tts, 3),
                         "json_valido": valido, "error": error, "audio_respuesta": audio_resp},
        })

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    pico_rss = ram_pico_proceso(proc)
    ram_llm, _ = P.memoria_modelo(args.llm)
    resumen = None
    if args.anotaciones and args.entrada:
        with open(args.anotaciones, encoding="utf-8") as f:
            anot = json.load(f)
        print("Evaluando contra las anotaciones (corre el LLM tambien sobre el texto correcto)...", flush=True)
        resumen = evaluar_extremo(resultados, anot, args.llm, formato, prompt, ejemplos, postproceso)

    lat = [r["metricas"]["latencia_s"] for r in resultados]
    validos = sum(r["metricas"]["json_valido"] for r in resultados)
    ram_total = pico_rss / 1e9 + (ram_llm or 0)
    print(f"\n{len(resultados)} ordenes | latencia mediana {statistics.median(lat):.2f} s "
          f"(max {max(lat):.1f} s) | JSON valido {100 * validos / len(resultados):.0f}%")
    print(f"RAM pico aproximada del sistema: {ram_total:.2f} GB "
          f"(pico Python+Whisper {pico_rss / 1e9:.2f} GB + LLM en Ollama {ram_llm or 0:.2f} GB)")
    if resumen:
        a, b = resumen["con_transcripcion"], resumen["con_texto_correcto"]
        print(f"\nExtremo a extremo ({resumen['n']} audios) | WER {resumen['wer_pct']}%")
        print(f"{'':<22}{'F1 prom':>8}{'Estruct':>9}{'Robust':>8}{'JSON%':>7}{'Exactas%':>9}")
        for nombre, m in (("con transcripcion", a), ("con texto correcto", b)):
            rob = "n/d" if m["robustez_f1"] is None else f"{m['robustez_f1']:.3f}"
            print(f"{nombre:<22}{m['f1_promedio']:>8.3f}{m['estructura_f1']:>9.3f}{rob:>8}"
                  f"{m['json_valido_pct']:>7}{m['exactas_pct']:>9}")
        resumen.update({"config": {"llm": args.llm, "whisper": args.whisper, "beam": args.beam,
                                   "version": args.version, "postproceso": postproceso,
                                   "entrada": args.entrada},
                        "latencia_mediana_s": round(statistics.median(lat), 3),
                        "latencia_max_s": round(max(lat), 3),
                        "ram_python_whisper_gb": round(pico_rss / 1e9, 2),
                        "ram_llm_gb": round(ram_llm or 0, 2), "ram_total_gb": round(ram_total, 2),
                        "predicciones": resultados})
        nombre = f"e2e_{args.llm.replace(':', '-')}_whisper-{args.whisper}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(nombre, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=1)
        print(f"Evaluacion extremo a extremo en {nombre}")
    print(f"Predicciones en {args.salida}")


if __name__ == "__main__":
    main()
