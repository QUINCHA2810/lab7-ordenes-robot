"""
Laboratorio 7 - Demo en vivo: microfono -> Whisper -> LLM -> JSON -> respuesta -> voz (Piper).

    py -3.12 demo.py                       # configuracion final
    py -3.12 demo.py --sin-voz             # solo texto (si Piper falla)
    py -3.12 demo.py --whisper small       # otro Whisper

Enter para empezar a hablar, Enter al terminar. Escribe q + Enter para salir.
Tambien puedes escribir la orden en vez de hablarla: t + Enter y luego el texto.
La latencia que se muestra va de fin del audio (el segundo Enter) a inicio de la respuesta.
"""
import argparse
import json
import time

import numpy as np
import sounddevice as sd

import predecir as D
import probar_llms as P

SR = 16000

EJEMPLOS_ENSAYO = [
    "Trae el vaso",
    "Lleva el paquete grande a la oficina principal",
    "Ve a la cocina, toma la taza roja y llévasela a Marta",
    "oye porfa me pasas el control que dejé en la sala",
    "Tráeme eso",
]


def grabar():
    bloques = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: bloques.append(i.copy())):
        input("   >>> HABLA AHORA... (Enter al terminar)")
    return np.concatenate(bloques)[:, 0] if bloques else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=D.LLM_POR_DEFECTO)
    ap.add_argument("--whisper", default=D.WHISPER_POR_DEFECTO)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--sin-voz", action="store_true")
    ap.add_argument("--piper-modelo", default=str(D.CARPETA_MODELOS / "es_ES-davefx-medium.onnx"))
    args = ap.parse_args()

    prompt, ejemplos = P.VERSIONES["v2"]
    print("Cargando Whisper y el LLM...", flush=True)
    whisper = D.cargar_whisper(args.whisper)
    D.calentar_whisper(whisper)
    formato = P.elegir_formato(args.llm, P.ESQUEMA, prompt, ejemplos, solo_cpu=True)
    voz = None
    if not args.sin_voz:
        try:
            voz = D.Voz(args.piper_modelo)
            voz.decir("Listo.")
        except Exception as e:
            print(f"[Piper no disponible: {e}] Sigo solo con texto.")
            voz = None

    print("\nOrdenes de ensayo sugeridas:")
    for e in EJEMPLOS_ENSAYO:
        print("  -", e)
    while True:
        op = input("\nEnter = hablar | t = escribir | q = salir: ").strip().lower()
        if op == "q":
            break
        if op == "t":
            texto = input("   Orden: ").strip()
            t0 = time.perf_counter()
        else:
            audio = grabar()
            t0 = time.perf_counter()  # fin del audio
            texto = D.transcribir(whisper, audio, args.beam)
        t_s2t = time.perf_counter() - t0
        acciones, valido, error = D.entender(texto, args.llm, formato, prompt, ejemplos, True)
        respuesta = D.generar_respuesta(acciones, valido)
        t_llm = time.perf_counter() - t0 - t_s2t
        print(f"   Transcripcion: \"{texto}\"")
        print(json.dumps({"transcripcion": texto, "acciones": acciones, "respuesta": respuesta},
                         ensure_ascii=False, indent=2))
        if voz:
            ondas = voz.sintetizar(respuesta)
            lat = time.perf_counter() - t0
            print(f"   [{lat:.2f} s hasta inicio de la voz: S2T {t_s2t:.2f} + LLM {t_llm:.2f} "
                  f"+ TTS {lat - t_s2t - t_llm:.2f}]")
            voz.reproducir(ondas)
        else:
            print(f"   [{time.perf_counter() - t0:.2f} s: S2T {t_s2t:.2f} + LLM {t_llm:.2f}]")


if __name__ == "__main__":
    main()
