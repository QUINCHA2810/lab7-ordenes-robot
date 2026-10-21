"""
Laboratorio 7 - Grabacion de las 30 ordenes de la particion 'prueba' (16 kHz, mono, WAV).

    py -3.12 grabar.py                    # graba todas las que falten
    py -3.12 grabar.py --ids O033 O040    # vuelve a grabar solo esas
    py -3.12 grabar.py --todas            # regraba todo (pregunta antes de sobrescribir cada una)

Para cada orden: muestra el texto, Enter para empezar, Enter para terminar,
luego [Enter]=guardar, r=repetir, e=escuchar, s=saltar, q=salir.
Guarda audios/O0xx.wav y audios/referencias.json (id -> texto) y audios/anotaciones.json (id -> JSON anotado).
No hagas clic dentro de la consola mientras grabas (el modo QuickEdit la congela).
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000


def grabar_una():
    """Graba desde que se pulsa Enter hasta el siguiente Enter. Devuelve un arreglo float32."""
    bloques, parar = [], threading.Event()

    def cb(indata, frames, t, status):
        bloques.append(indata.copy())

    input("   Enter para EMPEZAR a grabar...")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        print("   >>> GRABANDO... (Enter para terminar)")
        input()
    audio = np.concatenate(bloques)[:, 0] if bloques else np.zeros(0, dtype=np.float32)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_lab7.json")
    ap.add_argument("--carpeta", default="audios")
    ap.add_argument("--particion", default="prueba")
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--todas", action="store_true")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        ordenes = [o for o in json.load(f)["ordenes"] if o["particion"] == args.particion]
    carpeta = Path(args.carpeta)
    carpeta.mkdir(exist_ok=True)
    # Referencias y anotaciones de todas las ordenes de la particion (texto correcto y JSON esperado)
    with open(carpeta / "referencias.json", "w", encoding="utf-8") as f:
        json.dump({o["id"]: o["orden"] for o in ordenes}, f, ensure_ascii=False, indent=1)
    with open(carpeta / "anotaciones.json", "w", encoding="utf-8") as f:
        json.dump({o["id"]: {"nivel": o["nivel"], "orden": o["orden"], "gold": o["gold"]} for o in ordenes},
                  f, ensure_ascii=False, indent=1)

    if args.ids:
        ordenes = [o for o in ordenes if o["id"] in set(args.ids)]

    try:
        print("Microfono:", sd.query_devices(kind="input")["name"])
    except Exception as e:
        sys.exit(f"No encuentro microfono: {e}")
    print("Consejos: microfono cerca (20-30 cm), sin ruido de fondo, lee la orden tal cual, "
          "con tu voz normal.\n")

    for i, o in enumerate(ordenes, 1):
        destino = carpeta / f"{o['id']}.wav"
        if destino.exists() and not (args.ids or args.todas):
            continue
        print(f"[{i}/{len(ordenes)}] {o['id']} (nivel {o['nivel']}):\n\n    \"{o['orden']}\"\n")
        while True:
            audio = grabar_una()
            dur = len(audio) / SR
            pico = float(np.abs(audio).max()) if len(audio) else 0.0
            aviso = "  (MUY BAJO: acercate al microfono)" if pico < 0.05 else ""
            print(f"   {dur:.1f} s, nivel pico {pico:.2f}{aviso}")
            op = input("   [Enter]=guardar  r=repetir  e=escuchar  s=saltar  q=salir: ").strip().lower()
            if op == "e":
                sd.play(audio, SR); sd.wait()
                op = input("   [Enter]=guardar  r=repetir  s=saltar  q=salir: ").strip().lower()
            if op == "r":
                continue
            if op == "q":
                print("Salida. Lo grabado queda guardado; vuelve a correr el script para seguir.")
                return
            if op != "s":
                sf.write(destino, audio, SR, subtype="PCM_16")
                print(f"   guardado {destino}\n")
            break

    faltan = [o["id"] for o in ordenes if not (carpeta / f"{o['id']}.wav").exists()]
    print("Listo." if not faltan else f"Faltan: {' '.join(faltan)}")


if __name__ == "__main__":
    main()
