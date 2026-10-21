"""
Laboratorio 7 - Evaluacion de voz a texto (S2T): WER y latencia.

Compara faster-whisper (tiny, base, small; int8, CPU, language="es") y Vosk (vosk-model-small-es-0.42)
sobre los WAV de una carpeta contra su referencias.json (id -> texto correcto).

    py -3.12 evaluar_voz.py                                   # audios/ con todos los motores
    py -3.12 evaluar_voz.py --motores whisper-base vosk       # solo algunos
    py -3.12 evaluar_voz.py --entrada audios_sinteticos/      # otra carpeta

WER con jiwer tras normalizar: minusculas, sin tildes (la n con virgulilla se conserva) y sin puntuacion.
Latencia = tiempo de transcribir el archivo ya cargado el modelo (el modelo se carga y calienta antes).
Guarda eval_voz_<carpeta>_<fecha>.json (no sobrescribe nada).
"""
import argparse
import json
import re
import statistics
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

import jiwer

MOTORES = ["whisper-tiny", "whisper-base", "whisper-small", "vosk"]
CARPETA_MODELOS = Path(__file__).parent / "modelos"
VOSK_MODELO = CARPETA_MODELOS / "vosk-model-small-es-0.42"


def normalizar(texto):
    t = str(texto).lower().replace("ñ", "\0")
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").replace("\0", "ñ")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ruta_whisper(tamano):
    """Carpeta local en modelos/ si existe (sin internet); si no, el nombre (se descarga)."""
    local = CARPETA_MODELOS / f"whisper-{tamano}"
    return str(local) if (local / "model.bin").exists() else tamano


class Whisper:
    def __init__(self, tamano, beam=1):
        from faster_whisper import WhisperModel
        import numpy as np
        self.m = WhisperModel(ruta_whisper(tamano), device="cpu", compute_type="int8")
        self.beam = beam
        list(self.m.transcribe(np.zeros(16000, dtype=np.float32), language="es")[0])  # calentar

    def __call__(self, ruta):
        from predecir import leer_audio  # evita el fallo de PyAV >= 15 con faster-whisper 1.2.1
        segs, _ = self.m.transcribe(leer_audio(ruta), language="es", beam_size=self.beam)
        return " ".join(s.text.strip() for s in segs).strip()


class Vosk:
    def __init__(self, carpeta=VOSK_MODELO):
        import vosk
        vosk.SetLogLevel(-1)
        if not Path(carpeta).is_dir():
            raise FileNotFoundError(f"No existe {carpeta}: descarga vosk-model-small-es-0.42 en modelos/")
        self.vosk = vosk
        self.m = vosk.Model(str(carpeta))

    def __call__(self, ruta):
        with wave.open(str(ruta), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError(f"{ruta}: Vosk necesita WAV mono PCM 16 bits")
            rec = self.vosk.KaldiRecognizer(self.m, w.getframerate())
            while True:
                datos = w.readframes(4000)
                if not datos:
                    break
                rec.AcceptWaveform(datos)
        return json.loads(rec.FinalResult()).get("text", "").strip()


def crear_motor(nombre, beam):
    if nombre == "vosk":
        return Vosk()
    if nombre.startswith("whisper-"):
        return Whisper(nombre.split("-", 1)[1], beam)
    raise ValueError(nombre)


def duracion(ruta):
    with wave.open(str(ruta), "rb") as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="audios")
    ap.add_argument("--motores", nargs="+", default=MOTORES, choices=MOTORES)
    ap.add_argument("--beam", type=int, default=1, help="beam de Whisper (1 = el mismo que predecir.py)")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    carpeta = Path(args.entrada)
    with open(carpeta / "referencias.json", encoding="utf-8") as f:
        refs = json.load(f)
    audios = [(i, carpeta / f"{i}.wav") for i in sorted(refs) if (carpeta / f"{i}.wav").exists()]
    if not audios:
        raise SystemExit(f"No hay WAV en {carpeta} que coincidan con referencias.json")
    print(f"{len(audios)} audios con referencia en {carpeta}")

    resultados = []
    for nombre in args.motores:
        print(f"\n=== {nombre} ===", flush=True)
        t0 = time.perf_counter()
        try:
            motor = crear_motor(nombre, args.beam)
        except Exception as e:
            print(f"  [SALTADO] {e}")
            continue
        carga = time.perf_counter() - t0
        detalle, lat = [], []
        for i, ruta in audios:
            t = time.perf_counter()
            hip = motor(ruta)
            seg = time.perf_counter() - t
            lat.append(seg)
            w = jiwer.wer(normalizar(refs[i]), normalizar(hip) or "<vacio>")
            detalle.append({"id": i, "referencia": refs[i], "hipotesis": hip, "wer": round(w, 3),
                            "latencia_s": round(seg, 3), "duracion_audio_s": round(duracion(ruta), 2)})
            print(f"  {i} ({seg:.2f} s, WER {w:.2f}) {hip}", flush=True)
        wer_corpus = jiwer.wer([normalizar(refs[i]) for i, _ in audios],
                               [normalizar(d["hipotesis"]) or "<vacio>" for d in detalle])
        dur = sum(d["duracion_audio_s"] for d in detalle)
        resultados.append({"motor": nombre, "n": len(detalle), "wer_pct": round(100 * wer_corpus, 1),
                           "latencia_mediana_s": round(statistics.median(lat), 3),
                           "rtf": round(sum(lat) / dur, 3) if dur else None,
                           "carga_s": round(carga, 2), "detalle": detalle})
        del motor

    salida = args.salida or f"eval_voz_{carpeta.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump({"config": {"entrada": str(carpeta), "beam": args.beam, "n_audios": len(audios)},
                   "resultados": resultados}, f, ensure_ascii=False, indent=1)
    print(f"\n{'Motor':<15}{'WER %':>7}{'Lat med (s)':>13}{'RTF':>7}")
    for r in resultados:
        print(f"{r['motor']:<15}{r['wer_pct']:>7}{r['latencia_mediana_s']:>13}{r['rtf']:>7}")
    print("Rubrica WER: <10% excelente, 10-20% aceptable, >20% deficiente.")
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
