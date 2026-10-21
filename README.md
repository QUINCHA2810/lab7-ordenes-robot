# Laboratorio 7 — Comprensión de órdenes habladas en español con LLMs ligeros

Sistema para un robot de servicio (hogar y recepción) que convierte una orden hablada en un JSON con
acciones, objetos, personas, lugares y adjetivos, y responde con una frase (y opcionalmente con voz).

```
audio ──faster-whisper──▶ texto ──LLM ligero (Ollama, CPU)──▶ JSON ──plantillas──▶ respuesta ──Piper──▶ voz
```

- **Voz a texto:** faster-whisper `small` (int8, CPU, `language="es"`, beam 5). Comparado con tiny/base y Vosk en `evaluar_voz.py`.
- **Comprensión:** `qwen2.5:3b` en Ollama, solo CPU (`num_gpu: 0`), temperatura 0, con salida
  restringida por esquema JSON (tope de 5 acciones), prompt few-shot (`v2`) y post-proceso determinista.
- **Respuesta:** plantillas en Python; si a una acción le falta información, pregunta
  (p. ej. "Tráeme eso" → "¿Qué quieres que te traiga?").
- **Voz de salida (opcional):** Piper con la voz `es_ES-davefx-medium`.

La configuración por defecto (LLM y tamaño de Whisper) está en `predecir.py`
(`LLM_POR_DEFECTO`, `WHISPER_POR_DEFECTO`) y se justifica en el informe.

**Resultados medidos (30 órdenes grabadas, solo CPU):**

| Sistema (Whisper small + Piper) | WER | F1 audio | F1 texto | Robustez audio | JSON | Latencia mediana | RAM pico |
|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | 12.7 % | 0.635 | 0.702 | 0.400 | 100 % | 4.96 s | 2.04 GB |
| **qwen2.5:3b (final)** | 12.7 % | **0.788** | 0.812 | 0.671 | 100 % | 7.52 s | 2.91 GB |
| llama3.2:3b | 12.7 % | 0.825 | 0.873 | 0.677 | 100 % | 8.97 s | 3.31 GB |

Fuentes: `e2e_*.json`; se regeneran con `py -3.12 tablas.py`.

## Requisitos

- Windows, Linux o macOS con CPU x86-64 y ≥ 8 GB de RAM. No usa GPU ni internet durante la inferencia.
- Python 3.12 (en Windows: `py -3.12`; en otros sistemas: `python3.12`).
- [Ollama](https://ollama.com) ≥ 0.5 (necesario para la salida con esquema JSON).
- Micrófono, solo para `grabar.py` y `demo.py`.

## Instalación (una sola vez, con internet)

```powershell
# 1. Ollama: instalarlo y dejarlo abierto (icono en la bandeja).
#    Si aparece el error "untrusted mount point", definir otra carpeta de modelos y reiniciar Ollama:
setx OLLAMA_MODELS C:\ollama_models

# 2. Modelos de lenguaje (el primero es el del sistema final; los otros son para la comparación)
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull gemma2:2b
ollama pull llama3.2:3b

# 3. Dependencias de Python
py -3.12 -m pip install -r requirements.txt

# 4. Modelos de voz en la carpeta modelos/ (así el sistema funciona sin internet)
py -3.12 -c "from faster_whisper.utils import download_model as d; [d(t, output_dir=f'modelos/whisper-{t}') for t in ('tiny','base','small')]"
curl.exe -L -o modelos/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
py -3.12 -c "import zipfile; zipfile.ZipFile('modelos/vosk.zip').extractall('modelos')"
curl.exe -L -o modelos/es_ES-davefx-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx
curl.exe -L -o modelos/es_ES-davefx-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json
```

Vosk solo hace falta para la comparación de voz a texto; Piper solo para `--hablar` y la demo.

> **Nota (PyAV):** faster-whisper 1.2.1 falla al abrir archivos con PyAV ≥ 15
> (`open() got an unexpected keyword argument 'metadata_errors'`). Por eso `predecir.py` lee los WAV/FLAC/OGG
> con `soundfile` y le pasa a Whisper las muestras directamente. Para mp3/m4a sí se usa PyAV; si fallan,
> conviértelos a WAV o instala `av<15`.

## Uso

**Comando único del entregable** (todos los audios de una carpeta → un JSON):

```powershell
py -3.12 predecir.py --entrada audios/ --salida predicciones.json
```

Opciones útiles:

```powershell
py -3.12 predecir.py --texto "Tráeme eso"                       # probar sin audio
py -3.12 predecir.py --entrada audios/ --hablar                 # además guarda la respuesta hablada en respuestas_voz/
py -3.12 predecir.py --entrada audios/ --anotaciones audios/anotaciones.json   # F1 extremo a extremo + WER
py -3.12 predecir.py --entrada audios/ --llm qwen2.5:1.5b   # más rápido (~5 s por orden) pero menos preciso
```

**Demo en vivo** (micrófono → respuesta en texto y voz):

```powershell
py -3.12 demo.py              # Enter para hablar, Enter al terminar; "t" para escribir; "q" para salir
py -3.12 demo.py --sin-voz    # si Piper falla
```

No hagas clic dentro de la consola mientras corre (el modo QuickEdit de Windows la congela).

## Formato de salida

`predicciones.json` es una lista con un objeto por audio:

```json
{
  "archivo": "O033.wav",
  "transcripcion": "Lleva la taza roja a la cocina y entrégasela a María",
  "acciones": [
    {"accion": "llevar", "objetos": [{"nombre": "taza", "adjetivos": ["roja"]}],
     "personas": [], "lugares": [{"nombre": "cocina", "adjetivos": []}]},
    {"accion": "entregar", "objetos": [{"nombre": "taza", "adjetivos": ["roja"]}],
     "personas": [{"nombre": "María", "adjetivos": []}], "lugares": []}
  ],
  "respuesta": "Voy a llevar la taza roja a la cocina y entregar la taza roja a María",
  "metricas": {"latencia_s": 2.1, "s2t_s": 0.6, "llm_s": 1.5, "tts_s": 0.0, "json_valido": true, ...}
}
```

`transcripcion`, `acciones` y `respuesta` son los campos que pide la guía; `metricas` es información extra.
Si el JSON del LLM no es válido, `acciones` queda vacío y la respuesta pide repetir la orden.

## Archivos

| Archivo | Para qué |
|---|---|
| `predecir.py` | Sistema completo (comando único) y evaluación extremo a extremo (`--anotaciones`). |
| `probar_llms.py` | Núcleo del LLM: prompts `v1`/`v2`, ejemplos few-shot, esquema JSON, post-proceso. |
| `evaluar.py` | F1 por categoría, estructura, robustez, JSON válido, latencia y RAM sobre `dataset_lab7.json`. |
| `evaluar_voz.py` | WER y latencia de Whisper tiny/base/small y Vosk. |
| `grabar.py` | Graba las 30 órdenes de la partición `prueba` en `audios/` (16 kHz mono). |
| `demo.py` | Demo en vivo con micrófono. |
| `tablas.py` | Imprime las tablas del informe a partir de los archivos de resultados. |
| `generar_informe.py`, `textos_informe.py` | Generan `informe.docx` a partir de `tablas.json` (`py -3.12 tablas.py --json tablas.json`). |
| `dataset_lab7.json` | 60 órdenes anotadas por el equipo (30 `ajuste`, 30 `prueba`). |
| `audios/` | Las 30 órdenes grabadas, `referencias.json` (texto) y `anotaciones.json` (JSON esperado). |
| `eval_*.json`, `e2e_*.json`, `eval_voz_*.json`, `resultados_*.json` | Resultados medidos (no se sobrescriben). |

## Cómo reproducir las tablas del informe

```powershell
# LLMs sobre texto correcto, partición de prueba (4 modelos, prompt v2 + post-proceso)
py -3.12 evaluar.py --particion prueba --version v2 --postproceso
# Ablación: prompt v1 sin post-proceso
py -3.12 evaluar.py --particion prueba --version v1 --modelos qwen2.5:1.5b
# Voz a texto (necesita audios/ grabados con grabar.py)
py -3.12 evaluar_voz.py --entrada audios/
# Extremo a extremo con la configuración final
py -3.12 predecir.py --entrada audios/ --anotaciones audios/anotaciones.json --salida pred_e2e.json
# Tablas a partir de los archivos guardados
py -3.12 tablas.py
```

Las métricas son la interpretación del equipo de la sección 6.2 de la guía (no tenemos el script oficial).
Si llegan las 150 órdenes oficiales, basta con convertirlas al formato de `dataset_lab7.json` y correr `evaluar.py --dataset <archivo> --particion todo`.
