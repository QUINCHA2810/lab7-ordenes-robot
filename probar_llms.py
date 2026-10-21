"""
Laboratorio 7 - NLP y LLMs
Prueba de LLMs ligeros con Ollama: extraccion de acciones, objetos, personas y lugares.

Ejemplos de uso:
    py -3.12 probar_llms.py --version v1                       # reproduce la linea base
    py -3.12 probar_llms.py --version v2                       # prompt y ejemplos mejorados
    py -3.12 probar_llms.py --version v2 --postproceso         # v2 + limpieza en Python
    py -3.12 probar_llms.py --modelos qwen2.5:1.5b --repeticiones 3   # latencia mas estable
    py -3.12 probar_llms.py --sin-fewshot                      # sin ejemplos (ablacion)
    py -3.12 probar_llms.py --sin-esquema                      # si tu Ollama es viejo

Cada corrida guarda un archivo nuevo con fecha y hora (no se sobrescribe nada).
"""
import argparse
import copy
import json
import re
import statistics
import time
from datetime import datetime

import ollama

# Cliente con limite de espera: si una llamada se cuelga, falla en vez de quedarse esperando para siempre.
_cliente = ollama.Client(timeout=180)
# Tope de tokens generados por respuesta. Una orden de 3 acciones usa ~300 tokens; sin tope, un modelo
# que no cierra el JSON puede seguir generando hasta llenar el contexto (varios minutos en CPU).
MAX_TOKENS = 500

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
MODELOS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "gemma2:2b",
    "llama3.2:3b",
]

# Una o dos ordenes por nivel. Reemplazalas luego por las 150 ordenes de desarrollo.
# OJO: no uses estas ordenes (ni otras casi iguales) como ejemplos few-shot.
ORDENES = [
    ("1", "Trae el vaso"),
    ("1", "Saluda al visitante"),
    ("2", "Pon el libro azul sobre la mesa pequeña"),
    ("2", "Lleva el paquete grande a la oficina principal"),
    ("3", "Ve al cuarto, toma la cobija gris y tráesela a la abuela"),
    ("3", "Recibe a la señora Gómez, acompáñala a la sala de reuniones y avísale a Carlos"),
    ("4", "oye porfa me alcanzas el celular que dejé en el cuarto"),
    ("4", "eh mira dile al señor de la corbata roja que ya lo atienden"),
    ("5", "Tráeme eso"),
    ("5", "Qué demora la de hoy"),
]

# --- Version 1: linea base ---------------------------------------------------
PROMPT_V1 = """Eres un asistente que extrae informacion de ordenes en español dadas a un robot.
Devuelve UNICAMENTE un JSON valido con esta estructura:
{"acciones": [{"accion": "infinitivo",
               "objetos": [{"nombre": "...", "adjetivos": ["..."]}],
               "personas": [{"nombre": "...", "adjetivos": ["..."]}],
               "lugares": [{"nombre": "...", "adjetivos": ["..."]}]}]}

Reglas:
1. Hay UNA accion por cada verbo de la orden, en infinitivo y minuscula, fiel al verbo
   original ("pon" -> "poner", "lleva" -> "llevar", "saluda" -> "saludar", "dile" -> "decir").
   No cambies el verbo por otro ("pon" NO es "traer").
2. Los nombres van sin articulo ("la cobija" -> "cobija").
3. Los adjetivos van aparte del nombre y como aparecen ("paquete grande" -> nombre "paquete", adjetivos ["grande"]).
4. Los pronombres se reemplazan por la entidad a la que se refieren ("tráesela" -> la cobija y la abuela).
   Si no hay referente (por ejemplo "eso"), deja las listas vacias.
5. "tu" (el robot) y "yo" (quien habla) NUNCA son personas, salvo que la orden de un nombre propio.
6. Si no es una instruccion para el robot, devuelve {"acciones": []}.
7. NO inventes entidades ni lugares. Solo anota lo que aparece en la orden. Si falta informacion, deja listas vacias.
"""


def _e(nombre, *adjetivos):
    return {"nombre": nombre, "adjetivos": list(adjetivos)}


def _a(accion, objetos=(), personas=(), lugares=()):
    return {"accion": accion, "objetos": list(objetos),
            "personas": list(personas), "lugares": list(lugares)}


EJEMPLOS_V1 = [
    ("Busca las llaves en la sala",
     {"acciones": [_a("buscar", [_e("llaves")], [], [_e("sala")])]}),
    ("Pon la caja pequeña sobre el escritorio",
     {"acciones": [_a("poner", [_e("caja", "pequeña")], [], [_e("escritorio")])]}),
    ("Saluda a la señora Ruiz",
     {"acciones": [_a("saludar", [], [_e("señora Ruiz")], [])]}),
    ("Ve a la cocina, toma la botella verde y llévasela a Pedro",
     {"acciones": [
         _a("ir", [], [], [_e("cocina")]),
         _a("tomar", [_e("botella", "verde")], [], []),
         _a("llevar", [_e("botella", "verde")], [_e("Pedro")], []),
     ]}),
    ("ey porfa me pasas el control que está en la sala",
     {"acciones": [_a("pasar", [_e("control")], [], [_e("sala")])]}),
    ("Dame eso",
     {"acciones": [_a("dar")]}),
    ("Qué calor hace hoy",
     {"acciones": []}),
]

# --- Version 2: ataca los errores vistos en la linea base ---------------------
# (personas en "objetos", adjetivos pegados al nombre de un lugar, entidades repetidas)
PROMPT_V2 = PROMPT_V1.replace(
    '3. Los adjetivos van aparte del nombre y como aparecen ("paquete grande" -> nombre "paquete", adjetivos ["grande"]).',
    '3. Los adjetivos van aparte del nombre y como aparecen, tambien en los lugares\n'
    '   ("paquete grande" -> nombre "paquete", adjetivos ["grande"]; "bodega vieja" -> nombre "bodega", adjetivos ["vieja"]).',
) + """8. Las personas (nombres propios como Marta o Laura, y titulos como "señor Pérez") van SIEMPRE en "personas",
   nunca en "objetos". A quien se recibe, se acompaña, se llama o se le avisa es una persona, no un objeto.
9. Cada entidad va solo en la lista que le corresponde: no repitas la misma entidad en "objetos" y "personas".
"""

EJEMPLOS_V2 = EJEMPLOS_V1 + [
    ("Llama a Marta",
     {"acciones": [_a("llamar", [], [_e("Marta")], [])]}),
    ("Acompaña al señor Pérez hasta la puerta",
     {"acciones": [_a("acompañar", [], [_e("señor Pérez")], [_e("puerta")])]}),
    ("Lleva la caja mediana a la bodega vieja",
     {"acciones": [_a("llevar", [_e("caja", "mediana")], [], [_e("bodega", "vieja")])]}),
    ("Ponlo ahí",
     {"acciones": [_a("poner")]}),
    ("El café está muy frío",
     {"acciones": []}),
]

VERSIONES = {"v1": (PROMPT_V1, EJEMPLOS_V1), "v2": (PROMPT_V2, EJEMPLOS_V2)}

# ----------------------------------------------------------------------------
# Esquema JSON (salida restringida)
# ----------------------------------------------------------------------------
_ENTIDAD = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "adjetivos": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["nombre", "adjetivos"],
}
_LISTA_ENTIDADES = {"type": "array", "items": _ENTIDAD}

ESQUEMA = {
    "type": "object",
    "properties": {
        "acciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "accion": {"type": "string"},
                    "objetos": _LISTA_ENTIDADES,
                    "personas": _LISTA_ENTIDADES,
                    "lugares": _LISTA_ENTIDADES,
                },
                "required": ["accion", "objetos", "personas", "lugares"],
            },
        }
    },
    "required": ["acciones"],
}

# ----------------------------------------------------------------------------
# Validacion, revision de reglas y post-procesamiento
# ----------------------------------------------------------------------------
_INFINITIVO = re.compile(r"(ar|er|ir|ír|ér)$")
_ARTICULO = re.compile(r"^(el|la|los|las|un|una|unos|unas)\s", re.IGNORECASE)
_YO_TU = {"tu", "tú", "yo", "me", "te", "usted"}
# Pronombres/demostrativos que no son entidades (regla 4: sin referente no se anota nada)
_PRONOMBRES = {"eso", "esto", "aquello", "ello", "ese", "esa", "esos", "esas",
               "este", "esta", "estos", "estas", "aquel", "aquella",
               "lo", "la", "los", "las", "le", "les", "se"}
# Titulos y roles que indican una persona (lexico simple, ampliable)
_TITULOS_ROLES = {"señor", "señora", "señorita", "don", "doña", "doctor", "doctora",
                  "profesor", "profesora", "ingeniero", "ingeniera", "jefe", "jefa",
                  "abuelo", "abuela", "mamá", "papá", "madre", "padre", "hermano",
                  "hermana", "tío", "tía", "niño", "niña", "visitante", "cliente"}


def es_json_valido(texto):
    """True si el texto es JSON y cumple la estructura esperada."""
    try:
        data = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("acciones"), list):
        return False
    for acc in data["acciones"]:
        if not isinstance(acc, dict) or not isinstance(acc.get("accion"), str):
            return False
        for clave in ("objetos", "personas", "lugares"):
            if not isinstance(acc.get(clave), list):
                return False
            for ent in acc[clave]:
                if not isinstance(ent, dict) or "nombre" not in ent:
                    return False
                if not isinstance(ent.get("adjetivos", []), list):
                    return False
    return True


def revisar_reglas(salida):
    """Revisa automaticamente algunas reglas de anotacion (sin necesitar el JSON correcto).
    Devuelve una lista de avisos. No reemplaza el F1 con las anotaciones oficiales."""
    avisos = []
    try:
        data = json.loads(salida)
        acciones = data.get("acciones", [])
    except Exception:
        return ["JSON no interpretable"]
    for acc in acciones:
        verbo = str(acc.get("accion", "")).strip()
        if not _INFINITIVO.search(verbo.lower()):
            avisos.append(f"accion no infinitivo: '{verbo}'")
        for clave in ("objetos", "personas", "lugares"):
            for ent in acc.get(clave, []):
                nombre = str(ent.get("nombre", "")).strip()
                if not nombre:
                    avisos.append(f"entidad vacia en '{clave}'")
                    continue
                if nombre.lower() in _PRONOMBRES:
                    avisos.append(f"pronombre '{nombre}' anotado como entidad")
                if clave == "personas" and nombre.lower() in _YO_TU:
                    avisos.append(f"'{nombre}' anotado como persona")
                if _ARTICULO.match(nombre):
                    avisos.append(f"articulo en '{nombre}'")
    return avisos


def _parece_persona(nombre):
    """Heuristica simple: titulo/rol (señora, abuela...) o nombre propio con mayuscula."""
    tokens = nombre.split()
    if not tokens:
        return False
    if tokens[0].lower() in _TITULOS_ROLES:
        return True
    return any(t[0].isupper() for t in tokens)


def limpiar_salida(salida):
    """Post-procesamiento determinista (casi sin costo de latencia):
    - quita articulos iniciales y entidades vacias
    - elimina pronombres/demostrativos ("eso") y "tu/yo" como personas
    - elimina entidades repetidas dentro de una misma lista
    - mueve de 'objetos' a 'personas' lo que parece persona (señora Gómez, Carlos)
    - quita de 'objetos' lo que ya esta en 'personas'
    - colapsa acciones identicas consecutivas (el modelo a veces repite una accion en bucle)
    Si la salida no es un JSON valido, la devuelve sin cambios."""
    if not es_json_valido(salida):
        return salida
    data = json.loads(salida)
    nuevas = []
    for acc in data["acciones"]:
        listas = {}
        for clave in ("objetos", "personas", "lugares"):
            vistos, lista = set(), []
            for ent in acc[clave]:
                nombre = _ARTICULO.sub("", str(ent.get("nombre", "")).strip()).strip()
                if not nombre or nombre.lower() in _PRONOMBRES:
                    continue
                if clave == "personas" and nombre.lower() in _YO_TU:
                    continue
                if nombre.lower() in vistos:
                    continue
                vistos.add(nombre.lower())
                adjetivos = [str(a).strip() for a in ent.get("adjetivos", []) if str(a).strip()]
                lista.append({"nombre": nombre, "adjetivos": adjetivos})
            listas[clave] = lista

        en_personas = {e["nombre"].lower() for e in listas["personas"]}
        objetos = []
        for e in listas["objetos"]:
            if e["nombre"].lower() in en_personas:
                continue  # duplicada: ya esta en personas
            if _parece_persona(e["nombre"]):
                listas["personas"].append(e)
                en_personas.add(e["nombre"].lower())
                continue
            objetos.append(e)
        listas["objetos"] = objetos
        nueva = {"accion": acc["accion"].strip().lower(), **listas}
        if nuevas and nuevas[-1] == nueva:
            continue  # accion repetida en bucle
        nuevas.append(nueva)
    return json.dumps({"acciones": nuevas}, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Llamadas al modelo y medicion
# ----------------------------------------------------------------------------
def construir_mensajes(orden, prompt, ejemplos):
    mensajes = [{"role": "system", "content": prompt}]
    for ej_orden, ej_salida in ejemplos:
        mensajes.append({"role": "user", "content": f"Orden: {ej_orden}"})
        mensajes.append({"role": "assistant",
                         "content": json.dumps(ej_salida, ensure_ascii=False)})
    mensajes.append({"role": "user", "content": f"Orden: {orden}"})
    return mensajes


def extraer(modelo, orden, formato, prompt, ejemplos, solo_cpu=True):
    opciones = {"temperature": 0, "num_predict": MAX_TOKENS}
    if solo_cpu:
        opciones["num_gpu"] = 0  # 0 capas en GPU = todo en CPU (el taller exige solo CPU)
    t0 = time.perf_counter()
    r = _cliente.chat(
        model=modelo,
        messages=construir_mensajes(orden, prompt, ejemplos),
        format=formato,
        options=opciones,
    )
    return r["message"]["content"], time.perf_counter() - t0


MAX_ACCIONES = 5  # ninguna orden del taller necesita mas


def elegir_formato(modelo, formato, prompt, ejemplos, solo_cpu=True, max_acciones=MAX_ACCIONES):
    """Esquema JSON con tope de acciones (maxItems), y de paso calienta el modelo.
    Un modelo pequeño puede repetir la misma accion sin parar (lo vimos en O018); con el tope, el
    arreglo se cierra solo. Si esta version de Ollama rechaza el tope, se usa el esquema sin tope."""
    if not isinstance(formato, dict):
        return formato
    con_tope = copy.deepcopy(formato)
    con_tope["properties"]["acciones"]["maxItems"] = max_acciones
    try:
        extraer(modelo, "Trae el vaso", con_tope, prompt, ejemplos, solo_cpu)
        return con_tope
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:  # el modelo no existe: que lo informe quien llama
            raise
        print(f"  [Aviso] Ollama rechazo el tope de acciones ({e}); sigo sin tope.")
        extraer(modelo, "Trae el vaso", formato, prompt, ejemplos, solo_cpu)
        return formato


def memoria_modelo(modelo):
    """(tamaño total en memoria GB, parte en GPU GB) segun ollama.ps(). None si no se pudo."""
    try:
        for m in _cliente.ps().models:
            nombre = getattr(m, "model", None) or getattr(m, "name", "") or ""
            if nombre == modelo or nombre.startswith(modelo):
                total = (getattr(m, "size", 0) or 0) / 1e9
                vram = (getattr(m, "size_vram", 0) or 0) / 1e9
                return total, vram
    except Exception:
        pass
    return None, None


def descargar_modelo(modelo):
    """Libera la memoria del modelo antes de probar el siguiente."""
    try:
        _cliente.chat(model=modelo, messages=[], keep_alive=0)
    except Exception:
        pass


def probar_modelo(modelo, formato, prompt, ejemplos, solo_cpu, postproceso, repeticiones):
    print(f"\n{'=' * 70}\nModelo: {modelo}\n{'=' * 70}")
    try:
        extraer(modelo, "Trae el vaso", formato, prompt, ejemplos, solo_cpu)  # calentamiento
    except ollama.ResponseError as e:
        print(f"  [SALTADO] {e}. Descargalo con: ollama pull {modelo}")
        return None
    except Exception as e:  # servidor apagado, etc.
        print(f"  [ERROR] {e}")
        return None

    filas, latencias, validos, sin_avisos = [], [], 0, 0
    for nivel, orden in ORDENES:
        tiempos, bruta = [], None
        for _ in range(repeticiones):
            s, seg = extraer(modelo, orden, formato, prompt, ejemplos, solo_cpu)
            tiempos.append(seg)
            if bruta is None:
                bruta = s
        seg = statistics.median(tiempos)
        salida = limpiar_salida(bruta) if postproceso else bruta

        ok = es_json_valido(salida)
        avisos = revisar_reglas(salida) if ok else ["JSON invalido"]
        validos += ok
        sin_avisos += (not avisos)
        latencias.append(seg)

        print(f"\n  [N{nivel}] {orden}")
        if postproceso and salida != bruta:
            print(f"  bruta -> {bruta.strip()}")
            print(f"  limpia-> {salida.strip()}")
        else:
            print(f"  -> {salida.strip()}")
        print(f"  ({seg:.2f} s, JSON valido: {ok})")
        for a in avisos:
            print(f"  ! {a}")
        filas.append({"nivel": nivel, "orden": orden, "salida_bruta": bruta, "salida": salida,
                      "latencia_s": round(seg, 3), "json_valido": ok, "avisos": avisos})

    total_gb, vram_gb = memoria_modelo(modelo)
    if vram_gb:
        print(f"\n  AVISO: {vram_gb:.2f} GB del modelo estan en GPU. Esta medicion NO es solo CPU.")

    n = len(ORDENES)
    resumen = {
        "modelo": modelo,
        "latencia_mediana_s": round(statistics.median(latencias), 3),
        "json_valido_pct": round(100 * validos / n, 1),
        "sin_avisos_pct": round(100 * sin_avisos / n, 1),
        "ram_modelo_gb": round(total_gb, 2) if total_gb is not None else None,
        "detalle": filas,
    }
    descargar_modelo(modelo)
    return resumen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=MODELOS)
    ap.add_argument("--version", choices=sorted(VERSIONES), default="v2",
                    help="v1 = linea base; v2 = prompt y ejemplos mejorados")
    ap.add_argument("--sin-fewshot", action="store_true", help="no incluir ejemplos en el prompt")
    ap.add_argument("--postproceso", action="store_true", help="limpiar la salida con reglas en Python")
    ap.add_argument("--repeticiones", type=int, default=1,
                    help="repetir cada orden N veces y usar la mediana de latencia")
    ap.add_argument("--sin-esquema", action="store_true",
                    help='usar format="json" en vez del esquema (Ollama antiguo)')
    ap.add_argument("--permitir-gpu", action="store_true", help="permitir GPU (no es valido para el taller)")
    ap.add_argument("--salida", default=None, help="nombre del archivo; por defecto lleva fecha y hora")
    args = ap.parse_args()

    prompt, ejemplos = VERSIONES[args.version]
    if args.sin_fewshot:
        ejemplos = []
    formato = "json" if args.sin_esquema else ESQUEMA
    solo_cpu = not args.permitir_gpu

    etiqueta = args.version + ("_sinfs" if args.sin_fewshot else "") + ("_post" if args.postproceso else "")
    salida_archivo = args.salida or f"resultados_{etiqueta}_{datetime.now():%Y%m%d_%H%M%S}.json"

    resultados = [r for r in (probar_modelo(m, formato, prompt, ejemplos, solo_cpu,
                                            args.postproceso, max(1, args.repeticiones))
                              for m in args.modelos) if r]

    config = {"version": args.version, "ejemplos": len(ejemplos), "postproceso": args.postproceso,
              "repeticiones": args.repeticiones, "solo_cpu": solo_cpu, "esquema": not args.sin_esquema}
    with open(salida_archivo, "w", encoding="utf-8") as f:
        json.dump({"config": config, "resultados": resultados}, f, ensure_ascii=False, indent=2)

    print(f"\n\nConfiguracion: {config}")
    print(f"{'Modelo':<16}{'Latencia':>11}{'JSON ok':>10}{'Reglas ok':>11}{'RAM modelo':>12}")
    for r in resultados:
        ram_m = f"{r['ram_modelo_gb']} GB" if r["ram_modelo_gb"] is not None else "n/d"
        print(f"{r['modelo']:<16}{r['latencia_mediana_s']:>9} s{r['json_valido_pct']:>9}%"
              f"{r['sin_avisos_pct']:>10}%{ram_m:>12}")
    print(f"\nResultados completos en {salida_archivo}")


if __name__ == "__main__":
    main()
