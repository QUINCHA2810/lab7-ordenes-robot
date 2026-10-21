"""Contenido del informe. Las cifras se leen de tablas.json (TB); el texto cualitativo resume los
archivos de resultados citados en cada seccion."""


def _llm(TB, modelo):
    return next(f for f in TB["llms"] if f["modelo"] == modelo)


def _voz(TB, motor):
    return next((v for v in TB.get("voz", []) if v["motor"] == motor), None)


def _e2e(TB, llm):
    filas = [e for e in TB.get("e2e", []) if e["config"].startswith(llm + " ")]
    return filas[-1] if filas else None


def SECCIONES(TB, fmt):
    q15, q3, gm, ll = (_llm(TB, m) for m in ("qwen2.5:1.5b", "qwen2.5:3b", "gemma2:2b", "llama3.2:3b"))
    abl = {f["modelo"]: f for f in TB["ablacion"]}
    v1, sinp, conp = abl["qwen2.5:1.5b v1"], abl["qwen2.5:1.5b v2 sin post"], abl["qwen2.5:1.5b v2 + post"]
    vt, vb, vs, vk = (_voz(TB, m) for m in ("whisper-tiny", "whisper-base", "whisper-small", "vosk"))
    e15, e3, ell = (_e2e(TB, m) for m in ("qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b"))
    vs5 = next((v for v in TB.get("voz_beam5", []) if v["motor"] == "whisper-small"), None)
    r15, r3, rll = (e["rubrica"]["Total (de 80)"] for e in (e15, e3, ell))
    F = {}
    F["lectura"] = (
        f"Con texto correcto, el F1 crece con el tamaño: qwen2.5:1.5b {fmt(q15['f1'])}, qwen2.5:3b {fmt(q3['f1'])}, "
        f"gemma2:2b {fmt(gm['f1'])} y llama3.2:3b {fmt(ll['f1'])}, el único sobre el umbral de 0.85. La latencia crece "
        f"igual ({fmt(q15['latencia'], 2)} s a {fmt(ll['latencia'], 2)} s), salvo gemma2:2b, el más lento "
        f"({fmt(gm['latencia'], 2)} s) sin ser el más preciso. Todos tienen 100 % de JSON válido. La latencia depende "
        f"sobre todo de la longitud de la salida: en las órdenes compuestas (nivel 3), qwen2.5:1.5b sube de 2.6 s "
        f"(nivel 1) a 9.0 s de mediana. En la ablación, el prompt v2 sube el F1 de {fmt(v1['f1'])} a {fmt(sinp['f1'])} "
        f"y el post-proceso lo lleva a {fmt(conp['f1'])}, sin costo de latencia (mismas salidas del modelo). En voz a "
        f"texto, solo Whisper small baja del 20 % de WER ({fmt(vs['wer'], 1)} %; {fmt(vs5['wer'], 1)} % con beam 5); "
        f"Vosk ({fmt(vk['wer'], 1)} %) es más preciso que Whisper base ({fmt(vb['wer'], 1)} %) y tiny "
        f"({fmt(vt['wer'], 1)} %) con una latencia parecida a la de tiny. De extremo a extremo, la transcripción "
        f"real resta {fmt(e15['f1_texto'] - e15['f1_audio'])} de F1 a qwen2.5:1.5b, {fmt(e3['f1_texto'] - e3['f1_audio'])} "
        f"a qwen2.5:3b y {fmt(ell['f1_texto'] - ell['f1_audio'])} a llama3.2:3b. La mediana por etapa del sistema "
        f"final es S2T {fmt(e3['lat_s2t'], 2)} s + LLM {fmt(e3['lat_llm'], 2)} s + voz {fmt(e3['lat_tts'], 2)} s.")
    F["justificacion"] = (
        f"Elegimos qwen2.5:3b + Whisper small (beam 5) + Piper, con salida restringida por esquema, tope de 5 "
        f"acciones, prompt few-shot v2 y post-proceso. Aplicamos la rúbrica a las cifras extremo a extremo "
        f"(Tabla 5) en los criterios medibles: este sistema suma unos {r3:.0f} de 80 puntos, frente a {rll:.0f} "
        f"de llama3.2:3b y {r15:.0f} de qwen2.5:1.5b. Frente a qwen2.5:1.5b, que es el doble de rápido, gana en F1 con "
        f"audio ({fmt(e3['f1_audio'])} frente a {fmt(e15['f1_audio'])}: pasa de deficiente a aceptable) y en robustez "
        f"({fmt(e3['rob_audio'])} frente a {fmt(e15['rob_audio'])}). Eso compensa con creces pasar de "
        f"{fmt(e15['latencia'], 1)} s a {fmt(e3['latencia'], 1)} s. Frente a llama3.2:3b, que tiene el mejor F1 "
        f"({fmt(ell['f1_audio'])}, todavía en la banda aceptable), es más rápido ({fmt(e3['latencia'], 1)} s frente a "
        f"{fmt(ell['latencia'], 1)} s; la rúbrica marca deficiente a partir de 8 s) y cabe en 3 GB "
        f"({fmt(e3['ram_total'], 2)} frente a {fmt(ell['ram_total'], 2)} GB). Whisper small es la única opción con WER "
        f"aceptable; tiny, base y Vosk ahorran entre 1.3 y 1.5 s por orden, pero multiplican el WER por 1.9 a 3.1. Los riesgos son "
        f"que la latencia queda cerca del límite de 8 s (las órdenes de 3 acciones superan 15 s) y que la licencia "
        f"de qwen2.5:3b (Qwen Research License) permite uso académico, pero no comercial; qwen2.5:1.5b es Apache "
        f"2.0. Si el jurado prioriza la velocidad, basta con usar --llm qwen2.5:1.5b. Las técnicas se justifican "
        f"así: el esquema elimina el JSON inválido; el few-shot fija las convenciones de anotación (sin ejemplos, solo "
        f"el 30 % de las salidas cumplía las reglas en nuestro sondeo inicial); el post-proceso corrige errores "
        f"sistemáticos sin costo de latencia; y el tope de acciones corta los bucles de repetición.")
    F["errores_s2t"] = (
        "Whisper falla sobre todo en órdenes cortas o coloquiales: \"oye abre la ventana grande porfa\" → \"Uji y "
        "aurilo en Tana Grande, porfa\", \"Ábrelo\" → \"¡Ahorilo!\" y una alucinación típica de Whisper: "
        "\"Llévalas allá\" → \"¡Gracias a todos!\". En esas cuatro órdenes (O045, O047, O055, O057), el sistema "
        "final respondió \"No entendí ninguna orden\" en lugar de actuar. También cambia palabras sueltas "
        "(\"cobija\" → \"coija\", \"manta\" → \"amanta\", \"garaje\" → \"garage\") y el LLM las copia tal cual. "
        "Además, el LLM cambia su salida con diferencias mínimas de forma (un punto final o una mayúscula, como "
        "en O007, O009, O048 y O060), a veces para mejor. Por eso no todo el cambio entre texto y audio es error "
        "del S2T. Por último, la plantilla de respuesta hereda los errores: una acción vacía como \"mirar\" "
        "genera una pregunta innecesaria (O046), y \"trae\" sin lematizar produce \"Voy a trae…\" (O020).")
    F["propagacion"] = (
        f"Un error de transcripción cambia las entidades (\"coija\") o borra la orden completa; el LLM no puede "
        f"recuperarla porque solo ve el texto. Medimos la propagación corriendo el mismo LLM con la transcripción "
        f"de Whisper y con el texto correcto (Tabla 5): con qwen2.5:3b el F1 baja {fmt(e3['f1_texto'] - e3['f1_audio'])} "
        f"y la robustez {fmt(e3['rob_texto'] - e3['rob_audio']) if 'rob_texto' in e3 else 'n/d'}, sobre todo por las "
        f"órdenes coloquiales. El daño es mayor en el modelo pequeño (qwen2.5:1.5b pierde "
        f"{fmt(e15['f1_texto'] - e15['f1_audio'])}). Para mitigarlo: un Whisper más grande o con beam mayor (Radford "
        f"et al., 2022), un initial_prompt con el vocabulario del dominio, normalizar la transcripción antes del LLM "
        f"(minúsculas, sin puntuación), y aprovechar la confianza del S2T para pedir que se repita la orden en vez "
        f"de responder \"no entendí\".")
    S = []
    a = S.append

    # ------------------------------------------------------------------ 1
    a(("h1", "1. Introducción"))
    a(("p", "Un robot de servicio en el hogar o en una recepción debe convertir una orden hablada en una "
            "estructura que pueda ejecutar. Construimos un sistema que, en CPU, sin internet y con modelos de "
            "menos de 4 mil millones de parámetros, transforma el audio en un JSON con acciones (en infinitivo), "
            "objetos, personas, lugares y adjetivos, y responde con una confirmación o con una pregunta de "
            "aclaración cuando falta información. Comparamos cuatro LLMs ligeros y cuatro configuraciones de voz "
            "a texto (S2T), y medimos cuánto error introduce la transcripción en el resultado final."))

    # ------------------------------------------------------------------ 2
    a(("h1", "2. Sistema y metodología"))
    a(("pb", "Pipeline. ", "audio → faster-whisper (int8, CPU, language=\"es\") → LLM en Ollama (num_gpu = 0, "
            "temperatura 0, máximo 500 tokens) → JSON → respuesta por plantillas → Piper (voz es_ES-davefx-medium). "
            "Todo se ejecuta con un único comando: python predecir.py --entrada audios/ --salida predicciones.json."))
    a(("pb", "Técnicas de comprensión. ",
       "(1) Salida restringida con un esquema JSON (Ollama format=<esquema>), que obliga a las claves accion, "
       "objetos, personas y lugares y a entidades {nombre, adjetivos}, con un tope de 5 acciones (maxItems). "
       "(2) Prompt con las 6 reglas de anotación de la guía y 12 ejemplos few-shot como turnos de chat (versión "
       "v2; v1 tenía 7 ejemplos y menos reglas). (3) Post-proceso determinista en Python: quita artículos, "
       "pronombres sin referente y \"tú/yo\", elimina duplicados, mueve a personas lo que parece persona "
       "(títulos o mayúscula inicial) y colapsa acciones repetidas en bucle. (4) Respuesta por plantillas con "
       "concordancia de artículos; si una acción no tiene objeto, persona ni lugar, se pregunta "
       "(\"Tráeme eso\" → \"¿Qué quieres que te traiga?\")."))
    a(("pb", "Datos. ", "La guía prevé 150 órdenes oficiales y un script de evaluación, que no recibimos. "
            "Por eso creamos 60 órdenes anotadas (12 por nivel, mitad hogar y mitad recepción), divididas en "
            "una partición de ajuste (30), usada para diseñar prompts y post-proceso, y una de prueba (30), que "
            "solo se usó para medir. Las 30 órdenes de prueba se grabaron con micrófono (16 kHz, mono) y "
            "son las 30 órdenes grabadas y anotadas del entregable."))
    a(("pb", "Métricas (interpretación propia de la sección 6.2). ",
       "F1 micro por categoría sobre multiconjuntos de cadenas normalizadas (minúsculas, sin tildes ni "
       "artículo) y su promedio en las 5 categorías; estructura = F1 de tuplas (acción, categoría, nombre, "
       "adjetivos), es decir, entidad bien asignada a su acción y con sus adjetivos; robustez = F1 promedio "
       "en niveles 4 y 5; JSON válido = salida que cumple el esquema; latencia = mediana por orden del fin del "
       "audio al inicio de la respuesta (S2T + LLM + síntesis de voz); RAM = pico del proceso Python "
       "(Whisper y Piper) más la memoria del modelo que reporta ollama ps (psutil no ve la memoria del "
       "servidor de Ollama). WER con jiwer tras normalizar minúsculas, tildes y puntuación. Equipo de "
       "medición: Intel Core i7-11800H (8 núcleos, 16 hilos) y 8 GB de RAM (7.6 GB utilizables), Windows 11, Ollama 0.40.6."))

    # ------------------------------------------------------------------ 3
    a(("h1", "3. Resultados"))
    filas = []
    for f in TB["llms"]:
        filas.append([f["modelo"], f["params"], fmt(f["f1"]), fmt(f["estructura"]), fmt(f["robustez"]),
                      fmt(f["json"], pct=True), fmt(f["latencia"], 2), fmt(f["ram"], 2), fmt(f["tamano"], 2)])
    a(("tabla", ["Modelo (Ollama, Q4)", "Parám.", "F1 prom.", "Estructura", "Robustez N4–5", "JSON válido",
                 "Latencia (s)", "RAM modelo (GB)", "Disco (GB)"], filas,
       "Tabla 1. LLMs sobre el texto correcto, partición de prueba (30 órdenes), prompt v2 + post-proceso, "
       "solo CPU. Fuente: eval_prueba_v2_post_4modelos_final.json."))
    filas = [[f["modelo"]] + [fmt(f["f1_cat"][c]) for c in ("accion", "persona", "objeto", "adjetivo", "lugar")]
             + [fmt(TB["f1_por_nivel"][f["modelo"]][n], 2) for n in "12345"]
             for f in TB["llms"]]
    a(("tabla", ["Modelo", "Acción", "Persona", "Objeto", "Adjetivo", "Lugar", "N1", "N2", "N3", "N4", "N5"],
       filas, "Tabla 2. F1 por categoría y F1 promedio por nivel de dificultad (N1–N5, 6 órdenes cada uno; mismo archivo)."))
    filas = [[f["modelo"].replace("qwen2.5:1.5b ", ""), fmt(f["f1"]), fmt(f["estructura"]), fmt(f["robustez"]),
              fmt(f["json"], pct=True), fmt(f["exactas"], pct=True), fmt(f["latencia"], 2)]
             for f in (v1, sinp, conp)]
    a(("tabla", ["qwen2.5:1.5b", "F1 prom.", "Estructura", "Robustez", "JSON", "Órdenes exactas", "Latencia (s)"],
       filas, "Tabla 3. Ablación en la partición de prueba. \"Sin post\" se recalcula sobre las salidas brutas de "
              "la misma corrida (aísla el post-proceso). Fuentes: eval_prueba_v1_qwen15_final.json y la Tabla 1."))
    filas = [[v["motor"], fmt(v["wer"], 1) + " %", fmt(v["latencia"], 2), fmt(v["rtf"], 3)]
             for v in TB["voz"]]
    if TB.get("voz_beam5"):
        filas += [[v["motor"] + " (beam 5)", fmt(v["wer"], 1) + " %", fmt(v["latencia"], 2), fmt(v["rtf"], 3)]
                  for v in TB["voz_beam5"]]
    a(("tabla", ["Motor S2T (CPU)", "WER", "Latencia mediana (s)", "Factor tiempo real"], filas,
       "Tabla 4. Voz a texto sobre las 30 grabaciones (beam 1 salvo indicación). Fuentes: eval_voz_audios_*.json."))
    filas = [[e["config"], fmt(e["wer"], 1) + " %", fmt(e["f1_texto"]), fmt(e["f1_audio"]), fmt(e["estr_audio"]),
              fmt(e["rob_audio"]), fmt(e["json_audio"], pct=True), fmt(e["latencia"], 2), fmt(e["ram_total"], 2)]
             for e in TB.get("e2e", [])]
    a(("tabla", ["Sistema (Whisper small, beam 5, Piper)", "WER", "F1 texto", "F1 audio", "Estructura audio",
                 "Robustez audio", "JSON", "Latencia (s)", "RAM pico (GB)"], filas,
       "Tabla 5. Extremo a extremo sobre las 30 grabaciones. \"F1 texto\": mismo LLM con la transcripción "
       "correcta; \"F1 audio\": con la de Whisper. Latencia de fin de audio a inicio de la voz. Fuentes: e2e_*.json."))

    a(("pb", "Lectura de resultados. ", F["lectura"]))

    # ------------------------------------------------------------------ 4
    a(("h1", "4. Técnica elegida y justificación"))
    a(("p", F["justificacion"]))

    # ------------------------------------------------------------------ 5
    a(("h1", "5. Análisis de errores"))
    a(("p", "Revisamos orden por orden las diferencias de los cuatro modelos en la partición de prueba "
            "(evaluar.py --leer) y las transcripciones de cada motor S2T. Los errores más frecuentes fueron:"))
    for t, d in [
        ("Objeto anotado como lugar (los 4 modelos). ",
         "\"Enciende el televisor\" → lugar televisor; \"cierra la puerta\", \"apaga la luz\" y \"cierra la ventana\" "
         "igual. El verbo sin destino lleva al modelo a poner su complemento en lugares."),
        ("Adjetivo pegado al nombre en personas (los 4). ",
         "\"Busca al señor alto\" → persona \"señor alto\"; \"cliente nuevo\" igual. En la partición de ajuste ya "
         "aparecía con lugares (\"oficina principal\"); v2 lo redujo en lugares (no del todo: \"recepción "
         "principal\" en qwen2.5:1.5b), pero no en personas."),
        ("Marcadores del discurso tomados como acciones (los 4). ",
         "\"mira acompaña al cliente…\" → acción extra \"mirar\". Afecta sobre todo al nivel 4."),
        ("Verbo cambiado por sesgo de los ejemplos. ",
         "\"trae\" → \"llevar\" o \"tomar\" (llama3.2:3b en O020, O023 y O043; qwen y gemma en O035), por los "
         "ejemplos \"llévasela\" y \"toma la botella\". \"entrégasela\" → \"entrar\" (qwen2.5:1.5b) y verbos mal "
         "escritos (\"recebir\", \"entrregar\")."),
        ("Entidades inventadas. ",
         "\"Avísale que ya vamos\" → persona \"señora Ruiz\" (copiada de un ejemplo few-shot) o \"alguien\"; "
         "\"Ábrelo\" → objeto \"café frío\" (llama3.2:3b); \"me traes una cobija\" → persona \"abuela\". El "
         "post-proceso no puede detectarlas porque no compara contra la transcripción."),
        ("Comentarios convertidos en órdenes (sobre todo qwen2.5:1.5b). ",
         "\"El ascensor está dañado otra vez\" → \"reparar\"; \"Mañana llueve\" → \"predecir\". Los modelos de 3B "
         "acertaron más trampas (5 de 6 órdenes de nivel 5 exactas, frente a 2 de 6 de qwen2.5:1.5b)."),
        ("Bucles de repetición. ",
         "qwen2.5:1.5b llenó el tope de 5 acciones en 3 de 30 órdenes (p. ej. O048: guardar/ir repetidos) y "
         "tardó hasta 18 s. Con el tope, el JSON siempre cierra: 100 % de JSON válido en los cuatro modelos. "
         "En la partición de ajuste, antes de añadir el tope, qwen2.5:1.5b tenía 93.3 % (O018 se repetía hasta "
         "agotar los tokens)."),
        ("Errores de transcripción. ",
         F["errores_s2t"]),
    ]:
        a(("pb", t, d))

    # ------------------------------------------------------------------ 6
    a(("h1", "6. Preguntas guía"))
    preguntas = [
        ("1. ¿Por qué equilibrio y no el modelo más grande? ",
         "Porque el robot funciona con recursos limitados y en interacción con personas: una respuesta que llega "
         "tarde no sirve aunque sea correcta. En nuestra Tabla 1, pasar de 1.5B a 3B mejora el F1, pero casi "
         "duplica la latencia y la RAM. La arquitectura Transformer (Vaswani et al., 2017) tiene un costo por "
         "token que crece con el tamaño del modelo, y en CPU ese costo domina; por eso los modelos pequeños "
         "están pensados para ejecutarse en el dispositivo (Abdin et al., 2024; Gemma Team, 2024b)."),
        ("2. ¿Ventajas de la salida restringida? ",
         "El decodificador solo puede emitir tokens que mantienen el JSON dentro del esquema (Willard & Louf, "
         "2023), así que no hace falta parsear texto libre y el robot recibe siempre las mismas claves. Con "
         "esquema y tope de acciones obtuvimos 100 % de JSON válido en los 4 modelos. No garantiza que el "
         "contenido sea correcto: un JSON válido puede tener la entidad en la lista equivocada."),
        ("3. ¿Cómo afecta la cuantización? ",
         "Todos los modelos se usaron cuantizados a 4 bits (Q4_K_M; gemma2:2b en Q4_0), que es lo que "
         "distribuye Ollama: qwen2.5:1.5b ocupa 0.99 GB en disco y 1.17 GB en memoria, frente a unos 3 GB que "
         "ocuparía en 16 bits. No medimos variantes Q8 o FP16, así que no podemos cuantificar la pérdida de F1 "
         "en esta tarea; la literatura reporta pérdidas pequeñas a 4 bits (Dettmers et al., 2023). En CPU, "
         "menos bits implican menos memoria que leer por token y, por lo tanto, menos latencia."),
        ("4. ¿Cómo resolver pronombres como \"entrégasela\" sin internet? ",
         "Con el propio LLM y ejemplos few-shot que enseñan a copiar el referente de la acción anterior (lo que "
         "usamos), y con reglas locales: separar los clíticos (-se-la, -lo) con spaCy (Honnibal & Montani, "
         "2017) y enlazarlos al último objeto o persona compatible en género y número. Ninguna de las dos "
         "necesita conexión."),
        ("5. ¿Cómo detectar y penalizar la invención de entidades? ",
         "Las entidades inventadas cuentan como falsos positivos en el F1 y bajan la precisión. Para "
         "detectarlas en ejecución, se puede comprobar que cada nombre (o su lema) aparece en la transcripción "
         "y descartar o preguntar si no aparece. Así se eliminarían \"señora Ruiz\" o \"café frío\" del "
         "análisis de errores. No lo implementamos a tiempo."),
        ("6. ¿Qué papel juega spaCy? ",
         "Lematiza verbos (\"tráesela\" → \"traer\") y da categorías gramaticales para comprobar que la acción "
         "es un verbo y que los adjetivos lo son (Honnibal & Montani, 2017). Corregiría errores que vimos, como "
         "\"recebir\", \"entrar\" en lugar de \"entregar\" y \"mirar\" como acción. Lo dejamos como mejora: el "
         "sistema actual no lo usa."),
        ("7. ¿Cómo se propaga el error del S2T y cómo mitigarlo? ",
         F["propagacion"]),
        ("8. ¿Diferencias entre niveles 1–3 y 4–5? ",
         "Esperábamos menor F1 en 4–5, y así fue: la robustez es menor que el F1 global en los 4 modelos "
         "(Tabla 1). En el nivel 5 hay pocas entidades, así que una sola inventada (\"señora Ruiz\", \"allá\") "
         "hunde el F1, aunque la acción sea correcta. Sorprendió el nivel 3: las órdenes compuestas tienen el "
         "mejor F1 en los modelos de 3B (Tabla 2), quizá porque varios ejemplos few-shot tienen esa misma forma, "
         "pero son las más lentas, porque la salida es más larga. El nivel 2 también salió bajo por los "
         "adjetivos pegados al nombre. En voz, Whisper falla más en las órdenes cortas y coloquiales "
         "(niveles 4–5), lo que suma error justo donde el LLM ya es más débil."),
        ("9. ¿Qué técnicas de ajuste fino (LoRA) podrían ayudar? ",
         "LoRA entrena matrices de bajo rango sobre el modelo congelado (Hu et al., 2022), y QLoRA lo hace "
         "sobre el modelo en 4 bits (Dettmers et al., 2023), lo que cabe en una GPU modesta o en Colab. Con "
         "unos cientos de órdenes anotadas, un qwen2.5:1.5b ajustado podría prescindir de los 12 ejemplos "
         "few-shot (prompt más corto, menos latencia) y aprender las convenciones que hoy falla (adjetivos "
         "aparte, objeto frente a lugar). La inferencia seguiría en CPU con el adaptador fusionado y "
         "cuantizado. No lo probamos."),
        ("10. ¿Relación con la robótica de servicio real? ",
         "Es la interfaz de lenguaje de un robot de asistencia doméstica o de recepción: la salida estructurada "
         "se conecta a un planificador de tareas, la pregunta de aclaración evita acciones peligrosas o "
         "inútiles cuando falta información, y el funcionamiento local protege la privacidad del hogar y no "
         "depende de la red (Jurafsky & Martin, 2023)."),
    ]
    for t, d in preguntas:
        a(("pb", t, d))

    # ------------------------------------------------------------------ 7
    a(("h1", "7. Limitaciones"))
    a(("p", "(a) Las 60 órdenes las anotó el equipo con ayuda de IA; no son las 150 oficiales, y las métricas son "
            "nuestra interpretación de la guía porque no tuvimos el script oficial. (b) 30 órdenes de prueba "
            "dan intervalos amplios: una orden equivale a unos 3 puntos porcentuales de exactitud, y diferencias "
            "como v1 frente a v2 en ajuste (0.697 frente a 0.737) están dentro del ruido. (c) El prompt v2 y el "
            "post-proceso se diseñaron mirando errores de la partición de ajuste. En ajuste, qwen2.5:1.5b llegó "
            "a F1 0.770; en prueba bajó a " + fmt(q15["f1"]) + ", lo que confirma que las cifras de ajuste eran "
            "optimistas. (d) Varios ejemplos few-shot comparten estructura con órdenes de nivel 3. (e) El tamaño "
            "de Whisper y el beam se eligieron mirando los audios de prueba (ninguna otra decisión se tomó con "
            "esa partición). (f) Una sola voz, un solo micrófono y un solo equipo; cada configuración se midió "
            "una vez (sin repeticiones)."))

    # ------------------------------------------------------------------ refs
    a(("h1", "Referencias"))
    for r in [
        "Abdin, M., Jacobs, S. A., Awan, A. A., et al. (2024). Phi-3 technical report: A highly capable language model locally on your phone. arXiv:2404.14219.",
        "Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems, 36.",
        "Gemma Team, Google DeepMind. (2024a). Gemma: Open models based on Gemini research and technology. arXiv:2403.08295.",
        "Gemma Team, Google DeepMind. (2024b). Gemma 2: Improving open language models at a practical size. arXiv:2408.00118.",
        "Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.",
        "Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. Explosion AI.",
        "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations.",
        "Jurafsky, D., & Martin, J. H. (2023). Speech and language processing (3rd ed. draft). Stanford University.",
        "Qwen Team, Alibaba. (2024). Qwen2.5 technical report. arXiv:2412.15115.",
        "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. arXiv:2212.04356.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
        "Willard, B. T., & Louf, R. (2023). Efficient guided generation for large language models. arXiv:2307.09702.",
    ]:
        a(("ref", r))
    return S
