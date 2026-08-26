"""
Prompt IA — Prototipo con IA real (Groq)
-----------------------------------
El estudiante escribe su tarea en texto libre (como en un chat). La IA:
1. Clasifica automaticamente el tipo de tarea (ensayo, resumen, codigo, etc.)
2. Selecciona el modelo recomendado segun esa clasificacion (regla fija
   del equipo, no la inventa la IA)
3. Aplica el mecanismo de aprendizaje progresivo ya validado (a veces
   pregunta al estudiante en vez de responder directo)
4. Genera la respuesta real a la tarea con el modelo elegido

Mecanismo de aprendizaje progresivo (sin cambios, ya validado):
- Contador de aciertos consecutivos por tipo de tarea
- Probabilidad creciente de preguntarle al usuario en vez de responder directo
- Reseteo del contador al fallar
- Expiracion del contador por inactividad, con confirmacion al usuario
- Historial visible de aciertos/errores

Requiere una variable de entorno GROQ_API_KEY con tu API key.
Ponla en un archivo .env en la misma carpeta, con esta linea:
    GROQ_API_KEY=tu_key_aqui

Para correrlo:
    streamlit run prompt_ia_prototipo.py
"""

import hashlib
import json
import os
import random
import secrets
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

try:
    import psycopg2
except ImportError:
    psycopg2 = None

load_dotenv()


def _obtener_api_key():
    """
    Busca la API key primero en variables de entorno / .env (para correr
    local) y si no la encuentra, en st.secrets (para Streamlit Community
    Cloud, donde la key se configura en Settings > Secrets).
    """
    valor = os.getenv("GROQ_API_KEY")
    if valor:
        return valor
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


API_KEY = _obtener_api_key()
cliente_groq = Groq(api_key=API_KEY) if API_KEY else None

MODELO_GROQ = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------
# Modelos simulados: en este prototipo todos corren sobre el mismo modelo
# de Groq por debajo (no hay acceso real a 5 modelos distintos). Los
# nombres y descripciones son etiquetas pedagogicas de Klarblick, no
# modelos reales separados.
# ---------------------------------------------------------------------
MODELOS_INFO = {
    "Klarblick Redactor": "Fuerte en redaccion y razonamiento largo. Ideal para ensayos, cartas y textos argumentativos.",
    "Klarblick Sintesis": "Rapido y conciso. Ideal para resumir textos largos sin perder las ideas clave.",
    "Klarblick Codex": "Especializado en programacion. Escribe, explica y depura codigo en distintos lenguajes.",
    "Klarblick Numerica": "Fuerte en razonamiento numerico. Ideal para interpretar datos, tablas y estadisticas.",
    "Klarblick Visual": "Generador de imagenes a partir de descripciones de texto.",
}

# ---------------------------------------------------------------------
# Reglas fijas del equipo: que modelo conviene para cada tipo de tarea.
# Esta asignacion es una decision de producto, no la inventa la IA.
# ---------------------------------------------------------------------
TIPOS_TAREA = {
    "Ensayo": "Klarblick Redactor",
    "Resumen": "Klarblick Sintesis",
    "Codigo": "Klarblick Codex",
    "Analisis de datos": "Klarblick Numerica",
    "Generacion de imagenes": "Klarblick Visual",
}
OPCIONES_MODELO = list(TIPOS_TAREA.values())
DIAS_INACTIVIDAD_LIMITE = 14
LIMITE_DIARIO_LIBRE = 10

# ---------------------------------------------------------------------
# Base de datos (Supabase / Postgres): cuentas de usuario, historial
# persistente, y limite diario del plan gratis. Se conecta via la cadena
# guardada en st.secrets["SUPABASE_DB_URL"] (nunca hardcodeada en el
# codigo). Si el secret no esta configurado, la app sigue funcionando en
# modo "solo plan gratis, sin cuentas" (igual que antes).
# ---------------------------------------------------------------------
def _cadena_conexion_db():
    try:
        return st.secrets["SUPABASE_DB_URL"]
    except Exception:
        return None


DB_DISPONIBLE = psycopg2 is not None and _cadena_conexion_db() is not None
_DB_INICIALIZADA = False


def _ejecutar(consulta, parametros=None, fetch=False, fetchone=False):
    conn = psycopg2.connect(_cadena_conexion_db())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(consulta, parametros or ())
                if fetchone:
                    return cur.fetchone()
                if fetch:
                    return cur.fetchall()
    finally:
        conn.close()


def inicializar_db():
    global _DB_INICIALIZADA
    if _DB_INICIALIZADA or not DB_DISPONIBLE:
        return
    _ejecutar("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            creado TEXT NOT NULL
        )
    """)
    _ejecutar("""
        CREATE TABLE IF NOT EXISTS historial (
            id SERIAL PRIMARY KEY,
            usuario_id INTEGER NOT NULL REFERENCES usuarios(id),
            tipo TEXT,
            resultado TEXT,
            fecha TEXT,
            texto TEXT,
            modelo_usado TEXT,
            explicacion TEXT,
            respuesta_tarea TEXT,
            mensajes_extra TEXT
        )
    """)
    _ejecutar("""
        CREATE TABLE IF NOT EXISTS uso_libre (
            uid TEXT NOT NULL,
            fecha TEXT NOT NULL,
            conteo INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (uid, fecha)
        )
    """)
    _DB_INICIALIZADA = True


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    derivado = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    )
    return derivado.hex(), salt


def crear_usuario(email: str, password: str):
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Escribe un correo valido."
    if len(password) < 6:
        return False, "La contraseña debe tener al menos 6 caracteres."
    existente = _ejecutar("SELECT id FROM usuarios WHERE email = %s", (email,), fetchone=True)
    if existente:
        return False, "Ya existe una cuenta con ese correo."
    hash_hex, salt = _hash_password(password)
    _ejecutar(
        "INSERT INTO usuarios (email, password_hash, salt, creado) VALUES (%s, %s, %s, %s)",
        (email, hash_hex, salt, datetime.now().isoformat()),
    )
    return True, "Cuenta creada. Ahora inicia sesion."


def verificar_usuario(email: str, password: str):
    email = email.strip().lower()
    fila = _ejecutar(
        "SELECT id, email, password_hash, salt FROM usuarios WHERE email = %s",
        (email,), fetchone=True,
    )
    if fila is None:
        return None
    id_usuario, email_db, hash_guardado, salt = fila
    hash_hex, _ = _hash_password(password, salt)
    if hash_hex == hash_guardado:
        return {"id": id_usuario, "email": email_db}
    return None


def guardar_entrada_historial(usuario_id, entrada):
    fila = _ejecutar(
        """INSERT INTO historial
           (usuario_id, tipo, resultado, fecha, texto, modelo_usado,
            explicacion, respuesta_tarea, mensajes_extra)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (
            usuario_id, entrada.get("tipo"), entrada.get("resultado"),
            entrada["fecha"].isoformat(), entrada.get("texto"),
            entrada.get("modelo_usado"), entrada.get("explicacion"),
            entrada.get("respuesta_tarea"),
            json.dumps(entrada.get("mensajes_extra", [])),
        ),
        fetchone=True,
    )
    return fila[0]


def actualizar_mensajes_historial(id_db, mensajes_extra):
    _ejecutar(
        "UPDATE historial SET mensajes_extra = %s WHERE id = %s",
        (json.dumps(mensajes_extra), id_db),
    )


def cargar_historial_usuario(usuario_id):
    filas = _ejecutar(
        """SELECT id, tipo, resultado, fecha, texto, modelo_usado, explicacion,
                  respuesta_tarea, mensajes_extra
           FROM historial WHERE usuario_id = %s ORDER BY id""",
        (usuario_id,), fetch=True,
    )
    resultado = []
    for fila in filas or []:
        (id_db, tipo, res, fecha, texto, modelo_usado, explicacion,
         respuesta_tarea, mensajes_json) = fila
        resultado.append({
            "id_db": id_db, "tipo": tipo, "resultado": res,
            "fecha": datetime.fromisoformat(fecha) if fecha else datetime.now(),
            "texto": texto, "modelo_usado": modelo_usado,
            "explicacion": explicacion, "respuesta_tarea": respuesta_tarea,
            "mensajes_extra": json.loads(mensajes_json) if mensajes_json else [],
        })
    return resultado


def obtener_uid_anonimo():
    uid = st.query_params.get("uid")
    if not uid:
        uid = secrets.token_hex(8)
        st.query_params["uid"] = uid
    return uid


def contar_uso_libre(uid):
    hoy = datetime.now().strftime("%Y-%m-%d")
    fila = _ejecutar(
        "SELECT conteo FROM uso_libre WHERE uid = %s AND fecha = %s",
        (uid, hoy), fetchone=True,
    )
    return fila[0] if fila else 0


def incrementar_uso_libre(uid):
    hoy = datetime.now().strftime("%Y-%m-%d")
    _ejecutar(
        """INSERT INTO uso_libre (uid, fecha, conteo) VALUES (%s, %s, 1)
           ON CONFLICT (uid, fecha) DO UPDATE SET conteo = uso_libre.conteo + 1""",
        (uid, hoy),
    )


def etiqueta_modelo(nombre_modelo: str) -> str:
    """Nombre + descripcion corta, para mostrar en selectores tipo lista."""
    return f"**{nombre_modelo}** — {MODELOS_INFO[nombre_modelo]}"


def llamar_groq(mensajes):
    """
    Envia la conversacion (lista de mensajes role/content, o un string con
    un solo prompt de usuario) a Groq usando with_raw_response para poder
    leer, ademas de la respuesta, los headers de rate limit (tokens/
    requests restantes del plan). Acumula el consumo de tokens en
    st.session_state.
    """
    if isinstance(mensajes, str):
        mensajes = [{"role": "user", "content": mensajes}]

    raw = cliente_groq.chat.completions.with_raw_response.create(
        model=MODELO_GROQ,
        messages=mensajes,
    )
    headers = raw.headers
    completado = raw.parse()

    uso = st.session_state.uso_tokens
    if completado.usage:
        uso["prompt"] += completado.usage.prompt_tokens
        uso["completion"] += completado.usage.completion_tokens
        uso["total"] += completado.usage.total_tokens
        uso["llamadas"] += 1

    def _num(nombre):
        valor = headers.get(nombre)
        try:
            return int(valor)
        except (TypeError, ValueError):
            return None

    st.session_state.rate_limits = {
        "limite_tokens": _num("x-ratelimit-limit-tokens"),
        "restantes_tokens": _num("x-ratelimit-remaining-tokens"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
        "limite_requests": _num("x-ratelimit-limit-requests"),
        "restantes_requests": _num("x-ratelimit-remaining-requests"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
    }

    return completado.choices[0].message.content


def clasificar_tarea(texto_usuario: str) -> str:
    """
    Le pide a la IA que clasifique la tarea del estudiante en una de las
    categorias fijas de TIPOS_TAREA. Regresa el nombre exacto de la
    categoria (una de las llaves del diccionario).
    """
    categorias = ", ".join(TIPOS_TAREA.keys())
    prompt = (
        f"Clasifica la siguiente tarea de un estudiante en EXACTAMENTE una "
        f"de estas categorias: {categorias}. "
        f"Responde SOLO con el nombre exacto de la categoria, nada mas, "
        f"sin explicacion.\n\nTarea del estudiante: \"{texto_usuario}\""
    )
    try:
        categoria = llamar_groq(prompt).strip()
        # Validacion: si la IA regreso algo fuera de las categorias fijas,
        # buscamos la coincidencia mas cercana en vez de fallar en silencio.
        for nombre in TIPOS_TAREA:
            if nombre.lower() in categoria.lower():
                return nombre
        return None
    except Exception as error:
        st.error(f"Error al clasificar la tarea: {error}")
        return None


def generar_explicacion(tipo_tarea: str, modelo_correcto: str) -> str:
    """Genera la explicacion pedagogica de por que ese modelo conviene."""
    prompt = (
        f"Eres parte de una app educativa llamada Klarblick que ensena a "
        f"estudiantes a elegir el modelo de IA correcto para cada tarea. "
        f"Un estudiante tiene una tarea de tipo '{tipo_tarea}'. "
        f"El modelo recomendado para este tipo de tarea es: '{modelo_correcto}'. "
        f"Explica en maximo 3 frases, en español, por que ese tipo de modelo "
        f"conviene para ese tipo de tarea especifica. Se claro y pedagogico, "
        f"como si le hablaras directo al estudiante."
    )
    try:
        return llamar_groq(prompt)
    except Exception as error:
        return f"(Error al generar la explicacion: {error})"


def generar_respuesta_tarea(texto_usuario: str, tipo_tarea: str) -> str:
    """
    Genera la respuesta real a la tarea del estudiante. En este prototipo
    todas las categorias usan el mismo modelo de Groq por debajo (no
    tenemos acceso real a 5 modelos distintos); el nombre del "modelo
    recomendado" es simulado para efectos pedagogicos, pero la respuesta
    en si es generada por IA real.
    """
    prompt = (
        f"Eres un asistente academico ayudando a un estudiante con una "
        f"tarea de tipo '{tipo_tarea}'. Responde directamente a lo que "
        f"pide, en español, de forma clara y bien estructurada.\n\n"
        f"Tarea del estudiante: \"{texto_usuario}\""
    )
    try:
        return llamar_groq(prompt)
    except Exception as error:
        return f"(Error al generar la respuesta: {error})"


def guardar_en_historial(entrada):
    """Agrega la entrada al historial de la sesion y, si hay sesion
    iniciada, la persiste en la base de datos."""
    st.session_state.historial.append(entrada)
    usuario_activo = st.session_state.get("usuario")
    if usuario_activo and DB_DISPONIBLE:
        try:
            entrada["id_db"] = guardar_entrada_historial(usuario_activo["id"], entrada)
        except Exception as error:
            st.caption(f"(No se pudo guardar en tu cuenta: {error})")


def render_continuacion(entrada):
    """
    Muestra los mensajes de seguimiento de una tarea (activa o pasada) y
    una barra de chat al final para seguir la conversacion, en vez de
    tener que empezar una tarea nueva cada vez.
    """
    entrada.setdefault("mensajes_extra", [])
    for mensaje in entrada["mensajes_extra"]:
        with st.chat_message(mensaje["rol"]):
            st.write(mensaje["contenido"])

    nuevo_mensaje = st.chat_input("Continua la conversacion...")
    if nuevo_mensaje:
        conversacion = [
            {"role": "user", "content": entrada.get("texto", "")},
            {"role": "assistant", "content": entrada.get("respuesta_tarea", "")},
        ]
        for mensaje in entrada["mensajes_extra"]:
            conversacion.append({"role": mensaje["rol"], "content": mensaje["contenido"]})
        conversacion.append({"role": "user", "content": nuevo_mensaje})

        with st.spinner("Pensando..."):
            try:
                respuesta = llamar_groq(conversacion)
            except Exception as error:
                respuesta = f"(Error al responder: {error})"

        entrada["mensajes_extra"].append({"rol": "user", "contenido": nuevo_mensaje})
        entrada["mensajes_extra"].append({"rol": "assistant", "contenido": respuesta})

        if st.session_state.get("usuario") and entrada.get("id_db"):
            try:
                actualizar_mensajes_historial(entrada["id_db"], entrada["mensajes_extra"])
            except Exception:
                pass

        st.rerun()


# ---------------------------------------------------------------------
# Estado persistente de la sesion (en memoria; se pierde al reiniciar)
# ---------------------------------------------------------------------
def iniciar_estado():
    if "progreso" not in st.session_state:
        st.session_state.progreso = {
            tipo: {"aciertos_consecutivos": 0, "ultima_fecha": None}
            for tipo in TIPOS_TAREA
        }
    if "historial" not in st.session_state:
        st.session_state.historial = []
    if "pendiente_confirmacion" not in st.session_state:
        st.session_state.pendiente_confirmacion = None
    if "tarea_activa" not in st.session_state:
        st.session_state.tarea_activa = None
    if "uso_tokens" not in st.session_state:
        st.session_state.uso_tokens = {
            "prompt": 0, "completion": 0, "total": 0, "llamadas": 0
        }
    if "rate_limits" not in st.session_state:
        st.session_state.rate_limits = None
    if "vista_historial" not in st.session_state:
        st.session_state.vista_historial = None
    if "usuario" not in st.session_state:
        st.session_state.usuario = None


def probabilidad_de_preguntar(aciertos_consecutivos: int) -> float:
    if aciertos_consecutivos < 2:
        return 0.0
    incrementos = aciertos_consecutivos - 1
    return min(0.8, incrementos * 0.2)


def revisar_inactividad(tipo_tarea: str):
    dato = st.session_state.progreso[tipo_tarea]
    if dato["ultima_fecha"] is None:
        return
    dias_pasados = (datetime.now() - dato["ultima_fecha"]).days
    if dias_pasados >= DIAS_INACTIVIDAD_LIMITE and dato["aciertos_consecutivos"] > 0:
        st.session_state.pendiente_confirmacion = tipo_tarea


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.set_page_config(page_title="klArblIck", page_icon="🔎", layout="centered")
iniciar_estado()
inicializar_db()

# Enter envia el textarea (hace click en "Enviar"); Cmd/Ctrl+Enter inserta
# una linea nueva, como en la mayoria de apps de chat.
components.html(
    """
    <script>
    if (!window.parent.__kbEnterHandlerInstalled) {
        window.parent.__kbEnterHandlerInstalled = true;
        const setter = Object.getOwnPropertyDescriptor(
            window.parent.HTMLTextAreaElement.prototype, 'value'
        ).set;
        window.parent.document.addEventListener('keydown', function(e) {
            const esTextareaDescribir = e.target && e.target.tagName === 'TEXTAREA' &&
                e.target.getAttribute('aria-label') === 'Escribe aqui lo que necesitas hacer';
            if (esTextareaDescribir && e.key === 'Enter') {
                const ta = e.target;
                if (e.metaKey || e.ctrlKey) {
                    // Cmd/Ctrl+Enter: insertar salto de linea manualmente,
                    // bloqueando el atajo nativo de Streamlit (aplicar valor).
                    e.preventDefault();
                    e.stopPropagation();
                    const start = ta.selectionStart;
                    const end = ta.selectionEnd;
                    const nuevoValor = ta.value.slice(0, start) + '\\n' + ta.value.slice(end);
                    setter.call(ta, nuevoValor);
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.selectionStart = ta.selectionEnd = start + 1;
                } else {
                    // Enter solo: enviar la tarea.
                    e.preventDefault();
                    e.stopPropagation();
                    const botones = window.parent.document.querySelectorAll('button');
                    for (const b of botones) {
                        if (b.innerText.trim() === 'Enviar') {
                            b.click();
                            break;
                        }
                    }
                }
            }
        }, true);
    }
    </script>
    """,
    height=0,
)

# ---------------------------------------------------------------------
# Identidad visual original: acento cian/menta y tipografia monoespaciada
# tipo terminal, pero sobre fondo claro (no oscuro). Logo: simbolo ">"
# partido en dos trazos con un rombo de luz en el hueco ("apertura hacia
# la claridad").
# ---------------------------------------------------------------------
COLOR_FONDO = "#FFFFFF"
COLOR_FONDO_PANEL = "#F2FBF8"
COLOR_ACENTO = "#0E9C79"
COLOR_TEXTO = "#0D1117"
COLOR_TEXTO_TENUE = "#6B6560"
COLOR_BORDE = "#DCEEE8"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }}
    .stApp {{
        background-color: {COLOR_FONDO};
        color: {COLOR_TEXTO};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_FONDO_PANEL};
        border-right: 1px solid {COLOR_BORDE};
    }}
    .stApp header {{
        background-color: {COLOR_FONDO};
    }}
    p, span, label, li, .stMarkdown, .stCaption {{
        color: {COLOR_TEXTO} !important;
    }}
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {COLOR_FONDO} !important;
        color: {COLOR_TEXTO} !important;
        border-color: {COLOR_BORDE} !important;
    }}
    .stButton button {{
        background-color: {COLOR_ACENTO};
        color: #FFFFFF;
        border: 1px solid {COLOR_ACENTO};
        font-weight: 600;
    }}
    .stButton button:hover {{
        background-color: #0B7A5F;
        color: #FFFFFF;
        border-color: #0B7A5F;
    }}
    hr {{
        border-color: {COLOR_BORDE} !important;
    }}
    .kb-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }}
    .kb-wordmark {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        color: {COLOR_TEXTO};
    }}
    .kb-wordmark .kb-hl {{
        color: {COLOR_ACENTO} !important;
        font-weight: 800;
    }}
    .kb-slogan {{
        color: {COLOR_TEXTO_TENUE};
        font-size: 0.95rem;
        margin-top: -4px;
        margin-bottom: 1.2rem;
        font-style: italic;
    }}
    </style>

    <div class="kb-header">
        <svg width="42" height="42" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg">
            <polyline points="6,10 20,21 6,32" fill="none" stroke="{COLOR_ACENTO}"
                stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
            <polyline points="20,10 34,21 20,32" fill="none" stroke="{COLOR_TEXTO}"
                stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
            <rect x="18" y="19" width="5" height="5" fill="{COLOR_ACENTO}"
                transform="rotate(45 20.5 21.5)" />
        </svg>
        <div>
            <div class="kb-wordmark">kl<span class="kb-hl">A</span>rbl<span class="kb-hl">I</span>ck</div>
        </div>
    </div>
    <div class="kb-slogan">no eliges un modelo, eliges entender</div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Escribe tu tarea con tus propias palabras. La IA identifica el tipo "
    "de tarea, elige el modelo recomendado, y a veces te preguntara a ti "
    "primero para que desarrolles el criterio de eleccion."
)

with st.expander("Modelos disponibles"):
    for nombre_modelo, descripcion in MODELOS_INFO.items():
        st.markdown(f"**{nombre_modelo}**")
        st.caption(descripcion)

if cliente_groq is None:
    st.error("No se configuro GROQ_API_KEY. Revisa tu archivo .env.")
    st.stop()

# --- Confirmacion de inactividad pendiente ---
if st.session_state.pendiente_confirmacion:
    tipo = st.session_state.pendiente_confirmacion
    st.warning(
        f"No has usado '{tipo}' en un tiempo. "
        f"¿Sigues trabajando en este tipo de tarea o ya no?"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sigo en ello", key="sigue_si"):
            st.session_state.progreso[tipo]["ultima_fecha"] = datetime.now()
            st.session_state.pendiente_confirmacion = None
            st.rerun()
    with col2:
        if st.button("Ya no", key="sigue_no"):
            st.session_state.progreso[tipo]["aciertos_consecutivos"] = 0
            st.session_state.progreso[tipo]["ultima_fecha"] = None
            st.session_state.pendiente_confirmacion = None
            st.rerun()
    st.stop()

# --- Vista de solo lectura de una tarea pasada (clic en el historial) ---
if st.session_state.vista_historial is not None and not st.session_state.tarea_activa:
    entrada = st.session_state.vista_historial
    st.divider()
    st.caption(f"Tarea pasada · **{entrada['tipo']}** · {entrada['fecha'].strftime('%d/%m %H:%M')}")
    st.markdown(f"**Tu mensaje:** {entrada.get('texto', '(sin registrar)')}")
    if entrada.get("modelo_usado"):
        st.caption(f"Modelo usado: {entrada['modelo_usado']}")
    if entrada.get("explicacion"):
        st.write(entrada["explicacion"])
    if entrada.get("respuesta_tarea"):
        st.markdown("**Respuesta a tu tarea:**")
        st.write(entrada["respuesta_tarea"])
    render_continuacion(entrada)

# --- Paso 1: el estudiante escribe su tarea libremente ---
elif not st.session_state.tarea_activa:
    st.subheader("Describe tu tarea")
    texto_usuario = st.text_area(
        "Escribe aqui lo que necesitas hacer",
        placeholder="Ej: Necesito escribir un ensayo sobre el cambio climatico...",
        key="texto_tarea",
    )

    usuario_activo = st.session_state.usuario
    opciones_modelo_paso1 = ["Automatico (el sistema elige por ti)"]
    if usuario_activo:
        opciones_modelo_paso1 += OPCIONES_MODELO
    seleccion_modelo = st.selectbox("Modelo", opciones_modelo_paso1, key="modelo_manual_select")
    elegir_manual = seleccion_modelo != "Automatico (el sistema elige por ti)"
    modelo_elegido_manual = seleccion_modelo if elegir_manual else None
    if elegir_manual:
        st.caption(MODELOS_INFO[modelo_elegido_manual])
    elif not usuario_activo:
        st.caption("Inicia sesion para elegir el modelo tu mismo.")

    uid_anonimo = None
    usos_hoy = 0
    limite_alcanzado = False
    if not usuario_activo and DB_DISPONIBLE:
        try:
            uid_anonimo = obtener_uid_anonimo()
            usos_hoy = contar_uso_libre(uid_anonimo)
        except Exception:
            usos_hoy = 0
        limite_alcanzado = usos_hoy >= LIMITE_DIARIO_LIBRE
        if limite_alcanzado:
            st.warning(
                f"Alcanzaste el limite del plan gratis ({LIMITE_DIARIO_LIBRE} tareas/dia). "
                "Inicia sesion para seguir sin limite."
            )
        else:
            st.caption(f"Plan gratis: {usos_hoy}/{LIMITE_DIARIO_LIBRE} tareas usadas hoy.")

    if st.button("Enviar", disabled=limite_alcanzado) and texto_usuario.strip():
        with st.spinner("Identificando el tipo de tarea..."):
            tipo_detectado = clasificar_tarea(texto_usuario)

        if tipo_detectado is None:
            st.error(
                "No se pudo identificar el tipo de tarea. Intenta ser mas "
                "especifico (por ejemplo, menciona si es un ensayo, un "
                "resumen, codigo, analisis de datos o generacion de imagenes)."
            )
        else:
            if not usuario_activo and DB_DISPONIBLE and uid_anonimo:
                try:
                    incrementar_uso_libre(uid_anonimo)
                except Exception:
                    pass

            if elegir_manual:
                st.session_state.tarea_activa = {
                    "texto": texto_usuario,
                    "tipo": tipo_detectado,
                    "debe_preguntar": False,
                    "prob_mostrada": 0.0,
                    "modelo_manual": modelo_elegido_manual,
                }
                st.rerun()

            revisar_inactividad(tipo_detectado)
            if st.session_state.pendiente_confirmacion:
                st.rerun()

            dato = st.session_state.progreso[tipo_detectado]
            prob = probabilidad_de_preguntar(dato["aciertos_consecutivos"])
            debe_preguntar = random.random() < prob

            st.session_state.tarea_activa = {
                "texto": texto_usuario,
                "tipo": tipo_detectado,
                "debe_preguntar": debe_preguntar,
                "prob_mostrada": prob,
                "modelo_manual": None,
            }
            st.rerun()

# --- Paso 2 en adelante ---
if st.session_state.tarea_activa:
    activa = st.session_state.tarea_activa
    tipo = activa["tipo"]
    modelo_correcto = TIPOS_TAREA[tipo]

    st.divider()
    st.caption(f"Tarea detectada: **{tipo}**")

    ya_procesada = "respuesta_tarea" in activa

    if activa.get("modelo_manual"):
        modelo_manual = activa["modelo_manual"]
        st.info(f"Elegiste tú: {modelo_manual}")
        if modelo_manual == modelo_correcto:
            st.caption("Coincide con el modelo que el sistema habria recomendado.")
        else:
            st.caption(f"El sistema habria recomendado: {modelo_correcto}")

        if not ya_procesada:
            with st.spinner("Generando explicacion..."):
                activa["explicacion"] = generar_explicacion(tipo, modelo_correcto)
            with st.spinner("Generando respuesta a tu tarea..."):
                activa["respuesta_tarea"] = generar_respuesta_tarea(activa["texto"], tipo)
            activa["modelo_usado"] = modelo_manual
            activa["resultado"] = "manual"
            activa["fecha"] = datetime.now()
            guardar_en_historial(activa)

        st.write(activa["explicacion"])
        st.markdown("**Respuesta a tu tarea:**")
        st.write(activa["respuesta_tarea"])
        render_continuacion(activa)

    elif not activa["debe_preguntar"]:
        st.caption(
            f"Probabilidad de pregunta esta vez: {activa['prob_mostrada']*100:.0f}% "
            f"(aciertos consecutivos actuales: {st.session_state.progreso[tipo]['aciertos_consecutivos']})"
        )
        st.success(f"Modelo recomendado: {modelo_correcto}")

        if not ya_procesada:
            with st.spinner("Generando explicacion..."):
                activa["explicacion"] = generar_explicacion(tipo, modelo_correcto)
            with st.spinner("Generando respuesta a tu tarea..."):
                activa["respuesta_tarea"] = generar_respuesta_tarea(activa["texto"], tipo)
            activa["modelo_usado"] = modelo_correcto
            activa["resultado"] = "directo"
            activa["fecha"] = datetime.now()
            st.session_state.progreso[tipo]["ultima_fecha"] = activa["fecha"]
            guardar_en_historial(activa)

        st.write(activa["explicacion"])
        st.markdown("**Respuesta a tu tarea:**")
        st.write(activa["respuesta_tarea"])
        render_continuacion(activa)

    else:
        if not ya_procesada:
            st.caption(
                f"Probabilidad de pregunta esta vez: {activa['prob_mostrada']*100:.0f}% "
                f"(aciertos consecutivos actuales: {st.session_state.progreso[tipo]['aciertos_consecutivos']})"
            )
            st.info("Antes de responder: ¿que modelo usarias tu para esta tarea?")
            respuesta = st.radio(
                "Elige un modelo",
                OPCIONES_MODELO,
                format_func=etiqueta_modelo,
                key="respuesta_usuario",
            )

            if st.button("Confirmar respuesta"):
                acerto = respuesta == modelo_correcto
                st.session_state.progreso[tipo]["ultima_fecha"] = datetime.now()

                with st.spinner("Generando explicacion..."):
                    explicacion = generar_explicacion(tipo, modelo_correcto)

                if acerto:
                    st.session_state.progreso[tipo]["aciertos_consecutivos"] += 1
                    activa["resultado"] = "acierto"
                else:
                    st.session_state.progreso[tipo]["aciertos_consecutivos"] = 0
                    activa["resultado"] = "error"

                with st.spinner("Generando respuesta a tu tarea..."):
                    respuesta_tarea = generar_respuesta_tarea(activa["texto"], tipo)

                activa["explicacion"] = explicacion
                activa["respuesta_tarea"] = respuesta_tarea
                activa["modelo_usado"] = modelo_correcto
                activa["modelo_elegido_usuario"] = respuesta
                activa["acerto"] = acerto
                activa["fecha"] = datetime.now()
                guardar_en_historial(activa)
                st.rerun()
        else:
            if activa.get("acerto"):
                st.success("Correcto.")
            else:
                st.error(f"No era ese. El modelo correcto es: {modelo_correcto}")
            st.write(activa["explicacion"])
            st.markdown("**Respuesta a tu tarea:**")
            st.write(activa["respuesta_tarea"])
            render_continuacion(activa)

# --- Historial visible: barra lateral ---
with st.sidebar:
    st.markdown("### Cuenta")
    if not DB_DISPONIBLE:
        st.caption("Sistema de cuentas no configurado todavia (plan gratis para todos).")
    elif st.session_state.usuario:
        st.success(f"👋 {st.session_state.usuario['email']}")
        if st.button("Cerrar sesion", use_container_width=True):
            st.session_state.usuario = None
            st.session_state.historial = []
            st.session_state.tarea_activa = None
            st.session_state.vista_historial = None
            st.rerun()
    else:
        st.caption(
            f"Plan gratis: sin historial guardado, modelo automatico, "
            f"{LIMITE_DIARIO_LIBRE} tareas/dia."
        )
        modo_auth = st.radio(
            "Modo", ["Iniciar sesion", "Crear cuenta"],
            horizontal=True, label_visibility="collapsed", key="modo_auth",
        )
        correo_auth = st.text_input("Correo", key="auth_correo")
        contrasena_auth = st.text_input("Contraseña", type="password", key="auth_contrasena")

        if modo_auth == "Iniciar sesion":
            if st.button("Entrar", use_container_width=True):
                try:
                    usuario_encontrado = verificar_usuario(correo_auth, contrasena_auth)
                except Exception as error:
                    usuario_encontrado = None
                    st.error(f"Error de conexion: {error}")
                if usuario_encontrado:
                    st.session_state.usuario = usuario_encontrado
                    st.session_state.historial = cargar_historial_usuario(usuario_encontrado["id"])
                    st.session_state.tarea_activa = None
                    st.session_state.vista_historial = None
                    st.rerun()
                elif correo_auth:
                    st.error("Correo o contraseña incorrectos.")
        else:
            if st.button("Crear cuenta", use_container_width=True):
                try:
                    ok, mensaje = crear_usuario(correo_auth, contrasena_auth)
                except Exception as error:
                    ok, mensaje = False, f"Error de conexion: {error}"
                if ok:
                    st.success(mensaje)
                else:
                    st.error(mensaje)

    st.divider()
    st.markdown("### Historial")
    if st.session_state.tarea_activa or st.session_state.vista_historial:
        if st.button("+ Nueva tarea", use_container_width=True):
            st.session_state.tarea_activa = None
            st.session_state.vista_historial = None
            st.session_state["mostrar_siguiente"] = False
            st.rerun()
    st.divider()
    if not st.session_state.historial:
        st.caption("Todavia no hay historial.")
    else:
        for indice, entrada in reversed(list(enumerate(st.session_state.historial))):
            texto_corto = (entrada.get("texto") or entrada["tipo"])[:40]
            if len(entrada.get("texto", "")) > 40:
                texto_corto += "…"
            icono = {"acierto": "✅", "error": "❌", "directo": "💬", "manual": "🛠️"}.get(
                entrada["resultado"], "💬"
            )
            if st.button(
                f"{icono} {texto_corto}",
                key=f"hist_{indice}",
                use_container_width=True,
            ):
                st.session_state.tarea_activa = None
                st.session_state.vista_historial = entrada
                st.rerun()
            st.caption(f"{entrada['tipo']} · {entrada['fecha'].strftime('%d/%m %H:%M')}")

        with st.expander("Resumen por tipo de tarea"):
            for tipo in TIPOS_TAREA:
                entradas = [h for h in st.session_state.historial if h["tipo"] == tipo]
                if not entradas:
                    continue
                aciertos = sum(1 for e in entradas if e["resultado"] in ("acierto", "directo"))
                errores = sum(1 for e in entradas if e["resultado"] == "error")
                manuales = sum(1 for e in entradas if e["resultado"] == "manual")
                st.markdown(f"**{tipo}**")
                st.caption(
                    f"{len(entradas)} usos · {aciertos} aciertos · "
                    f"{errores} errores · {manuales} manual"
                )

    st.divider()
    st.markdown("### Uso de tokens")
    uso = st.session_state.uso_tokens
    st.caption(
        f"Esta sesion: {uso['total']} tokens en {uso['llamadas']} llamadas "
        f"({uso['prompt']} de entrada, {uso['completion']} de salida)"
    )

    limites = st.session_state.rate_limits
    if limites is None:
        st.caption("Los limites del plan se muestran despues de tu primera tarea.")
    else:
        if limites["limite_tokens"] and limites["restantes_tokens"] is not None:
            usados_tok = limites["limite_tokens"] - limites["restantes_tokens"]
            st.caption(
                f"Tokens/min del plan: {usados_tok}/{limites['limite_tokens']} "
                f"usados · {limites['restantes_tokens']} restantes "
                f"(se reinicia en {limites['reset_tokens']})"
            )
            st.progress(min(1.0, usados_tok / limites["limite_tokens"]))
        if limites["limite_requests"] and limites["restantes_requests"] is not None:
            usados_req = limites["limite_requests"] - limites["restantes_requests"]
            st.caption(
                f"Solicitudes/dia del plan: {usados_req}/{limites['limite_requests']} "
                f"usadas · {limites['restantes_requests']} restantes "
                f"(se reinicia en {limites['reset_requests']})"
            )
            st.progress(min(1.0, usados_req / limites["limite_requests"]))
