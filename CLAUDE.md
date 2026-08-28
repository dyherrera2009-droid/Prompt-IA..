# Prompt IA

Prototipo (Streamlit + Groq) de una app educativa que orienta a estudiantes a elegir el modelo de IA correcto segun el tipo de tarea, explicando el porque, para que desarrollen criterio propio.

## Mecanismo pedagogico

1. El estudiante describe su tarea en texto libre. La IA la clasifica en una de 5 categorias fijas (Ensayo, Resumen, Codigo, Analisis de datos, Generacion de imagenes) — la asignacion tipo→modelo es una regla de producto fija en `TIPOS_TAREA`, no la decide la IA.
2. Por cada tipo de tarea hay un contador de aciertos consecutivos.
3. La probabilidad de que la app le PREGUNTE al usuario (en vez de responder directo) empieza en 0% los primeros 2 usos y sube 20% por acierto consecutivo, hasta un techo de 80%.
4. Si pregunta y el usuario falla, se resetea el contador a 0 y se le explica el porque del modelo correcto (no se le da pista).
5. Si acierta, sube el contador.
6. El contador expira tras 14 dias de inactividad, pero antes de borrarlo se le pregunta al usuario si sigue en ese tipo de tarea.
7. Siempre se genera tambien la respuesta real a la tarea (no solo la explicacion del modelo), y se puede seguir la conversacion tipo chat despues.

## Limitaciones conocidas

- Los "5 modelos" (Modelo A-E) son etiquetas simuladas: todas las categorias corren sobre el mismo modelo de Groq (`openai/gpt-oss-120b`) por debajo. No hay acceso real a 5 modelos distintos — no inflar esto en una presentacion.
- El historial y el progreso viven solo en memoria de sesion (`st.session_state`); se pierden al cerrar o recargar la pestaña.
- Requiere `GROQ_API_KEY` en un archivo `.env` local (o en Secrets si se corre en Streamlit Cloud).
