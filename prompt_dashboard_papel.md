Quiero que desarrolles una aplicación profesional tipo dashboard industrial para control de calidad en producción de papel (test liner), enfocada en la predicción de bagginess a partir de mediciones de dureza.

La aplicación debe ser clara, visual, intuitiva para operarios y útil para toma de decisiones en planta. No debe sentirse académica, sino práctica e ingenieril.

🎯 OBJETIVO DE LA APLICACIÓN

Permitir predecir y prevenir reclamos de clientes mediante un modelo de machine learning basado en mediciones de dureza en 20 puntos del ancho de la bobina.

La herramienta debe ayudar a:

Detectar rollos con riesgo de bagginess
Tomar decisiones en tiempo real (rechazar, alerta, OK)
Analizar estabilidad del proceso productivo
Reducir reclamos en corrugadoras
🧩 ESTRUCTURA DE LA APLICACIÓN

🔐 PARTE 1: MODO ADMINISTRADOR (ENTRENAMIENTO DEL MODELO)

Debe permitir:

📥 Carga de archivo Excel

Contiene:
20 mediciones de dureza
Variables calculadas (o se pueden recalcular)
Columna objetivo: Reclamado (0 o 1)
⚙️ Procesamiento

Calcular automáticamente:
Variabilidad (max - min)
STDEV
Simetría
Curvatura
M1, M3, M4, M5
Simetría L3
🤖 Entrenamiento modelo

Usar XGBoost
Manejar desbalanceo de clases
Generar:
Probabilidad de reclamo (Prob_modelo)
📊 Mostrar resultados:

Matriz de confusión
Precision, Recall, F1-score
Importancia de variables
Gráficas:
Riesgo vs Variabilidad
Riesgo vs Simetría
Riesgo vs Curvatura
🧪 PARTE 2: PREDICCIÓN EN TIEMPO REAL (REEL A REEL)

Interfaz pensada para operarios de calidad.

🧾 Inputs:

20 mediciones de dureza
👉 Debe permitir pegar datos así (columna):

30,00
44,00
42,00
...

⚙️ Procesamiento automático:

Calcular:

Variabilidad
STDEV
Simetría
Curvatura
M1, M3, M4, M5
Simetría L3
🎯 Salidas:

🔥 Resultado principal:

🔴 RECHAZAR
🟡 ALERTA
🟢 OK
📊 Indicadores:

Probabilidad de reclamo (%)
STDEV
Curvatura
Simetría
Variabilidad
🎨 VISUALIZACIÓN CLAVE (MUY IMPORTANTE)

Mostrar el perfil de dureza como:

🧻 “Bobina visual”

Representación horizontal del ancho del papel
Curva de dureza (línea)
Colores por nivel de riesgo:
Verde (estable)
Amarillo (variación)
Rojo (crítico)
Esto debe ayudar a entender cómo está distribuida la tensión en la hoja.

📂 PARTE 3: ANÁLISIS HISTÓRICO

📥 Carga de archivo Excel con múltiples reels

📊 Salidas:

1. Resumen por día:

Total reels
Rechazados
Alerta
OK
2. Listado:

Reeles rechazados
Reeles en alerta
3. Análisis automático:

Ejemplos:

"Alta variabilidad en el proceso"
"Posible problema de perfil transversal"
"Condiciones estables"
Basado en:

STDEV promedio
Variabilidad
Curvatura
🧠 LÓGICA DE DECISIÓN

Basada en combinación de:

🤖 Modelo ML:

Probabilidad de reclamo

⚙️ Reglas físicas:

STDEV
Curvatura
Variabilidad
🎯 Clasificación final:

Si Prob ≥ 0.80 → RECHAZAR

 

Si Prob ≥ 0.60 y STDEV > 3.3 → RECHAZAR

 

Si Prob ≥ 0.50 o STDEV > 3 → ALERTA

 

Si no → OK

🎨 DISEÑO (MUY IMPORTANTE)

Fondo blanco
Colores:
Azul (información)
Verde (OK)
Amarillo (alerta)
Rojo (rechazo)
Morado (analítica)
Estilo:

Limpio
Industrial
Profesional
Fácil de usar en planta
🧭 ENFOQUE DEL LENGUAJE

La app debe:

Hablar en lenguaje de producción (no académico)
Ser directa
Orientada a decisiones
Ejemplo:
❌ “Alta desviación estándar detectada”
✅ “Alta variabilidad en la dureza – posible riesgo de bagginess”

⚡ REQUISITOS TÉCNICOS

Backend en Python
Modelo XGBoost
Interfaz tipo:
Streamlit (preferido) o web app
Debe permitir:
Subir archivos
Descargar resultados en Excel
🏭 CONTEXTO IMPORTANTE

Esta herramienta será usada en planta de producción, por lo que:

Debe ayudar a retener material con riesgo
Debe reducir reclamos en corrugadoras
No depende de una sola variable (usa modelo + física)
Prioriza precisión y bajo falso positivo
🎯 RESULTADO ESPERADO

Una aplicación que:

Permita entrenar el modelo
Permita predecir en tiempo real
Permita analizar históricos
Sea visual, clara y útil para operación