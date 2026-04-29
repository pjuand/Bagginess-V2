df = pd.read_excel(archivo_cargado, sheet_name="Dureza (2)")

    if row["Prob_modelo"] >= 0.80:

        return "🔴 RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.6 and row["STDEV"] > 3.3):

        return "🔴 RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.5 or row["STDEV"] > 3):

# ==========================================

# 1️⃣ LIBRERÍAS

# ==========================================

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

 

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

 

# ==========================================

# 2️⃣ CARGA DE DATOS

# ==========================================

df = pd.read_excel(archivo_cargado, sheet_name="Dureza (2)")

    if row["Prob_modelo"] >= 0.80:

        return "🔴 RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.6 and row["STDEV"] > 3.3):

        return "🔴 RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.5 or row["STDEV"] > 3):

import pandas as pd

import numpy as np

 

# ==========================================

# 1️⃣ CARGAR ARCHIVO

# ==========================================

from google.colab import files

uploaded = files.upload()

 

archivo = list(uploaded.keys())[0]

 

df = pd.read_excel(archivo)

df.columns = df.columns.str.strip()

 

print("Columnas detectadas:")

print(df.columns)

 

# ==========================================

# 2️⃣ CALCULAR VARIABLES AUTOMÁTICAS

# ==========================================

 

def calcular_variables(row):

 

    # Extraer las 20 durezas

    durezas = row[[f"Dureza Rollo ({i})" for i in range(1,21)]].values.astype(float)

 

    arr = np.array(durezas)

 

    # Zonas

    M5 = np.mean(arr[0:4])

    M4 = np.mean(arr[4:8])

    M3 = np.mean(arr[8:12])

    M1 = np.mean(arr[16:20])

 

    # Variables

    Variabilidad = np.max(arr) - np.min(arr)

    STDEV = np.std(arr)

    Simetria = M5 - M1

 

    Extremo = np.mean(np.concatenate([arr[1:4], arr[16:19]]))

    Curvatura = Extremo - M3

 

    Simetria_L3 = M5 - M4

 

    return pd.Series({

        "Variabilidad": Variabilidad,

        "Simetría": Simetria,

        "Curvatura": Curvatura,

        "STDEV": STDEV,

        "M5": M5,

        "M1": M1,

        "M3": M3,

        "M4": M4,

        "Simetría L3": Simetria_L3

    })

 

df_vars = df.apply(calcular_variables, axis=1)

df = pd.concat([df, df_vars], axis=1)

 

# ==========================================

# 3️⃣ PROBABILIDAD DEL MODELO

# (IMPORTANTE: ya debes tener xgb entrenado)

# ==========================================

 

features = [

    "Variabilidad","Simetría","Curvatura","STDEV",

    "M5","M1","M3","M4","Simetría L3"

]

 

df["Prob_modelo"] = xgb.predict_proba(df[features])[:,1]

 

# ==========================================

# 4️⃣ CLASIFICACIÓN

# ==========================================

 

def clasificar(row):

 

    if row["Prob_modelo"] >= 0.80:

        return "RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.6 and row["STDEV"] > 3.3):

        return "RECHAZAR"

 

    elif (row["Prob_modelo"] >= 0.5 or row["STDEV"] > 3):

        return "ALERTA"

 

    else:

        return "OK"

 

df["Decision"] = df.apply(clasificar, axis=1)

 

# ==========================================

# 5️⃣ RESUMEN POR FECHA

# ==========================================

 

resumen = df.groupby("Date").agg(

    Total_Reeles=("Name","count"),

    Rechazados=("Decision", lambda x: (x=="RECHAZAR").sum()),

    Alerta=("Decision", lambda x: (x=="ALERTA").sum()),

    OK=("Decision", lambda x: (x=="OK").sum())

).reset_index()

 

print("\n📊 RESUMEN POR FECHA")

print(resumen)

 

# ==========================================

# 6️⃣ LISTA DE REELES CRÍTICOS

# ==========================================

 

rechazados = df[df["Decision"]=="RECHAZAR"][["Date","Name","Product (short)","Prob_modelo"]]

alerta = df[df["Decision"]=="ALERTA"][["Date","Name","Product (short)","Prob_modelo"]]

 

print("\n🔴 REELES A RECHAZAR:")

print(rechazados)

 

print("\n🟡 REELES EN ALERTA:")

print(alerta)

 

# ==========================================

# 7️⃣ ANÁLISIS AUTOMÁTICO

# ==========================================

 

def analisis_fecha(grupo):

 

    texto = ""

 

    if grupo["STDEV"].mean() > 3:

        texto += "Alta variabilidad detectada. "

 

    if grupo["Variabilidad"].mean() > 10:

        texto += "Amplio rango de dureza. "

 

    if grupo["Curvatura"].mean() > 3:

        texto += "Posible problema de perfil transversal. "

 

    if texto == "":

        texto = "Condiciones estables."

 

    return texto

 

analisis = df.groupby("Date").apply(analisis_fecha).reset_index()

analisis.columns = ["Date","Analisis"]

 

print("\n🧠 ANÁLISIS AUTOMÁTICO:")

print(analisis)

 

# ==========================================

# 8️⃣ EXPORTAR RESULTADOS

# ==========================================

 

with pd.ExcelWriter("resultado_final.xlsx") as writer:

    df.to_excel(writer, sheet_name="Detalle", index=False)

    resumen.to_excel(writer, sheet_name="Resumen", index=False)

    rechazados.to_excel(writer, sheet_name="Rechazados", index=False)

    alerta.to_excel(writer, sheet_name="Alertas", index=False)

    analisis.to_excel(writer, sheet_name="Analisis", index=False)

 

print("\n✅ Archivo generado: resultado_final.xlsx")

from google.colab import files

files.download("resultado_final.xlsx")