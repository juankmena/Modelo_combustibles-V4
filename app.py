import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def leer_csv_eventos_robusto(ruta):
    """Lee CSV de eventos con separador ; o , y tolera textos con comas."""
    import pandas as pd
    try:
        return pd.read_csv(ruta, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(ruta, sep=",", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception as e:
            raise e

st.set_page_config(
    page_title="Modelo combustibles EIA",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Observatorio experimental del mercado internacional de combustibles")
st.caption("Datos públicos EIA | Clasificación supervisada | Compras Recope | Eventos relevantes de mercado")

URL_WTI = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
URL_GASOLINA = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
URL_DIESEL = "https://www.eia.gov/dnav/pet/hist_xls/EMD_EPD2D_PTE_NUS_DPGw.xls"
HISTORICO_PATH = "historico_predicciones_combustibles.csv"
RECOPE_PATH = "2016-2026_abril.csv"
EVENTOS_PATH = "eventos_geopoliticos_energia_2016_2026.csv"


def leer_archivo_eia(url: str, nombre_variable: str) -> pd.DataFrame:
    """Lee archivos históricos XLS de EIA desde la hoja Data 1.

    Los archivos históricos de EIA usados por esta app son .xls antiguos.
    Por eso se debe usar xlrd, no openpyxl. openpyxl solo funciona con .xlsx
    y produce BadZipFile cuando se intenta leer un .xls.
    """
    try:
        df_raw = pd.read_excel(
            url,
            sheet_name="Data 1",
            skiprows=2,
            engine="xlrd"
        )
    except ImportError as exc:
        st.error(
            "Falta instalar la dependencia xlrd para leer los archivos .xls de EIA. "
            "Agregue 'xlrd>=2.0.1' al requirements.txt y vuelva a desplegar la app."
        )
        raise exc
    except Exception as exc:
        st.error(
            "No fue posible leer el archivo histórico de EIA. "
            "Verifique conexión a internet, disponibilidad del enlace y formato del archivo."
        )
        raise exc

    df_raw = df_raw.dropna(how="all")

    fecha_col = df_raw.columns[0]
    valor_col = df_raw.columns[1]

    df = df_raw[[fecha_col, valor_col]].copy()
    df.columns = ["fecha", nombre_variable]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df[nombre_variable] = pd.to_numeric(df[nombre_variable], errors="coerce")

    df = df.dropna(subset=["fecha", nombre_variable])
    df = df.sort_values("fecha").reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    wti_diario = leer_archivo_eia(URL_WTI, "wti")
    gasolina = leer_archivo_eia(URL_GASOLINA, "gasolina_regular")
    diesel = leer_archivo_eia(URL_DIESEL, "diesel")

    gasolina = gasolina.sort_values("fecha")
    diesel = diesel.sort_values("fecha")
    wti_diario = wti_diario.sort_values("fecha")

    df = pd.merge_asof(
        gasolina,
        diesel,
        on="fecha",
        direction="nearest",
        tolerance=pd.Timedelta(days=4)
    )

    df = pd.merge_asof(
        df.sort_values("fecha"),
        wti_diario[["fecha", "wti"]].sort_values("fecha"),
        on="fecha",
        direction="backward"
    )

    df = df.dropna(subset=["gasolina_regular", "diesel", "wti"])
    df = df.sort_values("fecha").reset_index(drop=True)

    return df


def preparar_modelo(df: pd.DataFrame):
    datos = df.copy().sort_values("fecha").reset_index(drop=True)

    # Variables explicativas derivadas del comportamiento reciente del mercado.
    datos["gasolina_lag1"] = datos["gasolina_regular"].shift(1)
    datos["gasolina_lag2"] = datos["gasolina_regular"].shift(2)
    datos["diesel_lag1"] = datos["diesel"].shift(1)
    datos["wti_lag1"] = datos["wti"].shift(1)
    datos["var_gasolina"] = datos["gasolina_regular"].pct_change()
    datos["var_wti"] = datos["wti"].pct_change()
    datos["media_gasolina_4s"] = datos["gasolina_regular"].rolling(4).mean()
    datos["volatilidad_gasolina_4s"] = datos["gasolina_regular"].rolling(4).std()
    datos["mes"] = datos["fecha"].dt.month

    # Variable objetivo: 1 si la gasolina sube en la siguiente observación.
    datos["sube_gasolina"] = (
        datos["gasolina_regular"].shift(-1) > datos["gasolina_regular"]
    ).astype(int)

    # La última fila no tiene resultado real observable del siguiente periodo.
    datos_modelo = datos.iloc[:-1].dropna().copy()
    datos_prediccion = datos.dropna().copy()

    if len(datos_modelo) < 30:
        raise ValueError(
            f"Después de preparar los datos quedaron solo {len(datos_modelo)} registros. "
            "Revise la descarga de EIA y la unión de fechas."
        )

    columnas_x = [
        "gasolina_regular", "diesel", "wti", "gasolina_lag1", "gasolina_lag2",
        "diesel_lag1", "wti_lag1", "var_gasolina", "var_wti", "media_gasolina_4s",
        "volatilidad_gasolina_4s", "mes",
    ]

    X = datos_modelo[columnas_x]
    y = datos_modelo["sube_gasolina"]

    train_size = int(len(datos_modelo) * 0.8)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    fechas_test = datos_modelo.iloc[train_size:]["fecha"]

    param_grid = {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 3, 5, 10]}
    base_model = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    modelo = grid.best_estimator_
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)

    # Predicción del último periodo disponible.
    ultima_fila = datos_prediccion.iloc[[-1]].copy()
    X_ultimo = ultima_fila[columnas_x]

    pred_final = int(modelo.predict(X_ultimo)[0])
    prob_final = float(modelo.predict_proba(X_ultimo)[0][1])

    frecuencia = datos["fecha"].diff().mode()[0]
    fecha_base = ultima_fila["fecha"].iloc[0]
    periodo_predicho = fecha_base + frecuencia

    resultados = pd.DataFrame({
        "fecha": fechas_test.values,
        "real": y_test.values,
        "prediccion": y_pred,
        "probabilidad_sube": y_proba,
    })
    resultados["acierto"] = resultados["real"] == resultados["prediccion"]

    return {
        "datos": datos,
        "datos_modelo": datos_modelo,
        "modelo": modelo,
        "mejores_parametros": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "matriz": confusion_matrix(y_test, y_pred),
        "reporte": classification_report(y_test, y_pred, target_names=["No sube", "Sube"], output_dict=True),
        "cv_scores": cv_scores,
        "resultados": resultados,
        "fecha_base": fecha_base,
        "periodo_predicho": periodo_predicho,
        "pred_final": pred_final,
        "prob_final": prob_final,
        "frecuencia": frecuencia,
    }


def guardar_historico(fecha_base, periodo_predicho, prediccion, probabilidad):
    nueva = pd.DataFrame({
        "fecha_ejecucion": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "fecha_base_informacion": [fecha_base],
        "periodo_predicho": [periodo_predicho],
        "prediccion": ["Sube" if prediccion == 1 else "No sube"],
        "probabilidad_sube": [probabilidad]
    })

    if os.path.exists(HISTORICO_PATH):
        historico = pd.read_csv(HISTORICO_PATH)
        historico = pd.concat([historico, nueva], ignore_index=True)
    else:
        historico = nueva

    historico = historico.drop_duplicates(subset=["fecha_base_informacion", "periodo_predicho"], keep="last")
    historico.to_csv(HISTORICO_PATH, index=False)
    return historico


def _limpiar_numero_esp(x):
    """Convierte números con coma decimal y/o separadores extraños a float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # Si hay coma y punto, asumimos punto como miles y coma como decimal.
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")


def normalizar_producto(x):
    x = str(x).upper().strip()
    if "GAS" in x and ("91" in x or "REG" in x):
        return "gasolina_regular"
    if "GAS" in x and ("95" in x or "SUPER" in x or "SÚPER" in x):
        return "gasolina_super"
    if "DIESEL" in x or "DIÉSEL" in x:
        return "diesel"
    if "JET" in x or "AV TUR" in x:
        return "jet"
    if "LPG" in x or "GLP" in x or "PROPANO" in x or "BUTANO" in x:
        return "glp"
    if "ASFALTO" in x:
        return "asfalto"
    if "AV-GAS" in x or "AVGAS" in x:
        return "avgas"
    if "BUNKER" in x:
        return "bunker"
    return "otro"


@st.cache_data(show_spinner=False)
def cargar_compras_recope(path_o_buffer=None) -> pd.DataFrame:
    """Carga y prepara las compras Recope 2016-2026."""
    if path_o_buffer is None:
        if not os.path.exists(RECOPE_PATH):
            return pd.DataFrame()
        path_o_buffer = RECOPE_PATH

    # Los archivos históricos del usuario vienen normalmente en Latin-1 y separados por punto y coma.
    try:
        df = pd.read_csv(path_o_buffer, sep=";", encoding="latin1")
    except Exception:
        df = pd.read_csv(path_o_buffer, sep=",", encoding="utf-8")

    # Normaliza nombres de columnas para evitar problemas con tildes, espacios y símbolos.
    renombrar = {
        "AÑO": "anio",
        "MES": "mes",
        "FECHA_BL": "fecha_bl",
        "PRODUCTO": "producto",
        "BARRILES": "barriles",
        "COSTO_FOB": "costo_fob",
        "COSTO_FLETE": "costo_flete",
        "COSTO_SEGURO": "costo_seguro",
        "COSTO_CIF": "costo_cif",
        "COSTO_CFR": "costo_cfr",
        "PROVEEDOR": "proveedor",
        "NÚMERO_EMBARQUE": "numero_embarque",
        "NÚMERO_CONTRATO": "numero_contrato",
    }
    df = df.rename(columns={c: renombrar.get(c, c) for c in df.columns})

    for col in ["barriles", "costo_fob", "costo_flete", "costo_seguro", "costo_cif", "costo_cfr"]:
        if col in df.columns:
            df[col] = df[col].apply(_limpiar_numero_esp)

    df["fecha_bl"] = pd.to_datetime(df.get("fecha_bl"), errors="coerce", dayfirst=True)
    df["fecha_operacion"] = df["fecha_bl"]
    df["producto_normalizado"] = df.get("producto", "").apply(normalizar_producto)

    # Precios unitarios por barril. Se calculan desde los montos totales para mantener consistencia.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["precio_fob_bbl"] = df["costo_fob"] / df["barriles"]
        df["precio_cif_bbl"] = df["costo_cif"] / df["barriles"]
        df["precio_cfr_bbl"] = df["costo_cfr"] / df["barriles"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["fecha_operacion", "producto_normalizado", "barriles", "precio_cfr_bbl"])
    df = df[df["barriles"] > 0]

    df["anio"] = df["fecha_operacion"].dt.year
    df["mes"] = df["fecha_operacion"].dt.month
    df["semana_iso"] = df["fecha_operacion"].dt.isocalendar().week.astype(int)

    return df.sort_values("fecha_operacion").reset_index(drop=True)


def agregar_compras_periodo(df_recope: pd.DataFrame, producto: str, frecuencia="W-MON") -> pd.DataFrame:
    """Agrupa compras por producto y periodo con promedio ponderado por barriles."""
    base = df_recope[df_recope["producto_normalizado"] == producto].copy()
    if base.empty:
        return pd.DataFrame()

    base["periodo"] = base["fecha_operacion"].dt.to_period(frecuencia).dt.start_time

    def wavg(g, col):
        return np.average(g[col], weights=g["barriles"])

    out = base.groupby("periodo", as_index=False).apply(
        lambda g: pd.Series({
            "barriles": g["barriles"].sum(),
            "precio_real_fob_bbl": wavg(g, "precio_fob_bbl"),
            "precio_real_cif_bbl": wavg(g, "precio_cif_bbl"),
            "precio_real_cfr_bbl": wavg(g, "precio_cfr_bbl"),
            "compras": len(g),
        })
    ).reset_index(drop=True)
    out = out.rename(columns={"periodo": "fecha"})
    return out.sort_values("fecha")


def serie_eia_para_producto(df_eia: pd.DataFrame, producto: str) -> tuple[pd.DataFrame, str]:
    """Devuelve una serie EIA en USD/bbl para comparar como proxy de mercado."""
    eia = df_eia[["fecha", "gasolina_regular", "diesel", "wti"]].copy().sort_values("fecha")
    if producto in ["gasolina_regular", "gasolina_super"]:
        eia["precio_eia_bbl"] = eia["gasolina_regular"] * 42
        etiqueta = "Gasolina regular EIA × 42"
    elif producto == "diesel":
        eia["precio_eia_bbl"] = eia["diesel"] * 42
        etiqueta = "Diésel EIA × 42"
    else:
        eia["precio_eia_bbl"] = eia["wti"]
        etiqueta = "WTI EIA como proxy"
    return eia[["fecha", "precio_eia_bbl"]].dropna(), etiqueta


def construir_backtesting_recope(df_eia: pd.DataFrame, df_recope: pd.DataFrame, producto: str, objetivo: str):
    compras = agregar_compras_periodo(df_recope, producto)
    if compras.empty:
        return pd.DataFrame(), "Sin compras para el producto seleccionado"

    eia, etiqueta = serie_eia_para_producto(df_eia, producto)
    comp = pd.merge_asof(
        compras.sort_values("fecha"),
        eia.sort_values("fecha"),
        on="fecha",
        direction="nearest",
        tolerance=pd.Timedelta(days=10)
    ).dropna(subset=["precio_eia_bbl"])

    if comp.empty:
        return comp, etiqueta

    comp["precio_real_bbl"] = comp[objetivo]
    comp["diferencia_usd_bbl"] = comp["precio_real_bbl"] - comp["precio_eia_bbl"]
    comp["error_abs_usd_bbl"] = comp["diferencia_usd_bbl"].abs()
    comp["error_pct"] = np.where(comp["precio_real_bbl"] != 0, comp["error_abs_usd_bbl"] / comp["precio_real_bbl"] * 100, np.nan)
    comp["cercania_pct"] = (100 - comp["error_pct"]).clip(lower=0)
    comp["sesgo"] = np.where(comp["diferencia_usd_bbl"] > 0, "Recope > EIA", "Recope <= EIA")
    return comp.sort_values("fecha"), etiqueta


def evaluar_cercania(cercania):
    if pd.isna(cercania):
        return "Sin dato"
    if cercania >= 95:
        return "Muy cercana"
    if cercania >= 90:
        return "Aceptable"
    if cercania >= 80:
        return "Moderada"
    return "Lejana"



def formato_objetivo(nombre: str) -> str:
    return {
        "precio_real_cfr_bbl": "CFR por barril",
        "precio_real_cif_bbl": "CIF por barril",
        "precio_real_fob_bbl": "FOB por barril",
    }.get(nombre, nombre)


def texto_interpretacion_prediccion(pred_texto: str, prob: float, fecha_base, periodo_predicho) -> str:
    tendencia = "un incremento" if pred_texto == "Sube" else "que no se presentaría un incremento"
    return (
        f"Con la información EIA disponible al {pd.to_datetime(fecha_base).date()}, "
        f"el modelo estima {tendencia} para el periodo {pd.to_datetime(periodo_predicho).date()}. "
        f"La probabilidad estimada de subida es {prob:.2%}. "
        "El umbral de decisión utilizado es 50%; por encima de ese nivel se clasifica como 'Sube' y por debajo como 'No sube'."
    )


def texto_interpretacion_recope(producto: str, objetivo: str, n: int) -> str:
    return (
        f"Esta sección resume las compras reales de importación disponibles para el producto seleccionado ({producto}). "
        f"El precio real utilizado para el contraste es {formato_objetivo(objetivo)}. "
        f"La base contiene {n:,} registros limpios para análisis, luego de convertir fechas, montos y volúmenes a formato numérico."
    )


def texto_interpretacion_comparacion(ultimo: pd.Series, etiqueta_eia: str) -> str:
    cercania = float(ultimo["cercania_pct"])
    dif = float(ultimo["diferencia_usd_bbl"])
    sentido = "mayor" if dif > 0 else "menor"
    return (
        f"Para la última observación cruzada, el precio real Recope fue {sentido} que el proxy internacional por "
        f"{abs(dif):,.2f} USD/bbl. La cercanía calculada fue de {cercania:,.2f}%. "
        f"El proxy usado es {etiqueta_eia}. Esta comparación debe leerse como una medida de aproximación de mercado, "
        "no como una liquidación tarifaria ni como una validación contractual de una compra específica."
    )


def texto_analisis_tecnico(backtest: pd.DataFrame) -> str:
    if backtest.empty:
        return "No hay observaciones suficientes para construir una interpretación técnica del contraste."
    cerc = backtest["cercania_pct"].mean()
    mape = backtest["error_pct"].mean()
    mae = backtest["error_abs_usd_bbl"].mean()
    if cerc >= 90:
        nivel = "alta"
    elif cerc >= 75:
        nivel = "moderada"
    else:
        nivel = "limitada"
    return (
        f"El contraste histórico muestra una cercanía promedio de {cerc:,.2f}%, con un MAE de {mae:,.2f} USD/bbl "
        f"y un MAPE de {mape:,.2f}%. En términos prácticos, la capacidad del proxy para aproximar las compras reales es {nivel}. "
        "Las diferencias pueden explicarse por rezagos temporales entre mercado y compra, condiciones comerciales, fletes, seguros, "
        "márgenes de intermediación, composición del producto, puntos de entrega y otros componentes que no necesariamente están contenidos en la serie EIA."
    )




def calcular_correlacion(backtest: pd.DataFrame) -> float:
    """Calcula la correlación entre el proxy EIA y el precio real Recope."""
    if backtest.empty or len(backtest) < 3:
        return np.nan
    return backtest[["precio_eia_bbl", "precio_real_bbl"]].corr().iloc[0, 1]


def clasificar_correlacion(corr: float) -> str:
    if pd.isna(corr):
        return "Sin dato"
    abs_corr = abs(corr)
    if abs_corr >= 0.80:
        return "Alta"
    if abs_corr >= 0.60:
        return "Moderada"
    if abs_corr >= 0.40:
        return "Baja a moderada"
    return "Baja"


def texto_transmision_precios(backtest: pd.DataFrame, etiqueta_eia: str) -> str:
    corr = calcular_correlacion(backtest)
    nivel = clasificar_correlacion(corr)
    if pd.isna(corr):
        return "No hay observaciones suficientes para estimar correlación entre el proxy EIA y las compras reales."
    return (
        f"La correlación histórica entre el proxy internacional ({etiqueta_eia}) y el precio real de compra Recope "
        f"es de {corr:,.2f}, clasificada como {nivel.lower()}. Esto sugiere que el componente externo tiende a moverse "
        "en la misma dirección general que las compras reales, aunque no necesariamente con igual magnitud. En términos económicos, "
        "se observa una transmisión parcial de precios internacionales: las compras reales siguen la tendencia del mercado externo, "
        "pero con posibles rezagos, amortiguación logística, condiciones contractuales y diferencias propias del costo de importación."
    )


def mostrar_glosario_metricas():
    with st.expander("Glosario técnico de métricas"):
        st.markdown(
            """
            **MAE (Mean Absolute Error):** promedio absoluto de las diferencias entre el valor estimado y el valor real. En esta app se expresa en USD/bbl. Mientras menor sea, más cercana está la aproximación.

            **MAPE (Mean Absolute Percentage Error):** error porcentual absoluto promedio. Permite leer el error en términos relativos. Un MAPE de 10% indica que, en promedio, la desviación fue cercana al 10% del precio real.

            **Cercanía:** indicador comunicativo calculado como `100 - error porcentual absoluto`. No sustituye el análisis estadístico, pero ayuda a explicar proximidad entre proxy y dato real.

            **Accuracy:** porcentaje de aciertos del modelo de clasificación al estimar si el precio sube o no sube. Aplica a dirección, no al nivel exacto del precio.

            **Proxy EIA:** serie internacional aproximada usada como referencia de mercado. No representa necesariamente el precio contractual exacto de una compra de Recope.

            **Correlación:** mide el grado en que dos series se mueven juntas. Valores cercanos a 1 indican movimiento conjunto positivo; valores cercanos a 0 indican baja relación lineal.
            """
        )


def mostrar_definiciones_precios():
    with st.expander("Definiciones técnicas de precios de importación"):
        st.markdown(
            """
            **FOB (Free on Board):** precio del producto puesto a bordo en el puerto de origen. No incluye flete ni seguro internacional.

            **CIF (Cost, Insurance and Freight):** costo del producto más seguro y flete hasta el puerto de destino.

            **CFR (Cost and Freight):** costo del producto más flete internacional. En esta app se usa como referencia principal porque aproxima el costo de adquisición antes de otros componentes específicos.

            **USD/bbl:** dólares por barril. Permite comparar productos y compras de distintos volúmenes en una misma unidad.
            """
        )


def mostrar_limitaciones():
    st.markdown(
        """
        ### Limitaciones metodológicas

        - El proxy EIA no es el precio contractual exacto de las compras de Recope.
        - La comparación busca evaluar cercanía y transmisión de tendencia, no liquidar ni auditar una compra específica.
        - Puede haber diferencias por rezagos temporales, fletes, seguros, márgenes de intermediación, calidad del producto, puerto de origen, punto de entrega y condiciones contractuales.
        - Una correlación alta no implica causalidad perfecta; indica que ambas series tienden a moverse juntas en el periodo analizado.
        - Los eventos y noticias ayudan a contextualizar movimientos de mercado, pero no prueban causalidad por sí solos.
        """
    )


@st.cache_data(show_spinner=False)
def cargar_eventos_combustibles(path_o_buffer=None) -> pd.DataFrame:
    """Carga eventos de mercado. Funciona con archivo incluido o CSV cargado por el usuario."""
    if path_o_buffer is None:
        if not os.path.exists(EVENTOS_PATH):
            return pd.DataFrame()
        path_o_buffer = EVENTOS_PATH

    try:
        ev = pd.read_csv(path_o_buffer, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            ev = pd.read_csv(path_o_buffer, sep=",", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception:
            ev = pd.read_csv(path_o_buffer, sep=";", encoding="latin1", on_bad_lines="skip")

    ev.columns = [str(c).strip().lower() for c in ev.columns]
    if "fecha" not in ev.columns:
        return pd.DataFrame()
    ev["fecha"] = pd.to_datetime(ev["fecha"], errors="coerce", dayfirst=True)
    for col in ["categoria", "evento", "canal", "impacto_esperado", "comentario", "fuente"]:
        if col not in ev.columns:
            ev[col] = ""
    if "intensidad" not in ev.columns:
        ev["intensidad"] = 1
    ev["intensidad"] = pd.to_numeric(ev["intensidad"], errors="coerce").fillna(1).clip(1, 5)
    ev = ev.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
    return ev


def clasificar_canal_evento(texto: str) -> str:
    t = str(texto).lower()
    if any(k in t for k in ["opec", "production cut", "output", "supply", "saudi"]):
        return "oferta"
    if any(k in t for k in ["red sea", "shipping", "tanker", "suez", "freight", "vessel"]):
        return "logística"
    if any(k in t for k in ["iran", "israel", "russia", "ukraine", "war", "attack", "sanction"]):
        return "geopolítica"
    if any(k in t for k in ["refinery", "refining", "outage", "shutdown"]):
        return "refinación"
    if any(k in t for k in ["demand", "consumption", "china", "recession"]):
        return "demanda"
    return "mercado"


def estimar_impacto_evento(texto: str) -> str:
    t = str(texto).lower()
    alcista = ["cut", "attack", "sanction", "war", "disruption", "outage", "shutdown", "red sea", "risk"]
    bajista = ["increase output", "raise production", "demand falls", "weak demand", "glut", "surplus"]
    if any(k in t for k in alcista):
        return "alcista"
    if any(k in t for k in bajista):
        return "bajista"
    return "por clasificar"


def consultar_gdelt_noticias(query: str, dias: int = 30, max_records: int = 50) -> pd.DataFrame:
    """Consulta noticias recientes en GDELT DOC 2.0 y las transforma al formato de eventos de la app.

    GDELT no requiere API key. La consulta usa el modo ArtList y formato JSON.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "DateDesc",
        "timespan": f"{int(dias)}d",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    articulos = data.get("articles", [])

    filas = []
    for art in articulos:
        titulo = art.get("title", "")
        url_art = art.get("url", "")
        dominio = art.get("domain", "")
        fuente_pais = art.get("sourcecountry", "")
        fecha_raw = art.get("seendate", "") or art.get("datetime", "")

        fecha = pd.to_datetime(str(fecha_raw)[:14], format="%Y%m%d%H%M%S", errors="coerce")
        if pd.isna(fecha):
            fecha = pd.to_datetime(str(fecha_raw)[:8], format="%Y%m%d", errors="coerce")
        if pd.isna(fecha):
            continue

        texto = f"{titulo} {dominio} {url_art}"
        canal = clasificar_canal_evento(texto)
        impacto = estimar_impacto_evento(texto)

        filas.append({
            "fecha": fecha.normalize(),
            "categoria": "Noticia GDELT",
            "evento": titulo,
            "canal": canal,
            "impacto_esperado": impacto,
            "intensidad": 2,
            "comentario": f"Noticia reciente detectada por GDELT. País fuente: {fuente_pais}.",
            "fuente": url_art or dominio,
        })

    if not filas:
        return pd.DataFrame(columns=["fecha", "categoria", "evento", "canal", "impacto_esperado", "intensidad", "comentario", "fuente"])

    out = pd.DataFrame(filas)
    out = out.drop_duplicates(subset=["fecha", "evento", "fuente"])
    return out.sort_values("fecha", ascending=False).reset_index(drop=True)


def unir_eventos_base_y_noticias(eventos_base: pd.DataFrame, noticias: pd.DataFrame) -> pd.DataFrame:
    columnas = ["fecha", "categoria", "evento", "canal", "impacto_esperado", "intensidad", "comentario", "fuente"]

    base = eventos_base.copy() if eventos_base is not None else pd.DataFrame(columns=columnas)
    news = noticias.copy() if noticias is not None else pd.DataFrame(columns=columnas)

    for df_tmp in [base, news]:
        for col in columnas:
            if col not in df_tmp.columns:
                df_tmp[col] = ""

    combinado = pd.concat([base[columnas], news[columnas]], ignore_index=True)
    combinado["fecha"] = pd.to_datetime(combinado["fecha"], errors="coerce")
    combinado["evento"] = combinado["evento"].astype(str).str.strip()
    combinado["fuente"] = combinado["fuente"].astype(str).str.strip()
    combinado["intensidad"] = pd.to_numeric(combinado["intensidad"], errors="coerce").fillna(1).clip(1, 5)

    combinado = combinado.dropna(subset=["fecha"])
    combinado = combinado[combinado["evento"] != ""]
    combinado = combinado.drop_duplicates(subset=["fecha", "evento", "fuente"], keep="last")
    combinado = combinado.sort_values("fecha", ascending=False).reset_index(drop=True)

    return combinado[columnas]


def csv_eventos_descarga(df_eventos: pd.DataFrame) -> bytes:
    salida = df_eventos.copy()
    if "fecha" in salida.columns:
        salida["fecha"] = pd.to_datetime(salida["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
    return salida.to_csv(index=False, sep=";").encode("utf-8-sig")


def filtrar_eventos_periodo(eventos: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    if eventos.empty or backtest.empty:
        return pd.DataFrame()
    fecha_min = backtest["fecha"].min() - pd.Timedelta(days=30)
    fecha_max = backtest["fecha"].max() + pd.Timedelta(days=30)
    return eventos[(eventos["fecha"] >= fecha_min) & (eventos["fecha"] <= fecha_max)].copy()


def score_eventos(eventos: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Construye un score mensual sencillo por intensidad de eventos."""
    ev = filtrar_eventos_periodo(eventos, backtest)
    if ev.empty:
        return pd.DataFrame()
    ev["mes_evento"] = ev["fecha"].dt.to_period("M").dt.start_time
    out = ev.groupby("mes_evento", as_index=False).agg(
        eventos=("evento", "count"),
        intensidad_total=("intensidad", "sum"),
        intensidad_promedio=("intensidad", "mean")
    )
    out["nivel_riesgo"] = pd.cut(
        out["intensidad_total"],
        bins=[-1, 2, 5, 999],
        labels=["Bajo", "Medio", "Alto"]
    ).astype(str)
    return out.rename(columns={"mes_evento": "fecha"})


def texto_eventos_interpretacion(eventos_filtrados: pd.DataFrame) -> str:
    if eventos_filtrados.empty:
        return (
            "No hay eventos registrados para el periodo visible. Puede cargarse un CSV de eventos en la barra lateral "
            "para contextualizar shocks geopolíticos, logísticos, sociales o de oferta/demanda."
        )
    total = len(eventos_filtrados)
    cats = eventos_filtrados["categoria"].astype(str).str.strip().replace("", "Sin categoría").value_counts().head(3)
    cats_txt = ", ".join([f"{k} ({v})" for k, v in cats.items()])
    return (
        f"Para el periodo analizado se identifican {total} eventos de contexto. Las categorías más frecuentes son: {cats_txt}. "
        "Estos eventos no explican por sí solos el precio, pero ayudan a interpretar aumentos de volatilidad, cambios de tendencia, "
        "riesgos de oferta, disrupciones logísticas o variaciones de demanda internacional."
    )


def agregar_eventos_a_figura(fig, eventos: pd.DataFrame, max_eventos: int = 20):
    """Añade marcadores verticales a una figura Plotly sin saturarla.

    Nota técnica:
    En algunas versiones de Plotly/Streamlit Cloud, `add_vline` con fechas
    tipo pandas Timestamp y `annotation_text` genera TypeError porque Plotly
    intenta promediar fechas internamente. Para evitarlo, convertimos la fecha
    a texto ISO y separamos la línea vertical de la anotación.
    """
    if eventos is None or eventos.empty:
        return fig

    ev = eventos.copy()
    ev["fecha"] = pd.to_datetime(ev["fecha"], errors="coerce")
    ev = ev.dropna(subset=["fecha"])
    if ev.empty:
        return fig

    sort_cols = [c for c in ["intensidad", "fecha"] if c in ev.columns]
    if "intensidad" in ev.columns:
        ev = ev.sort_values(["intensidad", "fecha"], ascending=[False, True]).head(max_eventos)
    else:
        ev = ev.sort_values("fecha").head(max_eventos)

    for _, row in ev.iterrows():
        x_val = row["fecha"].strftime("%Y-%m-%d")
        label = str(row.get("categoria", "Evento"))[:18]

        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=0,
            y1=1,
            line=dict(width=1, dash="dot"),
        )

        fig.add_annotation(
            x=x_val,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
            textangle=-90,
            font=dict(size=10),
        )

    return fig

def pie_autor():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#6b7280; font-size:14px; line-height:1.6;'>
        Modelo experimental de análisis y predicción de precios de combustibles.<br>
        Datos públicos EIA y contraste con compras reales de importación.<br><br>
        <b>Hecho por: Lic. Juan Carlos Mena</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Opciones")
    st.write("Presioná el botón para descargar datos EIA, entrenar el modelo y generar la predicción.")
    ejecutar = st.button("Actualizar datos y entrenar modelo", type="primary")
    st.divider()
    st.subheader("Compras Recope")
    archivo_recope = st.file_uploader("CSV compras Recope", type=["csv"], help="Opcional. Si no se carga archivo, la app intenta usar 2016-2026_abril.csv en la carpeta del proyecto.")
    archivo_eventos = st.file_uploader("CSV eventos de mercado", type=["csv"], help="Opcional. Puede cargar aquí el CSV actualizado descargado desde el tab Eventos y noticias. Columnas: fecha, categoria, evento, canal, impacto_esperado, intensidad, comentario, fuente.")
    producto_recope = st.selectbox(
        "Producto para contraste",
        ["gasolina_regular", "gasolina_super", "diesel", "jet", "glp", "asfalto", "avgas", "bunker"],
        index=1,
    )
    objetivo_recope = st.selectbox(
        "Precio real Recope",
        ["precio_real_cfr_bbl", "precio_real_cif_bbl", "precio_real_fob_bbl"],
        index=0,
        help="CFR se usa por defecto porque aproxima FOB + flete. CIF y FOB quedan disponibles para análisis."
    )
    st.divider()
    st.caption("Fuente EIA: archivos históricos públicos. Fuente Recope: CSV suministrado por el usuario.")

if ejecutar:
    with st.spinner("Descargando datos EIA y entrenando modelo..."):
        df = cargar_datos()
        salida = preparar_modelo(df)
        historico = guardar_historico(salida["fecha_base"], salida["periodo_predicho"], salida["pred_final"], salida["prob_final"])

    pred_texto = "Sube" if salida["pred_final"] == 1 else "No sube"

    try:
        df_recope = cargar_compras_recope(archivo_recope if archivo_recope is not None else None)
    except Exception as exc:
        df_recope = pd.DataFrame()
        error_recope = str(exc)
    else:
        error_recope = None

    if not df_recope.empty:
        backtest, etiqueta_eia = construir_backtesting_recope(df, df_recope, producto_recope, objetivo_recope)
    else:
        backtest, etiqueta_eia = pd.DataFrame(), "Sin proxy disponible"

    try:
        df_eventos = cargar_eventos_combustibles(archivo_eventos if archivo_eventos is not None else None)
    except Exception as exc:
        df_eventos = pd.DataFrame()
        error_eventos = str(exc)
    else:
        error_eventos = None

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Predicción EIA",
        "2. Compras Recope",
        "3. Comparación",
        "4. Análisis técnico",
        "5. Eventos y noticias",
        "6. Histórico y detalles",
    ])

    with tab1:
        st.header("Predicción con datos EIA")
        st.caption("Esta sección muestra la predicción del siguiente periodo utilizando datos históricos públicos de EIA.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Última fecha EIA", str(salida["fecha_base"].date()))
        col2.metric("Periodo predicho", str(salida["periodo_predicho"].date()))
        col3.metric("Predicción", pred_texto)
        col4.metric("Probabilidad de subida", f"{salida['prob_final']:.2%}")

        st.info(texto_interpretacion_prediccion(pred_texto, salida["prob_final"], salida["fecha_base"], salida["periodo_predicho"]))

        st.subheader("Evaluación resumida del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy test temporal", f"{salida['accuracy']:.2%}")
        c2.metric("Promedio validación cruzada", f"{salida['cv_scores'].mean():.2%}")
        c3.metric("Frecuencia detectada", str(salida["frecuencia"]))

        st.markdown(
            """
            **Cómo leer esta sección:**  
            El modelo clasifica si la referencia internacional sube o no sube en el siguiente periodo semanal. 
            La probabilidad de subida no es un precio, sino la confianza relativa del clasificador respecto a la clase 'Sube'.
            """
        )

    with tab2:
        st.header("Compras reales de importación Recope")
        st.caption("Esta sección revisa la información real de compras disponible en el CSV suministrado.")
        if error_recope:
            st.error(f"No fue posible cargar el CSV de compras Recope: {error_recope}")
        elif df_recope.empty:
            st.warning("No se encontró un CSV de compras Recope. Cargá el archivo en la barra lateral o colocá 2016-2026_abril.csv en la carpeta del proyecto.")
        else:
            st.info(texto_interpretacion_recope(producto_recope, objetivo_recope, len(df_recope)))
            base_prod = df_recope[df_recope["producto_normalizado"] == producto_recope].copy()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registros del producto", f"{len(base_prod):,.0f}")
            c2.metric("Barriles acumulados", f"{base_prod['barriles'].sum():,.0f}")
            c3.metric("Primera compra", str(base_prod["fecha_operacion"].min().date()) if not base_prod.empty else "-")
            c4.metric("Última compra", str(base_prod["fecha_operacion"].max().date()) if not base_prod.empty else "-")

            if not base_prod.empty:
                serie = agregar_compras_periodo(df_recope, producto_recope)
                precio_col = objetivo_recope
                fig_compra = px.line(
                    serie,
                    x="fecha",
                    y=precio_col,
                    title=f"Precio real Recope - {producto_recope} ({formato_objetivo(objetivo_recope)})",
                    labels={precio_col: "USD/bbl", "fecha": "Fecha"},
                )
                st.plotly_chart(fig_compra, use_container_width=True)
                with st.expander("Ver muestra de compras normalizadas"):
                    cols = [c for c in ["fecha_operacion", "producto", "producto_normalizado", "barriles", "precio_fob_bbl", "precio_cif_bbl", "precio_cfr_bbl", "proveedor"] if c in base_prod.columns]
                    st.dataframe(base_prod[cols].sort_values("fecha_operacion", ascending=False).head(100), use_container_width=True)

    with tab3:
        st.header("Comparación entre proxy EIA y compras Recope")
        st.caption("Esta sección mide qué tan cerca está el proxy de mercado EIA del precio real de importación observado.")
        if backtest.empty:
            st.warning("No hay datos suficientes para cruzar el producto seleccionado contra la serie EIA.")
        else:
            ultimo = backtest.iloc[-1]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Última compra cruzada", str(pd.to_datetime(ultimo["fecha"]).date()))
            c2.metric("Proxy EIA", f"{ultimo['precio_eia_bbl']:,.2f} USD/bbl")
            c3.metric("Precio real Recope", f"{ultimo['precio_real_bbl']:,.2f} USD/bbl")
            c4.metric("Diferencia", f"{ultimo['diferencia_usd_bbl']:,.2f} USD/bbl")
            c5.metric("Cercanía", f"{ultimo['cercania_pct']:,.2f}%")

            evaluacion = evaluar_cercania(ultimo["cercania_pct"])
            mensaje_eval = f"Evaluación de la última compra: {evaluacion}"
            if ultimo["cercania_pct"] >= 95:
                st.success(mensaje_eval)
            elif ultimo["cercania_pct"] >= 90:
                st.warning(mensaje_eval)
            else:
                st.error(mensaje_eval)

            st.info(texto_interpretacion_comparacion(ultimo, etiqueta_eia))

            graf = backtest[["fecha", "precio_eia_bbl", "precio_real_bbl"]].melt(
                id_vars="fecha",
                value_vars=["precio_eia_bbl", "precio_real_bbl"],
                var_name="serie",
                value_name="USD/bbl"
            )
            fig_recope = px.line(
                graf,
                x="fecha",
                y="USD/bbl",
                color="serie",
                title=f"Proxy EIA vs compra real Recope - {producto_recope}"
            )
            st.plotly_chart(fig_recope, use_container_width=True)

            st.markdown(
                """
                **Nota de lectura:**  
                La cercanía se calcula como `100 - error porcentual absoluto`. 
                Un valor alto indica que el proxy y el precio real están próximos; un valor bajo indica una brecha importante que requiere análisis de composición, logística o temporalidad.
                """
            )

    with tab4:
        st.header("Análisis técnico e interpretación")
        st.caption("Esta sección transforma las métricas en una lectura técnica para el usuario final.")
        if backtest.empty:
            st.warning("No hay observaciones suficientes para el análisis técnico del contraste Recope.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cercanía histórica promedio", f"{backtest['cercania_pct'].mean():,.2f}%")
            m2.metric("MAE", f"{backtest['error_abs_usd_bbl'].mean():,.2f} USD/bbl")
            m3.metric("MAPE", f"{backtest['error_pct'].mean():,.2f}%")
            m4.metric("Observaciones cruzadas", f"{len(backtest):,.0f}")

            st.info(texto_analisis_tecnico(backtest))
            mostrar_glosario_metricas()

            st.markdown(
                """
                ### Hallazgos técnicos observados

                - El componente internacional presenta una relación observable con las compras reales de importación.
                - Las compras reales pueden mostrar menor volatilidad que el proxy internacional por efectos logísticos, contractuales y temporales.
                - Los periodos de alta volatilidad internacional suelen ampliar las brechas entre proxy y precio real.
                - La lectura conjunta de MAE, MAPE, cercanía y correlación permite valorar tanto el nivel del error como la dirección del movimiento.
                """
            )

            st.subheader("Distribución de errores")
            fig_err = px.histogram(
                backtest,
                x="diferencia_usd_bbl",
                nbins=40,
                title="Distribución de diferencias: Recope - proxy EIA",
                labels={"diferencia_usd_bbl": "Diferencia USD/bbl"},
            )
            st.plotly_chart(fig_err, use_container_width=True)

            st.subheader("Cercanía histórica")
            fig_cerc = px.line(
                backtest,
                x="fecha",
                y="cercania_pct",
                title="Cercanía porcentual por periodo",
                labels={"cercania_pct": "Cercanía (%)", "fecha": "Fecha"},
            )
            fig_cerc.add_hline(y=90, line_dash="dash")
            st.plotly_chart(fig_cerc, use_container_width=True)

    with tab5:
        st.header("Eventos y noticias relevantes")
        st.caption("Esta sección permite contextualizar movimientos de precios con eventos geopolíticos, logísticos, sociales o de oferta y demanda.")
        st.caption("La base curada incluida cubre eventos relevantes desde 2016 hasta abril de 2026. También puede cargar un CSV propio desde la barra lateral.")

        if error_eventos:
            st.error(f"No fue posible cargar el CSV de eventos: {error_eventos}")
        elif df_eventos.empty:
            st.warning("No se encontró archivo de eventos. La app incluye una plantilla CSV que podés ampliar o reemplazar desde la barra lateral.")
        else:
            eventos_filtrados = filtrar_eventos_periodo(df_eventos, backtest) if not backtest.empty else df_eventos.copy()
            st.info(texto_eventos_interpretacion(eventos_filtrados))

            c1, c2, c3 = st.columns(3)
            c1.metric("Eventos en periodo", f"{len(eventos_filtrados):,.0f}")
            c2.metric("Intensidad promedio", f"{eventos_filtrados['intensidad'].mean():,.2f}" if not eventos_filtrados.empty else "-")
            c3.metric("Máxima intensidad", f"{eventos_filtrados['intensidad'].max():,.0f}" if not eventos_filtrados.empty else "-")

            st.markdown("### CSV de eventos vigente")
            st.write(
                "Descargue la base de eventos que la app está usando actualmente. "
                "Este archivo puede editarse, enriquecerse y luego volver a subirse desde la barra lateral."
            )

            col_descarga_1, col_descarga_2 = st.columns([1, 2])
            with col_descarga_1:
                st.download_button(
                    "Descargar CSV actual de eventos",
                    csv_eventos_descarga(df_eventos),
                    file_name="eventos_geopoliticos_energia_actual.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("### Actualización semiautomática de noticias")
            st.write(
                "La app puede consultar noticias recientes en GDELT, unirlas con la base curada y generar un CSV actualizado. "
                "Como Streamlit Cloud no guarda cambios permanentes en el repositorio, descargue el CSV actualizado y vuelva a subirlo desde la barra lateral en futuras ejecuciones."
            )

            with st.expander("Consultar noticias recientes en GDELT y construir CSV actualizado", expanded=True):
                temas_default = [
                    "oil price",
                    "OPEC production cut",
                    "Red Sea shipping oil",
                    "Iran Israel oil",
                    "Russia oil sanctions",
                    "refinery outage diesel",
                ]
                consulta_gdelt = st.text_area(
                    "Consultas GDELT, una por línea",
                    value="\n".join(temas_default),
                    help="Use términos en inglés para mejorar la cobertura global de GDELT."
                )
                col_g1, col_g2 = st.columns(2)
                dias_gdelt = col_g1.slider("Días hacia atrás", min_value=1, max_value=90, value=30)
                max_gdelt = col_g2.slider("Máximo de noticias por consulta", min_value=5, max_value=100, value=25)

                if st.button("Actualizar noticias recientes con GDELT"):
                    consultas = [q.strip() for q in consulta_gdelt.splitlines() if q.strip()]
                    noticias_lista = []

                    with st.spinner("Consultando GDELT..."):
                        for q in consultas:
                            try:
                                tmp = consultar_gdelt_noticias(q, dias=dias_gdelt, max_records=max_gdelt)
                                if not tmp.empty:
                                    tmp["consulta_gdelt"] = q
                                    noticias_lista.append(tmp)
                            except Exception as exc:
                                st.warning(f"No fue posible consultar GDELT para: {q}. Detalle: {exc}")

                    if noticias_lista:
                        noticias_gdelt = pd.concat(noticias_lista, ignore_index=True)
                        noticias_gdelt = noticias_gdelt.drop_duplicates(subset=["fecha", "evento", "fuente"])
                        eventos_actualizados = unir_eventos_base_y_noticias(df_eventos, noticias_gdelt)

                        st.success(
                            f"Se incorporaron {len(noticias_gdelt):,.0f} noticias recientes. "
                            f"El CSV actualizado contiene {len(eventos_actualizados):,.0f} registros."
                        )

                        st.dataframe(noticias_gdelt.sort_values("fecha", ascending=False), use_container_width=True)

                        st.download_button(
                            "Descargar CSV actualizado con noticias GDELT",
                            csv_eventos_descarga(eventos_actualizados),
                            file_name="eventos_geopoliticos_energia_actualizado.csv",
                            mime="text/csv"
                        )

                        st.info(
                            "Para usar esta base actualizada en otra ejecución, descargue el CSV y súbalo en la barra lateral en el campo 'CSV eventos de mercado'."
                        )
                    else:
                        st.warning("GDELT no devolvió noticias para las consultas seleccionadas o no hubo conexión disponible.")

            if not backtest.empty:
                graf_eventos = backtest[["fecha", "precio_eia_bbl", "precio_real_bbl"]].melt(
                    id_vars="fecha",
                    value_vars=["precio_eia_bbl", "precio_real_bbl"],
                    var_name="serie",
                    value_name="USD/bbl"
                )
                fig_ev = px.line(
                    graf_eventos,
                    x="fecha",
                    y="USD/bbl",
                    color="serie",
                    title="Precios y eventos relevantes de mercado"
                )
                fig_ev = agregar_eventos_a_figura(fig_ev, eventos_filtrados)
                st.plotly_chart(fig_ev, use_container_width=True)

            score = score_eventos(df_eventos, backtest) if not backtest.empty else pd.DataFrame()
            if not score.empty:
                fig_score = px.bar(
                    score,
                    x="fecha",
                    y="intensidad_total",
                    color="nivel_riesgo",
                    title="Score mensual de eventos de mercado",
                    labels={"intensidad_total": "Intensidad total", "fecha": "Mes"}
                )
                st.plotly_chart(fig_score, use_container_width=True)

            st.markdown(
                """
                ### Cómo interpretar esta sección

                Los eventos no se tratan como una prueba causal automática. Su función es aportar contexto a movimientos de precios, especialmente en periodos de alta volatilidad. 
                Un evento puede afectar precios por distintos canales: oferta, demanda, logística, geopolítica, inventarios, refinación o expectativas del mercado.
                """
            )

            with st.expander("Ver eventos cargados"):
                st.dataframe(eventos_filtrados.sort_values("fecha", ascending=False), use_container_width=True)

            st.download_button(
                "Descargar plantilla/base de eventos CSV",
                csv_eventos_descarga(df_eventos),
                file_name="eventos_geopoliticos_energia_2016_2026.csv",
                mime="text/csv"
            )

    with tab6:
        st.header("Histórico, matriz y datos técnicos")
        st.caption("Esta sección conserva los detalles del modelo para auditoría, revisión y descarga.")

        st.subheader("Matriz de confusión")
        matriz_df = pd.DataFrame(salida["matriz"], index=["Real: No sube", "Real: Sube"], columns=["Pred: No sube", "Pred: Sube"])
        st.dataframe(matriz_df, use_container_width=True)

        st.subheader("Mejores hiperparámetros")
        st.json(salida["mejores_parametros"])

        st.subheader("Resultados del periodo de prueba")
        st.dataframe(salida["resultados"].sort_values("fecha", ascending=False), use_container_width=True)

        fig = px.line(salida["resultados"], x="fecha", y="probabilidad_sube", title="Probabilidad estimada de subida en el periodo de prueba")
        fig.add_hline(y=0.5, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Histórico de predicciones")
        st.dataframe(historico.sort_values("fecha_ejecucion", ascending=False), use_container_width=True)
        st.download_button(
            "Descargar histórico CSV",
            historico.to_csv(index=False),
            file_name="historico_predicciones_combustibles.csv",
            mime="text/csv"
        )

        if not backtest.empty:
            st.subheader("Detalle del contraste Recope")
            st.dataframe(backtest.sort_values("fecha", ascending=False), use_container_width=True)
            st.download_button(
                "Descargar contraste Recope CSV",
                backtest.to_csv(index=False),
                file_name=f"contraste_recope_{producto_recope}.csv",
                mime="text/csv"
            )

        st.subheader("Últimos datos EIA descargados")
        st.dataframe(df.tail(15).sort_values("fecha", ascending=False), use_container_width=True)

    pie_autor()
else:
    st.info("Presioná el botón de la izquierda para ejecutar el modelo.")
    st.markdown(
        """
        ### ¿Qué hace esta aplicación?
        Esta herramienta descarga datos públicos de EIA, entrena un modelo supervisado para clasificar la dirección del precio internacional de referencia y permite contrastar el resultado con compras reales de importación Recope.

        **Flujo de lectura recomendado:**
        1. Predicción EIA: resultado del modelo y probabilidad de subida.
        2. Compras Recope: comportamiento real de importaciones.
        3. Comparación: cercanía entre proxy internacional y compra real.
        4. Análisis técnico: interpretación de errores, brechas y desempeño.
        5. Eventos y noticias: consulta GDELT, descarga de CSV actualizado y carga posterior desde la barra lateral.
        6. Histórico: tablas y descargas para revisión.
        """
    )
    pie_autor()
