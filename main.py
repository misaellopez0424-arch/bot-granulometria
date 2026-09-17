from pathlib import Path
import re

src = Path("/mnt/data/main.py").read_text(encoding="utf-8")

# We'll produce a complete replacement based on the uploaded main.py,
# preserving its PDF/report functionality while adding JSON results and P80.
code = r'''import io
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


app = FastAPI(title="API de Granulometría Metalúrgica y Geotécnica")


# ============================================================
# 1. MODELOS DE ENTRADA
# ============================================================

class DatosMalla(BaseModel):
    sieve: str
    retained_weight_g: float


class PeticionAnalisis(BaseModel):
    report_title: Optional[str] = "LABORATORIO METALURGICO"
    sample_name: Optional[str] = "Muestra No Identificada"
    total_initial_weight: Optional[float] = None
    data: List[DatosMalla]


# ============================================================
# 2. ABERTURAS ESTÁNDAR
#    Unidad interna: mm
# ============================================================

ABERTURAS_MALLAS = {
    '3"': 75.0,
    '2"': 50.0,
    '1 1/2"': 37.5,
    '1"': 25.0,
    '3/4"': 19.0,
    '1/2"': 12.5,
    '3/8"': 9.5,
    'No. 4': 4.75,
    'No. 8': 2.36,
    'No. 10': 2.00,
    'No. 16': 1.18,
    'No. 30': 0.60,
    'No. 40': 0.425,
    'No. 50': 0.30,
    'No. 100': 0.15,
    'No. 200': 0.075,
    'No. 270': 0.053,
    'No. 325': 0.045,
    'No. 400': 0.038,
    'No. 500': 0.025,
    'No. 635': 0.020,
    'Pan': 0.001,
}


# ============================================================
# 3. NORMALIZACIÓN DE NOMBRES DE MALLA
# ============================================================

def normalizar_malla(sieve: str) -> str:
    """
    Convierte variantes comunes enviadas por IA/usuario a los
    nombres canónicos utilizados por ABERTURAS_MALLAS.
    """
    original = str(sieve).strip()

    # Normalización básica
    s = original.lower().strip()
    s = s.replace("mesh", "")
    s = s.replace("malla", "")
    s = s.replace("tamiz", "")
    s = s.replace("#", "")
    s = s.replace("n°", "")
    s = s.replace("nº", "")
    s = s.replace("no.", "no")
    s = s.replace("no ", "no")
    s = s.strip()

    # Fondo / pan
    if s in {"pan", "fondo", "bandeja", "bottom"}:
        return "Pan"

    # Mallas numéricas
    numeros = re.findall(r"\d+(?:\.\d+)?", s)
    if numeros:
        numero = numeros[0]
        try:
            n = int(float(numero))
        except ValueError:
            n = None

        if n is not None:
            candidato = f"No. {n}"
            if candidato in ABERTURAS_MALLAS:
                return candidato

    # Pulgadas y otros nombres exactos
    for clave in ABERTURAS_MALLAS:
        if s == clave.lower():
            return clave

    return original


# ============================================================
# 4. CÁLCULO DEL P80
# ============================================================

def calcular_p80(df: pd.DataFrame) -> Optional[float]:
    """
    Calcula P80 en micras mediante interpolación lineal en
    log10(tamaño), utilizando los puntos de % pasante.

    Se excluye Pan porque no representa un tamaño de abertura
    utilizable para interpolar el P80.
    """
    curva = df[df["Malla"] != "Pan"].copy()

    if curva.empty:
        return None

    # Ordenar por abertura creciente para localizar el cruce del 80%.
    curva = curva.sort_values("Abertura_mm", ascending=True)

    x = curva["Abertura_mm"].to_numpy(dtype=float)
    y = curva["Pct_Pasante"].to_numpy(dtype=float)

    # Eliminar valores no válidos
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return None

    # Si existe exactamente un 80%
    exactos = np.where(np.isclose(y, 80.0, atol=1e-12))[0]
    if len(exactos) > 0:
        return float(x[exactos[0]] * 1000.0)

    # Buscar el intervalo donde cruza el 80%.
    # Con x creciente, y también debe aumentar.
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        if (y1 <= 80.0 <= y2) or (y2 <= 80.0 <= y1):
            if np.isclose(y1, y2):
                return float(((x1 + x2) / 2.0) * 1000.0)

            # Interpolación lineal en log10 del tamaño:
            log_x1 = math.log10(x1)
            log_x2 = math.log10(x2)

            log_x80 = log_x1 + (
                (80.0 - y1) / (y2 - y1)
            ) * (log_x2 - log_x1)

            x80_mm = 10 ** log_x80
            return float(x80_mm * 1000.0)

    # Si todos los puntos están por encima o por debajo del 80%,
    # no se puede interpolar con la información disponible.
    return None


# ============================================================
# 5. ENDPOINT DE ANÁLISIS
# ============================================================

@app.post("/api/v1/granulometria")
async def generar_reporte_granulometrico(payload: PeticionAnalisis):
    try:
        if not payload.data:
            raise HTTPException(
                status_code=422,
                detail="No se recibieron datos granulométricos."
            )

        # ----------------------------------------------------
        # Convertir datos de entrada a DataFrame
        # ----------------------------------------------------
        registros = []

        for item in payload.data:
            malla = normalizar_malla(item.sieve)

            if item.retained_weight_g < 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"El peso retenido no puede ser negativo: {item.retained_weight_g} g."
                )

            abertura = ABERTURAS_MALLAS.get(malla, 0.001)

            registros.append({
                "Malla": malla,
                "Abertura_mm": abertura,
                "Retenido_g": float(item.retained_weight_g)
            })

        df = pd.DataFrame(registros)

        # ----------------------------------------------------
        # Orden granulométrico:
        # malla más gruesa → malla más fina → Pan
        # ----------------------------------------------------
        df["_orden"] = df["Abertura_mm"]
        df.loc[df["Malla"] == "Pan", "_orden"] = -1

        df = df.sort_values(
            "_orden",
            ascending=False
        ).drop(columns=["_orden"]).reset_index(drop=True)

        suma_retenido = float(df["Retenido_g"].sum())

        if suma_retenido <= 0:
            raise HTTPException(
                status_code=422,
                detail="La suma de los pesos retenidos debe ser mayor que cero."
            )

        # ----------------------------------------------------
        # Peso inicial
        # ----------------------------------------------------
        peso_inicial = payload.total_initial_weight

        if peso_inicial is not None and peso_inicial <= 0:
            raise HTTPException(
                status_code=422,
                detail="El peso inicial debe ser mayor que cero."
            )

        # Si no se proporciona peso inicial, se utiliza el peso
        # recuperado como referencia para la distribución.
        denominador = (
            float(peso_inicial)
            if peso_inicial is not None
            else suma_retenido
        )

        # ----------------------------------------------------
        # CÁLCULOS
        # ----------------------------------------------------

        df["Pct_Retenido"] = (
            df["Retenido_g"] / denominador
        ) * 100.0

        df["Cum_Retenido_g"] = df["Retenido_g"].cumsum()

        df["Cum_Pct_Retenido"] = df["Pct_Retenido"].cumsum()

        df["Pct_Pasante"] = (
            100.0 - df["Cum_Pct_Retenido"]
        )

        # Evitar pequeños negativos por errores de redondeo.
        df["Pct_Pasante"] = df["Pct_Pasante"].clip(lower=0, upper=100)

        # ----------------------------------------------------
        # ERROR / RECUPERACIÓN DE MASA
        # ----------------------------------------------------
        if peso_inicial is not None:
            recuperacion = (
                suma_retenido / float(peso_inicial)
            ) * 100.0

            error_masa = (
                (suma_retenido - float(peso_inicial))
                / float(peso_inicial)
            ) * 100.0
        else:
            recuperacion = 100.0
            error_masa = 0.0

        # ----------------------------------------------------
        # P80
        # ----------------------------------------------------
        p80_micras = calcular_p80(df)

        # ----------------------------------------------------
        # RESULTADO JSON PARA MAKE
        # ----------------------------------------------------
        resultados_mallas = []

        for _, row in df.iterrows():
            resultados_mallas.append({
                "sieve": row["Malla"],
                "opening_mm": round(float(row["Abertura_mm"]), 6),
                "opening_microns": round(float(row["Abertura_mm"]) * 1000.0, 3)
                    if row["Malla"] != "Pan" else None,
                "retained_weight_g": round(float(row["Retenido_g"]), 4),
                "retained_pct": round(float(row["Pct_Retenido"]), 4),
                "cumulative_retained_g": round(float(row["Cum_Retenido_g"]), 4),
                "cumulative_retained_pct": round(float(row["Cum_Pct_Retenido"]), 4),
                "passing_pct": round(float(row["Pct_Pasante"]), 4),
            })

        resultado = {
            "report_title": payload.report_title,
            "sample_name": payload.sample_name,
            "total_initial_weight_g": (
                round(float(peso_inicial), 4)
                if peso_inicial is not None else None
            ),
            "total_recovered_weight_g": round(suma_retenido, 4),
            "mass_recovery_pct": round(recuperacion, 4),
            "mass_error_pct": round(error_masa, 4),
            "p80_microns": round(p80_micras, 4)
                if p80_micras is not None else None,
            "p80_available": p80_micras is not None,
            "data": resultados_mallas
        }

        # ====================================================
        # GENERAR GRÁFICA
        # ====================================================

        df_grafica = df[df["Malla"] != "Pan"].copy()
        df_grafica = df_grafica.sort_values(
            "Abertura_mm",
            ascending=True
        )

        plt.figure(figsize=(7, 3.2))

        plt.plot(
            df_grafica["Abertura_mm"],
            df_grafica["Pct_Pasante"],
            marker="o",
            color="#1a5f7a",
            linewidth=2
        )

        plt.xscale("log")
        plt.xlim(100, 0.01)
        plt.ylim(0, 105)

        plt.title(
            "Curva de Distribución Granulométrica",
            fontsize=11,
            fontweight="bold"
        )

        plt.xlabel(
            "Abertura del Tamiz (mm) - Escala Logarítmica",
            fontsize=8
        )

        plt.ylabel(
            "Porcentaje Pasante (%)",
            fontsize=8
        )

        plt.grid(
            True,
            which="both",
            ls="--",
            color="#cccccc"
        )

        # Marcar P80 si existe
        if p80_micras is not None:
            p80_mm = p80_micras / 1000.0

            plt.axhline(
                80,
                linestyle="--",
                linewidth=1
            )

            plt.axvline(
                p80_mm,
                linestyle="--",
                linewidth=1
            )

        buf_imagen = io.BytesIO()

        plt.savefig(
            buf_imagen,
            format="png",
            bbox_inches="tight",
            dpi=150
        )

        buf_imagen.seek(0)
        plt.close()

        # ====================================================
        # GENERAR PDF
        # ====================================================

        buf_pdf = io.BytesIO()

        doc = SimpleDocTemplate(
            buf_pdf,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        estilos = getSampleStyleSheet()

        estilo_titulo = ParagraphStyle(
            "DocTitle",
            parent=estilos["Heading1"],
            fontSize=20,
            textColor=colors.HexColor("#1a5f7a"),
            spaceAfter=4,
            alignment=1
        )

        estilo_sub = ParagraphStyle(
            "DocSub",
            parent=estilos["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=10,
            alignment=1
        )

        estilo_meta = ParagraphStyle(
            "MetaText",
            parent=estilos["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6
        )

        story.append(
            Paragraph(
                (payload.report_title or "LABORATORIO METALURGICO").upper(),
                estilo_titulo
            )
        )

        story.append(
            Paragraph(
                "INFORME DE ENSAYO GRANULOMÉTRICO",
                estilo_sub
            )
        )

        story.append(Spacer(1, 5))

        story.append(
            Paragraph(
                f"<b>Identificación de la Muestra:</b> "
                f"{payload.sample_name}",
                estilo_meta
            )
        )

        story.append(
            Paragraph(
                f"<b>Peso Inicial Seco:</b> "
                f"{peso_inicial if peso_inicial is not None else 'N/A'} g | "
                f"<b>Peso Total Retenido:</b> "
                f"{suma_retenido:.2f} g",
                estilo_meta
            )
        )

        story.append(
            Paragraph(
                f"<b>Recuperación de Masa:</b> "
                f"{recuperacion:.2f}%",
                estilo_meta
            )
        )

        if p80_micras is not None:
            story.append(
                Paragraph(
                    f"<b>P80:</b> {p80_micras:.2f} µm",
                    estilo_meta
                )
            )
        else:
            story.append(
                Paragraph(
                    "<b>P80:</b> No disponible con los datos proporcionados",
                    estilo_meta
                )
            )

        color_estado = (
            "#27ae60"
            if abs(error_masa) <= 1.0
            else "#c0392b"
        )

        story.append(
            Paragraph(
                f"<b>Error de Masa:</b> "
                f"<font color='{color_estado}'>"
                f"{error_masa:.2f}%"
                f"</font> "
                f"(Límite tolerable: ±1%)",
                estilo_meta
            )
        )

        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # Tabla PDF
        # ----------------------------------------------------
        datos_tabla = [[
            "Tamiz",
            "Abertura (mm)",
            "Retenido (g)",
            "% Retenido",
            "% Ret. Acum.",
            "% Pasante"
        ]]

        for _, row in df.iterrows():
            datos_tabla.append([
                row["Malla"],
                (
                    f"{row['Abertura_mm']:.3f}"
                    if row["Malla"] != "Pan"
                    else "-"
                ),
                f"{row['Retenido_g']:.2f}",
                f"{row['Pct_Retenido']:.1f}%",
                f"{row['Cum_Pct_Retenido']:.1f}%",
                f"{row['Pct_Pasante']:.1f}%"
            ])

        tabla_pdf = Table(
            datos_tabla,
            colWidths=[70, 85, 80, 80, 100, 80]
        )

        tabla_pdf.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1a5f7a")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.whitesmoke
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    10
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, 0),
                    6
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#f8f9fa")
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#dddddd")
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    9
                ),
            ])
        )

        story.append(tabla_pdf)
        story.append(Spacer(1, 10))

        story.append(
            Image(
                buf_imagen,
                width=440,
                height=200
            )
        )

        doc.build(story)
        buf_pdf.seek(0)

        # ====================================================
        # RESPUESTA
        #
        # IMPORTANTE:
        # Por defecto devolvemos JSON para Make.
        #
        # Para obtener PDF:
        # POST /api/v1/granulometria/pdf
        # ====================================================

        return JSONResponse(content=resultado)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 6. ENDPOINT EXCLUSIVO PARA PDF
# ============================================================

@app.post("/api/v1/granulometria/pdf")
async def generar_pdf_granulometrico(payload: PeticionAnalisis):
    """
    Genera el mismo análisis y devuelve el PDF.
    Internamente reutiliza la lógica del endpoint principal.
    """

    try:
        # Repetimos el procesamiento para mantener este endpoint
        # independiente y sencillo de consumir.

        if not payload.data:
            raise HTTPException(
                status_code=422,
                detail="No se recibieron datos granulométricos."
            )

        registros = []

        for item in payload.data:
            malla = normalizar_malla(item.sieve)

            if item.retained_weight_g < 0:
                raise HTTPException(
                    status_code=422,
                    detail="El peso retenido no puede ser negativo."
                )

            abertura = ABERTURAS_MALLAS.get(malla, 0.001)

            registros.append({
                "Malla": malla,
                "Abertura_mm": abertura,
                "Retenido_g": float(item.retained_weight_g)
            })

        df = pd.DataFrame(registros)

        df["_orden"] = df["Abertura_mm"]
        df.loc[df["Malla"] == "Pan", "_orden"] = -1

        df = df.sort_values(
            "_orden",
            ascending=False
        ).drop(columns=["_orden"]).reset_index(drop=True)

        suma_retenido = float(df["Retenido_g"].sum())

        if suma_retenido <= 0:
            raise HTTPException(
                status_code=422,
                detail="La suma de los pesos retenidos debe ser mayor que cero."
            )

        peso_inicial = payload.total_initial_weight

        if peso_inicial is not None and peso_inicial <= 0:
            raise HTTPException(
                status_code=422,
                detail="El peso inicial debe ser mayor que cero."
            )

        denominador = (
            float(peso_inicial)
            if peso_inicial is not None
            else suma_retenido
        )

        df["Pct_Retenido"] = (
            df["Retenido_g"] / denominador
        ) * 100.0

        df["Cum_Retenido_g"] = df["Retenido_g"].cumsum()
        df["Cum_Pct_Retenido"] = df["Pct_Retenido"].cumsum()
        df["Pct_Pasante"] = (
            100.0 - df["Cum_Pct_Retenido"]
        ).clip(lower=0, upper=100)

        if peso_inicial is not None:
            recuperacion = (
                suma_retenido / float(peso_inicial)
            ) * 100.0

            error_masa = (
                (suma_retenido - float(peso_inicial))
                / float(peso_inicial)
            ) * 100.0
        else:
            recuperacion = 100.0
            error_masa = 0.0

        p80_micras = calcular_p80(df)

        # ----------------------------------------------------
        # Gráfica
        # ----------------------------------------------------
        df_grafica = df[df["Malla"] != "Pan"].copy()
        df_grafica = df_grafica.sort_values(
            "Abertura_mm",
            ascending=True
        )

        plt.figure(figsize=(7, 3.2))

        plt.plot(
            df_grafica["Abertura_mm"],
            df_grafica["Pct_Pasante"],
            marker="o",
            color="#1a5f7a",
            linewidth=2
        )

        plt.xscale("log")
        plt.xlim(100, 0.01)
        plt.ylim(0, 105)

        plt.title(
            "Curva de Distribución Granulométrica",
            fontsize=11,
            fontweight="bold"
        )

        plt.xlabel(
            "Abertura del Tamiz (mm) - Escala Logarítmica",
            fontsize=8
        )

        plt.ylabel(
            "Porcentaje Pasante (%)",
            fontsize=8
        )

        plt.grid(
            True,
            which="both",
            ls="--",
            color="#cccccc"
        )

        if p80_micras is not None:
            p80_mm = p80_micras / 1000.0
            plt.axhline(80, linestyle="--", linewidth=1)
            plt.axvline(p80_mm, linestyle="--", linewidth=1)

        buf_imagen = io.BytesIO()
        plt.savefig(
            buf_imagen,
            format="png",
            bbox_inches="tight",
            dpi=150
        )
        buf_imagen.seek(0)
        plt.close()

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------
        buf_pdf = io.BytesIO()

        doc = SimpleDocTemplate(
            buf_pdf,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        estilos = getSampleStyleSheet()

        estilo_titulo = ParagraphStyle(
            "DocTitlePDF",
            parent=estilos["Heading1"],
            fontSize=20,
            textColor=colors.HexColor("#1a5f7a"),
            spaceAfter=4,
            alignment=1
        )

        estilo_sub = ParagraphStyle(
            "DocSubPDF",
            parent=estilos["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=10,
            alignment=1
        )

        estilo_meta = ParagraphStyle(
            "MetaTextPDF",
            parent=estilos["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6
        )

        story.append(
            Paragraph(
                (payload.report_title or "LABORATORIO METALURGICO").upper(),
                estilo_titulo
            )
        )

        story.append(
            Paragraph(
                "INFORME DE ENSAYO GRANULOMÉTRICO",
                estilo_sub
            )
        )

        story.append(Spacer(1, 5))

        story.append(
            Paragraph(
                f"<b>Identificación de la Muestra:</b> "
                f"{payload.sample_name}",
                estilo_meta
            )
        )

        story.append(
            Paragraph(
                f"<b>Peso Inicial Seco:</b> "
                f"{peso_inicial if peso_inicial is not None else 'N/A'} g | "
                f"<b>Peso Total Retenido:</b> "
                f"{suma_retenido:.2f} g",
                estilo_meta
            )
        )

        story.append(
            Paragraph(
                f"<b>Recuperación de Masa:</b> "
                f"{recuperacion:.2f}%",
                estilo_meta
            )
        )

        story.append(
            Paragraph(
                (
                    f"<b>P80:</b> {p80_micras:.2f} µm"
                    if p80_micras is not None
                    else "<b>P80:</b> No disponible"
                ),
                estilo_meta
            )
        )

        color_estado = (
            "#27ae60"
            if abs(error_masa) <= 1.0
            else "#c0392b"
        )

        story.append(
            Paragraph(
                f"<b>Error de Masa:</b> "
                f"<font color='{color_estado}'>"
                f"{error_masa:.2f}%"
                f"</font> "
                f"(Límite tolerable: ±1%)",
                estilo_meta
            )
        )

        story.append(Spacer(1, 10))

        datos_tabla = [[
            "Tamiz",
            "Abertura (mm)",
            "Retenido (g)",
            "% Retenido",
            "% Ret. Acum.",
            "% Pasante"
        ]]

        for _, row in df.iterrows():
            datos_tabla.append([
                row["Malla"],
                (
                    f"{row['Abertura_mm']:.3f}"
                    if row["Malla"] != "Pan"
                    else "-"
                ),
                f"{row['Retenido_g']:.2f}",
                f"{row['Pct_Retenido']:.1f}%",
                f"{row['Cum_Pct_Retenido']:.1f}%",
                f"{row['Pct_Pasante']:.1f}%"
            ])

        tabla_pdf = Table(
            datos_tabla,
            colWidths=[70, 85, 80, 80, 100, 80]
        )

        tabla_pdf.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5f7a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ])
        )

        story.append(tabla_pdf)
        story.append(Spacer(1, 10))

        story.append(
            Image(
                buf_imagen,
                width=440,
                height=200
            )
        )

        doc.build(story)
        buf_pdf.seek(0)

        return StreamingResponse(
            buf_pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                "attachment; filename=reporte_granulometria.pdf"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
'''

# Add missing import re if not present (it is present in generated code).
out = Path("/mnt/data/main_corregido.py")
out.write_text(code, encoding="utf-8")

print(f"Archivo creado: {out}")
print(f"Líneas: {len(code.splitlines())}")
