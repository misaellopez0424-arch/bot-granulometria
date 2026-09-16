import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = FastAPI(title="API de Granulometría Metalúrgica y Geotécnica")

# 1. Modelos de entrada de datos actualizados (JSON)
class DatosMalla(BaseModel):
    sieve: str
    retained_weight_g: float

class PeticionAnalisis(BaseModel):
    report_title: Optional[str] = "LABORATORIO METALURGICO" # Título dinámico con valor por defecto
    sample_name: Optional[str] = "Muestra No Identificada"    # Nombre de muestra corregido por IA
    total_initial_weight: Optional[float] = None
    data: List[DatosMalla]

# Mapeo estándar de aberturas de tamices según normas internacionales
ABERTURAS_MALLAS = {
    '3"': 75.0, '2"': 50.0, '1 1/2"': 37.5, '1"': 25.0, '3/4"': 19.0, '1/2"': 12.5, '3/8"': 9.5,
    'No. 4': 4.75, 'No. 8': 2.36, 'No. 10': 2.00, 'No. 16': 1.18, 'No. 30': 0.60, 'No. 40': 0.425,
    'No. 50': 0.30, 'No. 100': 0.15, 'No. 200': 0.075, 'Pan': 0.001
}

@app.post("/api/v1/granulometria")
async def generar_reporte_granulometrico(payload: PeticionAnalisis):
    try:
        # 2. Convertir JSON a DataFrame de Pandas
        registros = []
        for item in payload.data:
            abertura = ABERTURAS_MALLAS.get(item.sieve, 0.001)
            registros.append({
                "Malla": item.sieve,
                "Abertura_mm": abertura,
                "Retenido_g": item.retained_weight_g
            })
            
        df = pd.DataFrame(registros)
        suma_retenido = df["Retenido_g"].sum()
        
        # 3. Cálculos Metalúrgicos / Geotécnicos
        df["Pct_Retenido"] = (df["Retenido_g"] / suma_retenido) * 100
        df["Cum_Retenido_g"] = df["Retenido_g"].cumsum()
        df["Cum_Pct_Retenido"] = df["Pct_Retenido"].cumsum()
        df["Pct_Pasante"] = 100 - df["Cum_Pct_Retenido"]
        
        error_masa = 0.0
        if payload.total_initial_weight and payload.total_initial_weight > 0:
            error_masa = ((suma_retenido - payload.total_initial_weight) / payload.total_initial_weight) * 100

        # 4. Generar la Gráfica Semilogarítmica en Memoria RAM
        df_grafica = df[df["Malla"] != "Pan"]
        plt.figure(figsize=(7, 3.2))
        plt.plot(df_grafica["Abertura_mm"], df_grafica["Pct_Pasante"], marker='o', color='#1a5f7a', linewidth=2)
        plt.xscale('log')
        plt.xlim(100, 0.01)
        plt.ylim(0, 105)
        plt.title('Curva de Distribución Granulométrica', fontsize=11, fontweight='bold')
        plt.xlabel('Abertura del Tamiz (mm) - Escala Logarítmica', fontsize=8)
        plt.ylabel('Porcentaje Pasante (%)', fontsize=8)
        plt.grid(True, which="both", ls="--", color='#cccccc')
        
        buf_imagen = io.BytesIO()
        plt.savefig(buf_imagen, format='png', bbox_inches='tight', dpi=150)
        buf_imagen.seek(0)
        plt.close()

        # 5. Diseñar la estructura del PDF con los nuevos campos
        buf_pdf = io.BytesIO()
        doc = SimpleDocTemplate(buf_pdf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        estilos = getSampleStyleSheet()
        
        # Estilos visuales personalizados (Color corporativo azul metalúrgico)
        estilo_titulo = ParagraphStyle('DocTitle', parent=estilos['Heading1'], fontSize=20, textColor=colors.HexColor('#1a5f7a'), spaceAfter=4, alignment=1)
        estilo_sub = ParagraphStyle('DocSub', parent=estilos['Heading2'], fontSize=12, textColor=colors.HexColor('#2c3e50'), spaceAfter=10, alignment=1)
        estilo_meta = ParagraphStyle('MetaText', parent=estilos['Normal'], fontSize=10, textColor=colors.HexColor('#333333'), spaceAfter=6)

        # Encabezado Dinámico del Reporte usando los nuevos campos
        story.append(Paragraph(payload.report_title.upper(), estilo_titulo))
        story.append(Paragraph("INFORME DE ENSAYO GRANULOMÉTRICO", estilo_sub))
        story.append(Spacer(1, 5))
        
        # Mostrar el nombre canónico corregido por la IA
        story.append(Paragraph(f"<b>Identificación de la Muestra:</b> {payload.sample_name}", estilo_meta))
        story.append(Paragraph(f"<b>Peso Inicial Seco:</b> {payload.total_initial_weight or 'N/A'} g | <b>Peso Total Retenido:</b> {suma_retenido:.2f} g", estilo_meta))
        
        color_estado = "#27ae60" if abs(error_masa) <= 1.0 else "#c0392b"
        story.append(Paragraph(f"<b>Error de Masa Dinámico:</b> <font color='{color_estado}'>{error_masa:.2f}%</font> (Límite tolerable: ±1%)", estilo_meta))
        story.append(Spacer(1, 10))

        # Estructuración de la Tabla de Datos
        datos_tabla = [["Tamiz", "Abertura (mm)", "Retenido (g)", "% Retenido", "% Ret. Acumulado", "% Pasante"]]
        for _, row in df.iterrows():
            datos_tabla.append([
                row["Malla"],
                f"{row['Abertura_mm']:.3f}" if row["Malla"] != "Pan" else "-",
                f"{row['Retenido_g']:.2f}",
                f"{row['Pct_Retenido']:.1f}%",
                f"{row['Cum_Pct_Retenido']:.1f}%",
                f"{row['Pct_Pasante']:.1f}%"
            ])
            
        tabla_pdf = Table(datos_tabla, colWidths=[70, 85, 80, 80, 100, 80])
        tabla_pdf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5f7a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0,1), (-1,-1), 9),
        ]))
        story.append(tabla_pdf)
        story.append(Spacer(1, 10))

        # Inserción de la curva gráfica dentro del PDF
        story.append(Image(buf_imagen, width=440, height=200))
        
        doc.build(story)
        buf_pdf.seek(0)
        
        return StreamingResponse(buf_pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=reporte_granulometria.pdf"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
