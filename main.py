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

app = FastAPI(title="API de Granulometría Geotécnica")

# 1. Modelos de entrada de datos (JSON)
class DatosMalla(BaseModel):
    sieve: str
    retained_weight_g: float

class PeticionAnalisis(BaseModel):
    total_initial_weight: Optional[float] = None
    data: List[DatosMalla]

# Mapeo estándar de aberturas de tamices según normas ASTM / AASHTO
ABERTURAS_MALLAS = {
    '3"': 75.0, '2"': 50.0, '1 1/2"': 37.5, '1"': 25.0, '3/4"': 19.0, '1/2"': 12.5, '3/8"': 9.5,
    'No. 4': 4.75, 'No. 8': 2.36, 'No. 10': 2.00, 'No. 16': 1.18, 'No. 30': 0.60, 'No. 40': 0.425,
    'No. 50': 0.30, 'No. 100': 0.15, 'No. 200': 0.075, 'Pan': 0.001  # El fondo usa un valor mínimo para la escala logarítmica
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
        
        # 3. Cálculos de Ingeniería Civil
        df["Pct_Retenido"] = (df["Retenido_g"] / suma_retenido) * 100
        df["Cum_Retenido_g"] = df["Retenido_g"].cumsum()
        df["Cum_Pct_Retenido"] = df["Pct_Retenido"].cumsum()
        df["Pct_Pasante"] = 100 - df["Cum_Pct_Retenido"]
        
        # Control de calidad: Cálculo del Error de Masa
        error_masa = 0.0
        if payload.total_initial_weight and payload.total_initial_weight > 0:
            error_masa = ((suma_retenido - payload.total_initial_weight) / payload.total_initial_weight) * 100

        # 4. Generar la Gráfica Semilogarítmica en Memoria RAM
        df_grafica = df[df["Malla"] != "Pan"] # Excluimos el fondo para que la curva no caiga a cero abruptamente
        plt.figure(figsize=(7, 3.5))
        plt.plot(df_grafica["Abertura_mm"], df_grafica["Pct_Pasante"], marker='o', color='#1a5f7a', linewidth=2)
        plt.xscale('log')
        plt.xlim(100, 0.01) # Inversión del eje X: mallas gruesas a la izquierda, finas a la derecha
        plt.ylim(0, 105)
        plt.title('Curva de Distribución Granulométrica', fontsize=12, fontweight='bold')
        plt.xlabel('Abertura del Tamiz (mm) - Escala Logarítmica', fontsize=9)
        plt.ylabel('Porcentaje Pasante (%)', fontsize=9)
        plt.grid(True, which="both", ls="--", color='#cccccc')
        
        buf_imagen = io.BytesIO()
        plt.savefig(buf_imagen, format='png', bbox_inches='tight', dpi=150)
        buf_imagen.seek(0)
        plt.close()

        # 5. Diseñar la estructura del PDF Dinámico
        buf_pdf = io.BytesIO()
        doc = SimpleDocTemplate(buf_pdf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        estilos = getSampleStyleSheet()
        
        # Estilos visuales personalizados
        estilo_titulo = ParagraphStyle('DocTitle', parent=estilos['Heading1'], fontSize=18, textColor=colors.HexColor('#1a5f7a'), spaceAfter=6)
        estilo_meta = ParagraphStyle('MetaText', parent=estilos['Normal'], fontSize=10, textColor=colors.HexColor('#333333'), spaceAfter=12)

        # Encabezado del Reporte
        story.append(Paragraph("INFORME DE LABORATORIO GEOTÉCNICO", estilo_titulo))
        story.append(Paragraph(f"<b>Peso Inicial de la Muestra:</b> {payload.total_initial_weight or 'N/A'} g | <b>Peso Total Lavado/Retenido:</b> {suma_retenido:.2f} g", estilo_meta))
        
        # Alerta visual si el error supera el límite de la norma (típicamente +-1%)
        color_estado = "#27ae60" if abs(error_masa) <= 1.0 else "#c0392b"
        story.append(Paragraph(f"<b>Error de Masa Calculado:</b> <font color='{color_estado}'>{error_masa:.2f}%</font> (Límite tolerable: ±1%)", estilo_meta))
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
            
        tabla_pdf = Table(datos_tabla, colWidths=[80, 80, 80, 80, 100, 80])
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
        story.append(Spacer(1, 15))

        # Inserción de la curva gráfica dentro del PDF
        story.append(Image(buf_imagen, width=450, height=225))
        
        # Renderizar documento y preparar el flujo de salida
        doc.build(story)
        buf_pdf.seek(0)
        
        return StreamingResponse(buf_pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=reporte_granulometria.pdf"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
