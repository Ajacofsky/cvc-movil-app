import streamlit as st
import cv2
import numpy as np
import math
import base64
import os
import tempfile
from PIL import Image
from fpdf import FPDF

st.set_page_config(page_title="CVC Móvil", layout="wide")

st.title("📱 Escáner Móvil Pericial CVC")
st.markdown("""
**Modo Captura por Cámara**
Tome una foto directa del campo visual impreso. 
*💡 Consejo de Oro: Acerque la cámara para que el gráfico ocupe casi toda la pantalla.*
""")

# ==========================================
# 🔒 MOTOR DE VISIÓN (RADAR)
# ==========================================
def find_and_clean_axes(thresh):
    alto, ancho = thresh.shape
    
    k_len_h = max(30, int(ancho * 0.18))
    k_len_v = max(30, int(alto * 0.18))
    kernel_h_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len_h, 1))
    kernel_v_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len_v))
    
    lineas_h_puras = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h_clean)
    lineas_v_puras = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v_clean)
    
    interseccion = cv2.bitwise_and(lineas_h_puras, lineas_v_puras)
    zona_media_inter = interseccion[int(alto*0.2):int(alto*0.8), int(ancho*0.2):int(ancho*0.8)]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(zona_media_inter)
    
    if max_val > 0:
        cx = max_loc[0] + int(ancho*0.2)
        cy = max_loc[1] + int(alto*0.2)
    else:
        cy = int(alto * 0.5)
        cx = int(ancho * 0.5)
    
    y_coords, x_coords = np.where(lineas_h_puras > 0)
    tolerancia_y = max(30, int(alto * 0.08)) 
    mask_cerca_cy = np.abs(y_coords - cy) < tolerancia_y
    x_validos = x_coords[mask_cerca_cy]
    
    if len(x_validos) > 0:
        dist_60_izq = cx - np.min(x_validos)
        dist_60_der = np.max(x_validos) - cx
        dist_60 = (dist_60_izq + dist_60_der) / 2.0
    else:
        dist_60 = ancho * 0.35 
        
    dist_60 = min(dist_60, ancho * 0.49) 
    
    grosor_fino_h = max(2, int(alto*0.003))
    grosor_fino_v = max(2, int(ancho*0.003))
    borrador_h_ticks = cv2.dilate(lineas_h_puras, np.ones((grosor_fino_h, 1), np.uint8))
    borrador_v_ticks = cv2.dilate(lineas_v_puras, np.ones((1, grosor_fino_v), np.uint8))
    borrador_anti_regla = cv2.add(borrador_h_ticks, borrador_v_ticks)
    
    return (cx, cy), borrador_anti_regla, dist_60

# ==========================================
# 🧠 RECONOCIMIENTO DE PATRONES (MÉTODO BORDES)
# ==========================================
def classify_symbol(roi_bin):
    h, w = roi_bin.shape
    tinta = cv2.countNonZero(roi_bin)
    
    if tinta < 4:
        return 'ignorar'

    # MÉTODO DE BORDES: Un cuadrado siempre llena sus bordes. Un círculo no.
    # Evaluamos la periferia del símbolo (el 25% externo).
    margen_h = max(1, int(h * 0.25))
    margen_w = max(1, int(w * 0.25))
    
    # Extraer los 4 bordes
    borde_sup = roi_bin[0:margen_h, :]
    borde_inf = roi_bin[h-margen_h:h, :]
    borde_izq = roi_bin[:, 0:margen_w]
    borde_der = roi_bin[:, w-margen_w:w]
    
    tinta_bordes = cv2.countNonZero(borde_sup) + cv2.countNonZero(borde_inf) + cv2.countNonZero(borde_izq) + cv2.countNonZero(borde_der)
    area_bordes = float(borde_sup.size + borde_inf.size + borde_izq.size + borde_der.size)
    
    # Si la periferia está muy llena de negro, es el bloque de un cuadrado macizo.
    # Un círculo, por más engordado que esté, siempre tendrá esquinas y bordes con menos tinta.
    if (tinta_bordes / area_bordes) > 0.60:
        return 'fallado'
    else:
        return 'visto'

def detect_and_classify_symbols(img_bin, borrador_anti_regla, centro, pixels_por_10_grados):
    alto, ancho = img_bin.shape
    img_auditoria = np.zeros((alto, ancho, 3), dtype=np.uint8) 
    img_auditoria[:,:] = [255, 255, 255] 
    
    campo_limpio = cv2.subtract(img_bin, borrador_anti_regla)
    
    grosor_pegamento = max(2, int(alto*0.002))
    simbolos_unidos = cv2.dilate(campo_limpio, np.ones((grosor_pegamento, grosor_pegamento), np.uint8))
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(simbolos_unidos, connectivity=8)
    
    area_min = (ancho * 0.0005) ** 2 
    area_max = (ancho * 0.04) ** 2
    cx, cy = centro
    cuadrados_count, circulos_count = 0, 0
    
    if pixels_por_10_grados <= 0: pixels_por_10_grados = 1.0 
            
    for i in range(1, num_labels): 
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        
        if area_min < area < area_max and 0.3 < (w/float(h)) < 3.0:
            px, py = x + w/2.0, y + h/2.0
            
            if (math.hypot(px - cx, py - cy) / pixels_por_10_grados) * 10.0 <= 41.0:
                # Damos un pequeño respiro alrededor de la mancha para la prueba de bordes
                y1 = max(0, y - 1)
                y2 = min(alto, y + h + 1)
                x1 = max(0, x - 1)
                x2 = min(ancho, x + w + 1)
                roi = campo_limpio[y1:y2, x1:x2]
                
                tipo = classify_symbol(roi)
                
                if tipo == 'fallado':
                    cuadrados_count += 1
                    cv2.rectangle(img_auditoria, (x, y), (x+w, y+h), (0, 0, 255), 2)
                elif tipo == 'visto':
                    circulos_count += 1
                    cv2.rectangle(img_auditoria, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return img_auditoria, cuadrados_count, circulos_count

# ==========================================
# GENERADOR DE PDF
# ==========================================
def generar_pdf_moderno(incap_od, grados_od, img_od_orig, incap_oi, grados_oi, img_oi_orig, incap_total, modo):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_fill_color(41, 64, 115) 
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 16, "  DICTAMEN PERICIAL - CAMPO VISUAL COMPUTARIZADO", 0, 1, 'C', fill=True)
    pdf.ln(8) 

    y_images = pdf.get_y()
    
    if modo == "Bilateral (OD y OI)":
        if img_od_orig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_od:
                cv2.imwrite(tmp_od.name, img_od_orig)
                pdf.image(tmp_od.name, x=10, y=y_images, w=90) 
            os.remove(tmp_od.name)
        if img_oi_orig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_oi:
                cv2.imwrite(tmp_oi.name, img_oi_orig)
                pdf.image(tmp_oi.name, x=110, y=y_images, w=90) 
            os.remove(tmp_oi.name)
        pdf.set_y(y_images + 95) 
    else:
        img_val = img_od_orig if img_od_orig is not None else img_oi_orig
        if img_val is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img_val)
                pdf.image(tmp.name, x=55, y=y_images, w=100) 
            os.remove(tmp.name)
        pdf.set_y(y_images + 115)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "RESULTADOS DE LA EVALUACION (AREA 40 GRADOS)", 0, 1, 'L')
    pdf.ln(2)
    
    if incap_od > 0 or (modo == "Unilateral (1 Ojo)" and img_od_orig is not None):
        pdf.set_fill_color(235, 245, 255) 
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, " OJO DERECHO (OD) / EVALUADO", 0, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"   - Grados de perdida visual:  {grados_od:.1f} grados", 0, 1)
        pdf.cell(0, 8, f"   - Incapacidad Unilateral:    {incap_od:.2f}%", 0, 1)
        pdf.ln(3)
        
    if incap_oi > 0 or (modo == "Bilateral (OD y OI)" and img_oi_orig is not None):
        pdf.set_fill_color
