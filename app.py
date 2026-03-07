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
# 🔒 MOTOR DE VISIÓN (RADAR DE PRECISIÓN)
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
# 🧠 EL CLASIFICADOR GEOMÉTRICO INMUTABLE
# ==========================================
def classify_symbol(roi_bin, pixels_por_10_grados):
    h, w = roi_bin.shape
    area_caja = float(w * h)
    tinta = cv2.countNonZero(roi_bin)

    # 1. Ignorar pelusas o ruido de la foto
    if tinta < 4 or area_caja < 4:
        return 'ignorar'

    lado_esperado = pixels_por_10_grados * 0.15 
    area_esperada = lado_esperado ** 2

    # 2. Si la mancha es muy pequeña comparada con un cuadrado estándar, es un círculo verde.
    if area_caja < (area_esperada * 0.40) or tinta < (area_esperada * 0.30):
        return 'visto'

    # 3. DOBLE FILTRO BLINDADO
    # Filtro A (Rayos X): Analizamos el corazón del símbolo
    cx, cy = w // 2, h // 2
    rw, rh = max(1, int(w * 0.15)), max(1, int(h * 0.15))
    nucleo = roi_bin[cy-rh : cy+rh+1, cx-rw : cx+rw+1]
    
    if nucleo.size > 0:
        densidad_nucleo = cv2.countNonZero(nucleo) / float(nucleo.size)
        # Si el centro no es negro intenso (<70% tinta), es un círculo que se emborronó
        if densidad_nucleo < 0.70:
            return 'visto'

    # Filtro B (La Ley de Pi/4): 
    # Un círculo, por más sólido que se vea, no puede llenar las 4 esquinas de su caja.
    # Su máxima densidad teórica es 0.785. Un cuadrado supera el 0.85.
    densidad_total = tinta / area_caja
    
    if densidad_total > 0.75: # Umbral mágico que divide el círculo del cuadrado
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
                roi = campo_limpio[y:y+h, x:x+w]
                
                # Clasificación Geométrica
                tipo = classify_symbol(roi, pixels_por_10_grados)
                
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
        pdf.set_fill_color(235, 245, 255)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, " OJO IZQUIERDO (OI)", 0, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"   - Grados de perdida visual:  {grados_oi:.1f} grados", 0, 1)
        pdf.cell(0, 8, f"   - Incapacidad Unilateral:    {incap_oi:.2f}%", 0, 1)
        pdf.ln(3)
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    if modo == "Bilateral (OD y OI)":
        pdf.set_fill_color(46, 134, 193) 
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 14, f" INCAPACIDAD TOTAL BILATERAL: {incap_total:.2f}%", 0, 1, 'C', fill=True)
    else:
        pdf.set_fill_color(46, 134, 193)
        pdf.set_text_color(255, 255, 255)
        val = incap_od if incap_od > 0 else incap_oi
        pdf.cell(0, 14, f" INCAPACIDAD UNILATERAL DEFINITIVA: {val:.2f}%", 0, 1, 'C', fill=True)
        
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(65, pdf.get_y(), 145, pdf.get_y())
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, "Firma y Sello del Perito Medico", 0, 1, 'C')
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return base64.b64encode(pdf_bytes).decode()

# ==========================================
# INTERFAZ WEB MÓVIL
# ==========================================
modo_evaluacion = st.radio("Seleccione el Tipo de Evaluación:", ["Unilateral (1 Ojo)", "Bilateral (OD y OI)"], horizontal=True)
st.divider()

def procesar_panel_camara(titulo_ojo, key_suffix):
    archivo = st.camera_input(f"📷 Capturar - {titulo_ojo}", key=f"cam_{key_suffix}")
    
    incapacidad_final, grados_finales, img_original = 0.0, 0.0, None
    t_cuad, t_circ = 0, 0
    
    if archivo is not None:
        with st.spinner(f"Aplicando Inteligencia Geométrica..."):
            nparr = np.frombuffer(archivo.getvalue(), np.uint8)
            img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            alto_raw, ancho_raw = img_raw.shape[:2]
            max_dimension = 1800 
            if ancho_raw > max_dimension or alto_raw > max_dimension:
                escala = max_dimension / max(ancho_raw, alto_raw)
                img = cv2.resize(img_raw, (int(ancho_raw * escala), int(alto_raw * escala)), interpolation=cv2.INTER_CUBIC)
            else:
                img = img_raw

            img_original = img.copy()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
            
            try:
                centro, borrador_anti_regla, dist_60 = find_and_clean_axes(thresh)
                pixels_por_10_grados = float(dist_60 / 6.0)
                
                img_auditoria_bin, t_cuad, t_circ = detect_and_classify_symbols(thresh, borrador_anti_regla, centro, pixels_por_10_grados)
                
                img_pantalla = img.copy()
                for i in range(3):
                    mask = img_auditoria_bin[:,:,i] != 255
                    img_pantalla[mask, i] = img_auditoria_bin[mask, i]
                cv2.circle(img_pantalla, centro, int(4.0 * pixels_por_10_grados), (0, 165, 255), 3)

                st.success("✅ ¡Análisis completado!")
                st.image(Image.fromarray(cv2.cvtColor(img_pantalla, cv2.COLOR_BGR2RGB)), caption=f"Auditoría Visual {titulo_ojo}", use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Error al encuadrar la foto. Acerque más la cámara al círculo. Detalle: {e}")
                
            st.markdown(f"**Corrección Pericial Manual**")
            col_a, col_b = st.columns(2)
            with col_a:
                val_cuad_seguro = min(t_cuad, 104)
                cuadrados_final = st.number_input("Cuadrados (Fallados):", min_value=0, max_value=104, value=val_cuad_seguro, step=1, key=f"cuad_{key_suffix}")
            with col_b:
                val_circ_seguro = min(t_circ, 104)
                circulos_final = st.number_input("Círculos (Vistos):", min_value=0, max_value=104, value=val_circ_seguro, step=1, key=f"circ_{key_suffix}")
                
            grados_finales = (cuadrados_final / 104.0) * 320.0
            incapacidad_final = (grados_finales / 320.0) * 100 * 0.25
            
            st.info(f"👉 Incapacidad Detectada: {incapacidad_final:.2f}%")
            
    return incapacidad_final, grados_final
