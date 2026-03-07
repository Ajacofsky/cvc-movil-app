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
# 🔒 MOTOR DE VISIÓN Y "PRUEBA DEL CINCEL"
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

def classify_symbol(roi_bin):
    """La Prueba del Cincel (Erosión Topológica)"""
    h, w = roi_bin.shape
    area_caja = float(w * h)
    tinta_total = cv2.countNonZero(roi_bin)

    if tinta_total < 4 or area_caja < 4:
        return 'ignorar'

    densidad = tinta_total / area_caja

    # 1. Si es un bloque negro denso casi perfecto, es un cuadrado.
    if densidad > 0.65:
        return 'fallado'

    # 2. Si es dudoso, aplicamos "El Cincel" (Erosión).
    # Calculamos un cincel del 35% del tamaño del símbolo
    k_size = max(2, int(min(w, h) * 0.35))
    kernel = np.ones((k_size, k_size), np.uint8)
    
    # Raspamos la imagen
    eroded = cv2.erode(roi_bin, kernel, iterations=1)
    tinta_nucleo = cv2.countNonZero(eroded)

    # Si después de raspar los bordes, sobrevive un núcleo fuerte en el centro, era macizo (Cuadrado).
    # Si desaparece casi todo, era un anillo hueco (Círculo).
    if tinta_nucleo > (area_caja * 0.05):
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
    
    # Tolerancia de tamaños permitidos
    area_min, area_max = (ancho * 0.001) ** 2, (ancho * 0.03) ** 2
    cx, cy = centro
    cuadrados_count, circulos_count = 0, 0
    
    if pixels_por_10_grados <= 0: pixels_por_10_grados = 1.0 
            
    for i in range(1, num_labels): 
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        
        if area_min < area < area_max and 0.4 < (w/float(h)) < 2.5:
            px, py = x + w/2.0, y + h/2.0
            
            if (math.hypot(px - cx, py - cy) / pixels_por_10_grados) * 10.0 <= 41.0:
                roi = campo_limpio[y:y+h, x:x+w]
                
                # Enviamos el símbolo a la prueba pericial
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
        with st.spinner(f"Analizando con IA Pericial Topológica..."):
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
            # Filtro ideal para fotos de celular
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 6)
            
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
                cuadrados_final = st.number_input("Cuadrados (Fallados):", min_value=0, max_value=104, value=t_cuad, step=1, key=f"cuad_{key_suffix}")
            with col_b:
                circulos_final = st.number_input("Círculos (Vistos):", min_value=0, max_value=104, value=t_circ, step=1, key=f"circ_{key_suffix}")
                
            grados_finales = (cuadrados_final / 104.0) * 320.0
            incapacidad_final = (grados_finales / 320.0) * 100 * 0.25
            
            st.info(f"👉 Incapacidad Detectada: {incapacidad_final:.2f}%")
            
    return incapacidad_final, grados_finales, img_original

if modo_evaluacion == "Unilateral (1 Ojo)":
    incap_od, grados_od, img_od_orig = procesar_panel_camara("Ojo Evaluado", "unico")
    incap_oi, grados_oi, img_oi_orig = 0.0, 0.0, None
else:
    incap_od, grados_od, img_od_orig = procesar_panel_camara("Ojo Derecho (OD)", "od")
    st.divider()
    incap_oi, grados_oi, img_oi_orig = procesar_panel_camara("Ojo Izquierdo (OI)", "oi")

st.divider()

st.header("📋 Exportar Dictamen")
nombre_archivo_input = st.text_input("Nombre del paciente para el archivo:", placeholder="Ej: Perez_Juan")
incap_total_bilateral = 0.0

if modo_evaluacion == "Bilateral (OD y OI)":
    if img_od_orig is not None or img_oi_orig is not None:
        suma_aritmetica = incap_od + incap_oi
        incap_total_bilateral = suma_aritmetica * 1.5
        st.metric("INCAPACIDAD TOTAL BILATERAL", f"{incap_total_bilateral:.2f}%")
else:
    if img_od_orig is not None:
        st.metric("INCAPACIDAD UNILATERAL", f"{incap_od:.2f}%")

if img_od_orig is not None or img_oi_orig is not None:
    nombre_archivo = nombre_archivo_input.strip().replace(" ", "_") if nombre_archivo_input.strip() else "Dictamen"
    b64_pdf = generar_pdf_moderno(incap_od, grados_od, img_od_orig, incap_oi, grados_oi, img_oi_orig, incap_total_bilateral, modo_evaluacion)
    
    html_btn = f'''
    <a href="data:application/pdf;base64,{b64_pdf}" download="Dictamen_Pericial_{nombre_archivo}.pdf" style="display: block; padding: 15px; background-color: #2980b9; color: white; text-align: center; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 18px; margin-top: 20px;">
        📥 DESCARGAR PDF
    </a>
    '''
    st.markdown(html_btn, unsafe_allow_html=True)
