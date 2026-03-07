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
**Modo Captura en Vivo**
Encuadre el gráfico en la pantalla y tome la foto directamente. 
*💡 Consejo de Oro: Si ve su rostro, toque el botón de las flechas (🔄 Switch camera) para usar la cámara trasera.*
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
# 🧠 IA DE RECONOCIMIENTO DE NÚCLEO (Anti-borroso)
# ==========================================
def classify_symbol(roi_bin):
    h, w = roi_bin.shape
    tinta = cv2.countNonZero(roi_bin)
    area_caja = float(w * h)
    
    # 1. Ignorar ruido
    if tinta < 4 or area_caja < 4:
        return 'ignorar'

    densidad = tinta / area_caja

    # 2. Si no es lo suficientemente denso (tiene mucho espacio blanco), es un círculo.
    if densidad < 0.50:
        return 'visto'

    # 3. Rayos X al centro: Si es un cuadrado macizo, su centro debe ser puro negro.
    cx, cy = w // 2, h // 2
    rw, rh = max(1, int(w * 0.20)), max(1, int(h * 0.20))
    nucleo = roi_bin[cy-rh : cy+rh+1, cx-rw : cx+rw+1]
    
    tinta_nucleo = cv2.countNonZero(nucleo)
    
    if nucleo.size > 0 and (tinta_nucleo / float(nucleo.size)) > 0.65:
        return 'fallado' # Núcleo sólido = Cuadrado
    else:
        return 'visto' # Núcleo hueco o difuso = Círculo

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
