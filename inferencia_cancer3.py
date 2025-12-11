import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import pandas as pd 
import json 
import time
import base64 
from io import BytesIO 
import numpy as np
import cv2 # Necesario para generar el mapa de calor (Grad-CAM)

# --- CONSTANTES ---
MODEL_FILENAME = "modelo_cancer_mobilenet.pth"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME 
CLASSES = ["Benigno", "Maligno", "Normal"]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# --- CONSTANTES DE BASE DE DATOS Y SEGURIDAD ---
PATIENT_DB_FILE = "patient_records.csv"
SEARCH_PASSWORD = "SALUD123" 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HopeScam tu asistente medico personal🩺🔬", page_icon="🩺", layout="wide")

# Inicialización de Session State (Memoria de Streamlit)
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'auth_search' not in st.session_state:
    st.session_state.auth_search = False
if 'current_dui' not in st.session_state:
    st.session_state.current_dui = None
if "chat_messages" not in st.session_state: 
    st.session_state.chat_messages = []

# --- TÍTULO DE LA APLICACIÓN (USANDO ESTILOS NATIVOS) ---
st.title("HopeScam tu asistente medico de confianza🩺🔬 - El Salvador")
st.markdown("---")


# --- FUNCIONES DE PERSISTENCIA DE DATOS ---

@st.cache_data
def load_patient_db():
    """Carga la base de datos de pacientes desde el CSV. Retorna un DataFrame."""
    try:
        if Path(PATIENT_DB_FILE).exists():
            df = pd.read_csv(PATIENT_DB_FILE, dtype={'DUI': str})
        else:
            df = pd.DataFrame(columns=['Nombres', 'Apellidos', 'Edad', 'Género', 'DUI', 'Fecha_Registro', 'Resultado_IA', 'Confianza_IA'])
    except Exception as e:
        st.error(f"Error al cargar la base de datos de pacientes: {e}")
        df = pd.DataFrame(columns=['Nombres', 'Apellidos', 'Edad', 'Género', 'DUI', 'Fecha_Registro', 'Resultado_IA', 'Confianza_IA'])
    return df

def save_patient_data(patient_data, scan_results):
    """Guarda los datos del nuevo paciente y los resultados del escaneo en el CSV."""
    df = load_patient_db()
    
    record = {
        'Nombres': patient_data['nombres'],
        'Apellidos': patient_data['apellidos'],
        'Edad': patient_data['edad'],
        'Género': patient_data['genero'],
        'DUI': patient_data['dui'],
        'Fecha_Registro': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Resultado_IA': scan_results.get('result', 'N/A'),
        'Confianza_IA': scan_results.get('confidence', 'N/A')
    }
    new_df = pd.DataFrame([record])
    
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(PATIENT_DB_FILE, index=False)
    
    load_patient_db.clear()
    st.session_state.current_dui = patient_data['dui']
    st.success(f"✔️ ¡Análisis y datos de {patient_data['nombres']} guardados exitosamente!")

# --- FUNCIONES DE IA (CON GRAD-CAM) ---

# Variables globales para guardar activaciones y gradientes
activations = None
gradients = None

def save_activations(module, input, output):
    """Hook para guardar la salida (activaciones) de la capa."""
    global activations
    activations = output.cpu()

def save_gradients(module, grad_input, grad_output):
    """Hook para guardar los gradientes."""
    global gradients
    gradients = grad_output[0].cpu()

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de PyTorch y registra los hooks de Grad-CAM."""
    with st.spinner("Cargando modelo..."):
        if not MODEL_PATH.exists():
            st.error(f"❌ Error: No se encontró el archivo del modelo en la ruta: `{MODEL_PATH}`.")
            return None, None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
            
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval() 
            
            # --- CONFIGURACIÓN PARA GRAD-CAM: MobileNetV2 features[18] ---
            target_layer = model.features[18]
            target_layer.register_forward_hook(save_activations)
            target_layer.register_full_backward_hook(save_gradients)
            
            return model, device
        except Exception as e:
            st.error(f"🚨 Error cargando el modelo. Detalles: {e}")
            return None, None

def predecir_imagen(model, device, image: Image.Image):
    """Realiza el preprocesamiento de la imagen y la predicción."""
    try:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Ejecutar la predicción (sin torch.no_grad() para que se calculen los gradientes)
        outputs = model(img_tensor)
        
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = outputs.argmax(dim=1).item()
        
        return pred_idx, probs.cpu().numpy(), img_tensor
            
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
        return None, None, None

def generate_grad_cam(model, target_tensor, pred_idx):
    """Genera el mapa de calor de Grad-CAM y lo superpone en la imagen original."""
    global activations, gradients
    
    if activations is None or gradients is None:
        return None

    model.zero_grad()
    
    # 1. Backward pass para la clase predicha
    one_hot = torch.zeros_like(model(target_tensor))
    one_hot[0][pred_idx] = 1
    
    target_tensor.requires_grad_(True) 
    log_probs = torch.nn.functional.log_softmax(model(target_tensor), dim=1)
    target_score = log_probs[0, pred_idx]
    target_score.backward(retain_graph=True) 

    # 2. Computar el peso alfa (media de los gradientes)
    weights = gradients.mean(dim=[2, 3], keepdim=True) 
    
    # 3. Generar el mapa de calor
    cam = (weights * activations).sum(dim=1, keepdim=True).squeeze(0).squeeze(0)
    cam = torch.relu(cam) 
    
    # 4. Normalizar y redimensionar el mapa de calor (usando numpy y cv2)
    heatmap = cam.numpy()
    heatmap = heatmap - np.min(heatmap)
    if np.max(heatmap) == 0:
        return None
    heatmap = heatmap / np.max(heatmap)
    
    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 5. Superponer en la imagen original
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1/s for s in NORM_STD]
    )
    img_pil = transforms.ToPILImage()(inv_normalize(target_tensor.squeeze(0).cpu()))
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (IMAGE_SIZE, IMAGE_SIZE))

    superimposed_img = heatmap * 0.4 + img_cv 
    superimposed_img = np.uint8(superimposed_img)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(superimposed_img)

# --- FUNCIÓN DE REPORTE ---

def create_report_content(patient_data, scan_results):
    """Genera el contenido del reporte descargable en formato HTML con las imágenes de Grad-CAM."""
    if not patient_data or not scan_results:
        return "ERROR: No hay datos de paciente o resultados de escaneo disponibles."
        
    prob_lines = "".join([
        f"<li><strong>{cls}:</strong> {prob:.2f}%</li>" for cls, prob in zip(CLASSES, scan_results['probabilities_percent'])
    ])
    
    image_b64 = scan_results.get('image_b64', '')
    gradcam_b64 = scan_results.get('gradcam_b64', '')

    image_tag = ""
    if image_b64:
        image_tag = f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color:#2F3E46;">Imagen de Ultrasonido Analizada</h3>
                <img src="data:image/jpeg;base64,{image_b64}" alt="Imagen de Ultrasonido" style="max-width: 90%; height: auto; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px;">
            </div>
        """
    
    gradcam_tag = ""
    if gradcam_b64:
         gradcam_tag = f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color:#9C27B0;">Visualización Grad-CAM (Foco de la IA)</h3>
                <img src="data:image/png;base64,{gradcam_b64}" alt="Visualización Grad-CAM" style="max-width: 90%; height: auto; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px;">
                <p><i>El color **rojo/naranja** indica la zona en la que la IA se centró para la predicción.</i></p>
            </div>
        """

    result_value = scan_results.get('result', '')
    if result_value == 'Maligno':
        recommendation_text = 'ALTO RIESGO: Se recomienda CONSULTA MÉDICA ESPECIALIZADA URGENTE.'
        result_style = 'background-color: #FFEBEE; border-left: 5px solid #F44336; color: #F44336;'
    elif result_value == 'Normal':
        recommendation_text = 'SIN HALLAZGOS: No se observan patrones de riesgo relevantes.'
        result_style = 'background-color: #E8F5E9; border-left: 5px solid #4CAF50; color: #4CAF50;'
    else:
        recommendation_text = 'BAJO RIESGO: Se recomienda SEGUIMIENTO PROFESIONAL REGULAR.'
        result_style = 'background-color: #FFF8E1; border-left: 5px solid #FFC107; color: #FFC107;'

    style = """
    <style>
        @page { size: A4; margin: 15mm; }
        body { font-family: sans-serif; margin: 0; padding: 0; }
        .report-container { width: 100%; max-width: 210mm; margin: 20px auto; border: 1px solid #e0e0e0; background-color: #ffffff;}
        .header-report { background-color: #2F3E46; color: white; padding: 20px; text-align: center; border-bottom: 5px solid #9C27B0; }
        .content-section { padding: 25px; }
        h2 { color: #9C27B0; border-bottom: 2px solid #9C27B0; padding-bottom: 8px; margin-top: 30px; font-size: 1.4em; }
        .result-ia { padding: 15px; margin-top: 20px; border-radius: 5px; font-weight: bold; }
        .disclaimer { font-size: 0.8em; color: #999; margin-top: 20px; border-top: 1px dashed #e0e0e0; padding-top: 15px; }
    </style>
    """

    content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Reporte EcoScan - {patient_data.get('apellidos', '')}</title>
        {style}
    </head>
    <body>
        <div class="report-container">
            <div class="header-report">
                <h1>ECOSCAN IA - Diagnóstico Automatizado</h1>
            </div>

            <div class="content-section">
                <h2>📋 Datos del Paciente</h2>
                <ul>
                    <li><strong>Nombres:</strong> {patient_data.get('nombres', 'N/A')}</li>
                    <li><strong>Apellidos:</strong> {patient_data.get('apellidos', 'N/A')}</li>
                    <li><strong>DUI:</strong> {patient_data.get('dui', 'N/A')}</li>
                </ul>

                <h2>🔎 Resultados del Escaneo por Inteligencia Artificial</h2>
                <div class="result-ia" style="{result_style}">
                    <p>Clase Predicha: **{result_value.upper() if result_value else 'N/A'}**</p>
                    <p>Nivel de Confianza: **{scan_results.get('confidence', 'N/A')}%**</p>
                    <p>Recomendación del Sistema: **{recommendation_text}**</p>
                </div>

                <h3>📊 Distribución de Probabilidades</h3>
                <ul> {prob_lines} </ul>

                {image_tag} 
                {gradcam_tag} 

                <div class="disclaimer">
                    ⚠️ Aviso Importante: Este reporte es generado por un sistema de IA y **NO SUSTITUYE** el diagnóstico de un médico especialista.
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return content.strip()

# --- VISTAS SECUNDARIAS (NUEVO ANÁLISIS) ---

def view_nuevo_analisis(model, device):
    """Contenido de la pestaña 'Nuevo Análisis'."""
    
    with st.expander("📝 1. Registro Obligatorio del Paciente", expanded=not st.session_state.form_submitted):
        if not st.session_state.form_submitted:
            with st.form(key='patient_form'):
                nombres = st.text_input("Nombres del Paciente (Obligatorio)", key='nombres_input')
                apellidos = st.text_input("Apellidos del Paciente (Obligatorio)", key='apellidos_input')
                edad = st.number_input("Edad del Paciente (Años)", min_value=0, max_value=120, value=30, step=1, key='edad_input')
                genero = st.selectbox("Género", ["Femenino"], key='genero_input')
                dui = st.text_input("DUI (Obligatorio para mayores de 18)", placeholder="Ej: 01234567-8", key='dui_input') if edad >= 18 else ""
                
                submit_button = st.form_submit_button(label='✅ Guardar Datos e Ir al Análisis', type="primary")

                if submit_button:
                    if not nombres or not apellidos or (edad >= 18 and not dui):
                        st.error("Por favor, complete todos los campos obligatorios.")
                    else:
                        st.session_state.patient_data = {
                            'nombres': nombres,
                            'apellidos': apellidos,
                            'edad': edad,
                            'genero': genero,
                            'dui': dui if edad >= 18 else 'Menor de 18',
                        }
                        st.session_state.form_submitted = True
                        st.success("Datos guardados. ¡Ahora puede proceder!")
                        st.rerun() 
        else:
            st.info(f"Paciente actual: **{st.session_state.patient_data['nombres']} {st.session_state.patient_data['apellidos']}** ({st.session_state.patient_data['edad']} años).")
            if st.button("Modificar datos del paciente"):
                st.session_state.form_submitted = False
                st.rerun()


    # ----------------------------------------------------
    # 2. ANÁLISIS DE IMAGEN 
    # ----------------------------------------------------
    if st.session_state.form_submitted:
        
        st.subheader("📸 2. Subida de Imagen de Ultrasonido")
        uploaded_file = st.file_uploader("Sube una imagen de ultrasonido (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="uploader_nuevo")
        
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert("RGB")
            
            # Convertir la imagen original a Base64
            buffered = BytesIO()
            image_to_analyze.save(buffered, format="JPEG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            if st.session_state.scan_results is None:
                st.session_state.scan_results = {}
            st.session_state.scan_results['image_b64'] = image_b64
            st.session_state.scan_results['gradcam_b64'] = None 

            st.subheader("🚀 3. Análisis Predictivo")
            
            if st.button("🚀 Iniciar Análisis Predictivo", type="primary", use_container_width=True, key="btn_analizar"):
                
                current_image_b64 = st.session_state.scan_results.get('image_b64', '')
                st.session_state.scan_results = {'image_b64': current_image_b64}
                
                with st.spinner("Analizando imagen con MobileNetV2 y generando Grad-CAM..."):
                    
                    # 1. Realizar la predicción
                    pred_idx, probs, img_tensor = predecir_imagen(model, device, image_to_analyze)
                    
                    if pred_idx is not None:
                        resultado = CLASSES[pred_idx]
                        confianza = probs[pred_idx] * 100
                        
                        # 2. Generar el Grad-CAM
                        gradcam_img = generate_grad_cam(model, img_tensor, pred_idx)
                        gradcam_b64 = None
                        if gradcam_img:
                            buffered_cam = BytesIO()
                            gradcam_img.save(buffered_cam, format="PNG") 
                            gradcam_b64 = base64.b64encode(buffered_cam.getvalue()).decode()
                        
                        # 3. Guardar resultados
                        st.session_state.scan_results.update({
                            'result': resultado,
                            'confidence': f"{confianza:.2f}",
                            'probabilities_percent': [p * 100 for p in probs],
                            'gradcam_b64': gradcam_b64 
                        })
                        
                        # 4. Guardar registro en la base de datos (CSV)
                        save_patient_data(st.session_state.patient_data, st.session_state.scan_results)

                        st.subheader("✅ 4. Resultados del Diagnóstico por IA")
                        
                        # --- FEEDBACK ---
                        if resultado == "Maligno":
                            st.error(f"🔴 CLASIFICACIÓN: {resultado.upper()} | Confianza: {confianza:.2f}%")
                            st.warning("La IA sugiere patrón de alto riesgo. **CONSULTA MÉDICA ESPECIALIZADA URGENTE**.")
                        elif resultado == "Benigno":
                            st.info(f"🟢 CLASIFICACIÓN: {resultado.upper()} | Confianza: {confianza:.2f}%")
                        else:
                            st.success(f"🔵 CLASIFICACIÓN: {resultado.upper()} | Confianza: {confianza:.2f}%")
                            
                        st.markdown("---")
                        
                        # Mostrar imágenes: Original y Grad-CAM
                        st.markdown("#### 🔬 Análisis de Explicabilidad (Grad-CAM)")
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(image_to_analyze, caption="Imagen de Ultrasonido Original", use_container_width=True)
                        with col_img2:
                            if gradcam_img:
                                st.image(gradcam_img, caption="Visualización Grad-CAM (Enfoque de la IA)", use_container_width=True)
                                st.markdown("El mapa de calor **rojo/amarillo** indica las áreas más relevantes para la predicción.")
                            else:
                                st.warning("No se pudo generar la visualización Grad-CAM. (Verifique la instalación de OpenCV).")
                                
                        st.markdown("---")
                        
                        st.subheader("📊 Distribución de Probabilidades")
                        prob_data = {"Clase": CLASSES, "Probabilidad (%)": [p * 100 for p in probs]}
                        df_prob = pd.DataFrame(prob_data)
                        st.bar_chart(df_prob, x='Clase', y='Probabilidad (%)', height=250)
                
                else:
                    st.warning("⚠️ No se pudo obtener el diagnóstico.")
        
        # ----------------------------------------------------
        # 3. OPCIÓN DE DESCARGA
        # ----------------------------------------------------
        if st.session_state.scan_results and st.session_state.scan_results.get('result') is not None:
            report_content = create_report_content(st.session_state.patient_data, st.session_state.scan_results)
            
            st.markdown("---")
            st.subheader("💾 Descargar Reporte (Paso Final)")
            
            st.download_button(
                label="⬇️ Descargar Reporte Completo (HTML)",
                data=report_content,
                file_name=f"Reporte_EcoScan_{st.session_state.patient_data['apellidos']}_{st.session_state.current_dui}.html",
                mime="text/html", 
                use_container_width=True,
                type="primary"
            )
            
            if st.button("🔄 Registrar Nuevo Paciente", use_container_width=True, key="btn_nuevo"):
                st.session_state.form_submitted = False
                st.session_state.patient_data = {}
                st.session_state.scan_results = None
                st.session_state.current_dui = None
                st.rerun()

def view_buscar_paciente():
    """Contenido de la pestaña 'Buscar Paciente' con autenticación."""
    
    st.subheader("🔒 Búsqueda de Pacientes - Acceso Restringido")
    
    if not st.session_state.auth_search:
        password_input = st.text_input("Contraseña de Acceso", type="password")
        
        if st.button("🔑 Ingresar", type="primary"):
            if password_input == SEARCH_PASSWORD:
                st.session_state.auth_search = True
                st.success("Acceso concedido.")
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
    
    if st.session_state.auth_search:
        
        st.subheader("🔍 Base de Datos de Pacientes Registrados")
        df_db = load_patient_db()
        
        if df_db.empty:
            st.warning("La base de datos de pacientes está vacía.")
            return

        search_term = st.text_input("Buscar por Nombre, Apellido o DUI:", key='search_term').strip().lower()
        
        if search_term:
            mask = (
                df_db['Nombres'].str.lower().str.contains(search_term, na=False) |
                df_db['Apellidos'].str.lower().str.contains(search_term, na=False) |
                df_db['DUI'].str.lower().str.contains(search_term, na=False)
            )
            filtered_df = df_db[mask]
            st.markdown(f"**Resultados encontrados:** {len(filtered_df)}")
            
            if not filtered_df.empty:
                st.dataframe(filtered_df.sort_values(by='Fecha_Registro', ascending=False), use_container_width=True)
            else:
                st.warning("No se encontraron pacientes que coincidan.")
        else:
            st.markdown("Mostrando los últimos 10 registros.")
            st.dataframe(df_db.sort_values(by='Fecha_Registro', ascending=False).head(10), use_container_width=True)

        st.markdown("---")
        if st.button("🚪 Cerrar Sesión de Búsqueda"):
            st.session_state.auth_search = False
            st.rerun()

# --- INICIALIZACIÓN DE LA APLICACIÓN ---
model, device = cargar_modelo()

# --- INTERFAZ PRINCIPAL ---

if model:
    
    col1, col2 = st.columns([1, 1.5]) 
    
    # IZQUIERDA (col1): INFORMACIÓN Y CONTACTOS
    with col1:
        st.subheader("ℹ️ Información y Contactos")
        
        with st.expander("Hospitales y Centros Oncológicos 🏥", expanded=True):
            st.markdown("### Hospital Oncológico del ISSS")
            st.write("Teléfono: `2591-5000`")
            st.write("Dirección: San Salvador.")
            st.markdown("---")
            st.markdown("### Centro Internacional de Cáncer (CIC)")
            st.write("Teléfono: `+503 2506-2001`")
            st.write("Dirección: Colonia Escalón, San Salvador.")
            

        with st.expander("Información General 🧠", expanded=False):
            st.markdown("El **Cáncer de Mama** es el cáncer más común en mujeres. La detección temprana es VITAL.")
            st.markdown("Para reducir riesgos, es fundamental realizar chequeos periódicos.")

    
    # DERECHA (col2): PESTAÑAS Y FUNCIONALIDAD CENTRAL
    with col2:
        # Implementación de Pestañas
        tab_new, tab_search = st.tabs(["🆕 Nuevo Análisis", "📂 Buscar Paciente"])

        with tab_new:
            view_nuevo_analisis(model, device)

        with tab_search:
            view_buscar_paciente()
            
else:
    st.error("⚠️ La aplicación no puede funcionar. Revisa que el modelo esté en la ruta correcta y que PyTorch esté bien configurado.")


# ==============================================
#  CHATBOT SIMPLE
# ==============================================

# El chatbot se coloca en un Expander al final de la página para que sea simple y no requiera CSS
with st.expander("💬 Asistente Virtual EcoScan (Ayuda Rápida)", expanded=False):
    
    # Mostrar mensajes en el chat
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.