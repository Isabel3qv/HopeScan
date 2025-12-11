import torch
from torchvision import transforms, models
from PIL import Image
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Ajusta estas rutas según tu carpeta real
# Nota: Usamos 'r' antes de las comillas para evitar errores con las barras \ en Windows
MODEL_PATH = r"C:\Users\MINEDUCYT\Downloads\bootcamp\modelo_cancer_mobilenet(2).pth"
IMAGE_PATH = r"C:\Users\MINEDUCYT\Downloads\bootcamp-ia\src\test_img\b01.png"  # <--- ¡Asegúrate de tener una imagen aquí!

# ==========================================
# 1. PREPARACIÓN
# ==========================================
print("🚀 Iniciando sistema de diagnóstico...")

# Verificar si los archivos existen antes de intentar cargar nada
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR CRÍTICO: No encuentro el modelo en: {MODEL_PATH}")
    print("   -> ¿Ejecutaste el entrenamiento? ¿Descargaste el archivo .pth?")
    exit()

if not os.path.exists(IMAGE_PATH):
    print(f"❌ ERROR CRÍTICO: No encuentro la imagen en: {IMAGE_PATH}")
    print("   -> Descarga una imagen de ultrasonido y guárdala con ese nombre.")
    exit()

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Usando dispositivo: {device}")

# Definir transformaciones (Las mismas que en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ==========================================
# 2. CARGAR EL MODELO
# ==========================================
print("🧠 Cargando arquitectura y pesos del modelo...")

try:
    # 1. Crear la arquitectura vacía
    model = models.mobilenet_v2(weights=None)
    # 2. Ajustar la capa final (IMPORTANTE: Debe coincidir con el entrenamiento)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    
    # 3. Cargar los pesos aprendidos
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # 4. Mover al dispositivo y poner en modo evaluación
    model = model.to(device)
    model.eval()
    print("✅ Modelo cargado correctamente.")

except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print("   -> Verifica que la arquitectura coincida con la del script de entrenamiento.")
    exit()

# ==========================================
# 3. PREDECIR
# ==========================================
def predecir(ruta):
    print(f"🔍 Analizando imagen: {ruta}")
    try:
        # Abrir imagen
        img = Image.open(ruta).convert("RGB")
        
        # Transformar y agregar dimensión de lote (batch dim)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inferencia (sin calcular gradientes)
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Obtener probabilidades con Softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Obtener la clase ganadora
            pred_idx = outputs.argmax(dim=1).item()
            
        clases = ["Benigno", "Maligno"] # 0 y 1
        resultado = clases[pred_idx]
        confianza = probs[pred_idx].item() * 100
        
        return resultado, confianza

    except Exception as e:
        print(f"❌ Error procesando la imagen: {e}")
        return None, 0

# ==========================================
# 4. EJECUCIÓN
# ==========================================
diagnostico, probabilidad = predecir(IMAGE_PATH)

if diagnostico:
    print("\n" + "="*30)
    print(f"🩺 RESULTADO DEL DIAGNÓSTICO")
    print("="*30)
    print(f"📂 Imagen: {os.path.basename(IMAGE_PATH)}")
    print(f"🦠 Predicción: {diagnostico.upper()}")
    print(f"📊 Confianza:  {probabilidad:.2f}%")
    print("="*30 + "\n")