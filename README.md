# 🌍 Sistema de Clasificación Automática de ODS

Sistema de inteligencia artificial para clasificar automáticamente opiniones ciudadanas según los Objetivos de Desarrollo Sostenible (ODS) 1, 3 y 4 de la ONU.

**Desarrollado para:** Fondo de Población de las Naciones Unidas (UNFPA)

---

## Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Ejecución](#-ejecución)
- [Uso de la API](#-uso-de-la-api)
- [Aplicación Web](#-aplicación-web)
- [Re-entrenamiento del Modelo](#-re-entrenamiento-del-modelo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Solución de Problemas](#-solución-de-problemas)
- [Documentación Técnica](#-documentación-técnica)

---

## Características

- 🤖 **Clasificación Automática**: Predice si una opinión ciudadana corresponde a ODS 1 (Pobreza), 3 (Salud) o 4 (Educación)
- 📊 **Probabilidades por Clase**: Proporciona el nivel de confianza de cada predicción
- 🔄 **Re-entrenamiento Continuo**: Los expertos pueden mejorar el modelo con nuevos datos
- 🎯 **Alta Precisión**: Modelo LinearSVC con precisión > 95%
- 🌐 **Interfaz Web Intuitiva**: React + Tailwind CSS
- 🔌 **API REST**: NestJS con documentación Swagger
- 💾 **Gestión de Versiones**: Backups automáticos del modelo

---

## Arquitectura

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │ ───▶ │  API NestJS  │ ───▶ │   Python    │
│  Frontend   │ ◀─── │   Backend    │ ◀─── │  ML Model   │
└─────────────┘      └──────────────┘      └─────────────┘
   Puerto 3000          Puerto 3001         LinearSVC+TF-IDF
```

**Tecnologías:**
- **Frontend**: React 18, Vite, Tailwind CSS, Lucide Icons
- **Backend**: NestJS, TypeScript, class-validator
- **ML**: Python 3.8+, scikit-learn, joblib, numpy
- **Modelo**: LinearSVC con TF-IDF (Stemming + Stopwords en español)

---

## Requisitos

### **Software Necesario:**

- **Node.js** 18.x o superior ([Descargar](https://nodejs.org/))
- **Python** 3.8 o superior ([Descargar](https://www.python.org/))
- **npm** 8.x o superior (incluido con Node.js)
- **Git** (opcional, para clonar el repositorio)

### **Sistema Operativo:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, Debian 11+, etc.)

---

## Instalación

### **Paso 1: Clonar el Repositorio**

```bash
git clone https://github.com/tu-usuario/ods-classifier.git
cd ods-classifier
```

### **Paso 2: Configurar el Backend (API)**

```bash
cd api

# Instalar dependencias de Node.js
npm install

# Instalar dependencias de Python
pip install joblib scikit-learn numpy pandas

# O usando requirements.txt
pip install -r requirements.txt
```

**Verificar instalación de Python:**
```bash
python --version   # Windows
python3 --version  # Linux/Mac
```

### **Paso 3: Verificar el Modelo**

Asegúrate de que existe el modelo entrenado:

```bash
# Debe existir:
api/src/ml_models/modelo_reentrenado.joblib
```

Si no existe, contacta al equipo para obtener el modelo pre-entrenado.

**Probar el modelo:**
```bash
cd src/python_scripts
python test_model.py

# Salida esperada:
# ✅ Modelo funciona correctamente!
```

### **Paso 4: Configurar el Frontend (Web)**

```bash
cd ../../web

# Instalar dependencias
npm install
```

---

## ▶️ Ejecución

### **Opción A: Modo Desarrollo (Recomendado para desarrollo)**

**Terminal 1 - Backend:**
```bash
cd api
npm run start:dev

# Salida esperada:
#  API corriendo en http://localhost:3001
#  Documentación disponible en http://localhost:3001/api/docs
```

**Terminal 2 - Frontend:**
```bash
cd web
npm run dev

# Salida esperada:
# VITE ready in 523 ms
# ➜  Local:   http://localhost:3000/
```


## Uso de la API

### **Documentación Interactiva (Swagger)**

Abre en tu navegador: **http://localhost:3001/api/docs**

### **Endpoint 1: Predicción**

**POST** `/api/ml/predict`

**Solicitud:**
```json
{
  "instances": [
    {
      "textos": "Los programas de ayuda económica reducen la pobreza extrema"
    },
    {
      "textos": "Las campañas de vacunación mejoran la salud comunitaria"
    }
  ]
}
```

**Respuesta:**
```json
[
  {
    "predicted_label": 1,
    "probability": 0.94,
    "textos": "Los programas de ayuda económica reducen la pobreza extrema",
    "class_probabilities": {
      "ODS_1": 0.94,
      "ODS_3": 0.04,
      "ODS_4": 0.02
    }
  },
  {
    "predicted_label": 3,
    "probability": 0.91,
    "textos": "Las campañas de vacunación mejoran la salud comunitaria",
    "class_probabilities": {
      "ODS_1": 0.03,
      "ODS_3": 0.91,
      "ODS_4": 0.06
    }
  }
]
```

### **Endpoint 2: Re-entrenamiento**

**POST** `/api/ml/retrain`

**Solicitud:**
```json
{
  "instances": [
    {
      "textos": "Subsidios alimentarios para familias vulnerables",
      "labels": 1
    },
    {
      "textos": "Atención médica primaria en zonas rurales",
      "labels": 3
    },
    {
      "textos": "Infraestructura educativa en comunidades remotas",
      "labels": 4
    }
  ]
}
```

**Nota:** Se requieren **mínimo 5 instancias** para re-entrenar.

**Respuesta:**
```json
{
  "precision": 0.96,
  "recall": 0.95,
  "f1_score": 0.955,
  "precision_per_class": {
    "ODS_1": 0.94,
    "ODS_3": 0.97,
    "ODS_4": 0.96
  },
  "recall_per_class": {
    "ODS_1": 0.93,
    "ODS_3": 0.96,
    "ODS_4": 0.96
  },
  "f1_score_per_class": {
    "ODS_1": 0.935,
    "ODS_3": 0.965,
    "ODS_4": 0.960
  },
  "retrained_at": "2025-10-13T19:30:22.456Z",
  "model_version": "v_20251013_193022",
  "training_size": 1250
}
```

### **Ejemplo con cURL:**

```bash
# Predicción
curl -X POST http://localhost:3001/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"textos": "Educación digital transforma vidas"}]}'

# Re-entrenamiento
curl -X POST http://localhost:3001/api/ml/retrain \
  -H "Content-Type: application/json" \
  -d '{"instances": [
    {"textos": "Transferencias monetarias condicionadas", "labels": 1},
    {"textos": "Telemedicina para diagnóstico temprano", "labels": 3},
    {"textos": "Becas universitarias para jóvenes", "labels": 4},
    {"textos": "Programas de vivienda social", "labels": 1},
    {"textos": "Campañas de prevención de enfermedades", "labels": 3}
  ]}'
```

---

## 🖥️ Aplicación Web

### **Acceso:**
Abre tu navegador en: **http://localhost:3000**

### **Funcionalidades:**

#### **1. Pestaña "Análisis de Texto" (Usuarios Generales)**

- Ingresa una o varias opiniones ciudadanas
- Haz clic en **"Clasificar Texto"**
- Visualiza:
  - ODS predicho (1, 3 o 4)
  - Nivel de confianza (porcentaje)
  - Distribución de probabilidades por clase

#### **2. Pestaña "Re-entrenamiento (Experto)" (Usuarios Avanzados)**

- Agrega múltiples instancias con sus etiquetas
- Mínimo 5 instancias requeridas
- Haz clic en **"Re-entrenar Modelo"**
- Visualiza métricas del nuevo modelo:
  - Precision, Recall, F1-Score (global y por clase)
  - Versión del modelo
  - Fecha de re-entrenamiento

---

## 🔄 Re-entrenamiento del Modelo

### **¿Qué es el Re-entrenamiento?**

El re-entrenamiento permite **mejorar continuamente** el modelo con nuevos datos validados por expertos.

### **¿Cómo Funciona?**

1. **Acumulación de Datos:**
   - El sistema guarda TODOS los datos históricos
   - Nuevos datos se agregan al histórico
   - Ejemplo: 1,000 (históricos) + 50 (nuevos) = 1,050 totales

2. **Entrenamiento desde Cero:**
   - Se entrena un NUEVO modelo con TODOS los datos
   - NO es incremental (evita "olvido catastrófico")
   - Garantiza consistencia

3. **Backup Automático:**
   - El modelo anterior se guarda como backup
   - Formato: `modelo_reentrenado_backup_YYYYMMDD_HHMMSS.joblib`
   - Se puede restaurar si es necesario

4. **Actualización en Producción:**
   - El nuevo modelo reemplaza al anterior automáticamente
   - Las siguientes predicciones usan el modelo mejorado

### **¿Cuándo Re-entrenar?**

✅ **Situaciones Recomendadas:**
- Detectas errores consistentes en las predicciones
- Hay un nuevo programa o iniciativa del UNFPA
- El lenguaje ciudadano evoluciona (post-pandemia, nuevas tecnologías)
- Quieres adaptar el modelo a una región específica
- Acumulas 50-100+ nuevas instancias etiquetadas

❌ **Evitar Re-entrenar:**
- Con menos de 5 instancias nuevas (mínimo requerido)
- Con datos mal etiquetados o de baja calidad
- Cada vez que hay un solo error (espera acumular varios casos)

### **Mejores Prácticas:**

1. **Validación Manual:** Antes de re-entrenar, valida que las etiquetas sean correctas
2. **Balance de Clases:** Intenta tener cantidades similares de ODS 1, 3 y 4
3. **Monitoreo de Métricas:** Verifica que el F1-Score no disminuya significativamente
4. **Documentación:** Anota por qué re-entrenaste y qué esperabas mejorar

---

## 📂 Estructura del Proyecto

```
ods-classifier/
│
├── api/                              # Backend NestJS
│   ├── src/
│   │   ├── controllers/
│   │   │   └── ml.controller.ts      # Endpoints /predict y /retrain
│   │   ├── services/
│   │   │   ├── ml.service.ts         # Lógica de negocio ML
│   │   │   ├── pipeline.service.ts   # Preprocesamiento de texto
│   │   │   └── python-bridge.service.ts  # Comunicación con Python
│   │   ├── dto/
│   │   │   ├── prediction.dto.ts     # DTOs de predicción
│   │   │   └── retrain.dto.ts        # DTOs de re-entrenamiento
│   │   ├── python_scripts/
│   │   │   ├── predict.py            # Script de predicción
│   │   │   ├── retrain.py            # Script de re-entrenamiento
│   │   │   ├── test_model.py         # Pruebas del modelo
│   │   │   └── compat_funcs.py       # Funciones auxiliares
│   │   ├── ml_models/
│   │   │   ├── modelo_reentrenado.joblib      # Modelo en producción
│   │   │   ├── training_data.joblib           # Datos históricos acumulados
│   │   │   └── modelo_reentrenado_backup_*.joblib  # Backups
│   │   ├── app.module.ts
│   │   └── main.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── requirements.txt              # Dependencias Python
│   └── README.md
│
├── web/                              # Frontend React
│   ├── src/
│   │   ├── App.jsx                   # Componente principal
│   │   ├── main.jsx                  # Entry point
│   │   └── index.css                 # Estilos Tailwind
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── README.md
│
└── README.md                         # Este archivo
```

---

## 🐛 Solución de Problemas

### **Problema: "Python: can't open file 'predict.py'"**

**Causa:** Los scripts Python no están en la ubicación correcta.

**Solución:**
```bash
# Verifica que existan:
ls api/src/python_scripts/predict.py
ls api/src/python_scripts/retrain.py

# Si no existen, cópialos a la ubicación correcta
```

---

### **Problema: "ModuleNotFoundError: No module named 'joblib'"**

**Causa:** Falta instalar dependencias de Python.

**Solución:**
```bash
pip install joblib scikit-learn numpy pandas
```

---

### **Problema: "Error: predict.py tarda más de 4 minutos"**

**Causa:** El script está esperando input de STDIN que nunca llega.

**Solución:** Ya está corregido en la versión actual. Verifica que `predict.py` lea de `sys.argv[1]` primero.

---

### **Problema: "CORS Error" en el navegador**

**Causa:** El frontend no puede conectar con la API.

**Solución:**
1. Verifica que la API esté corriendo en `http://localhost:3001`
2. Verifica que el frontend esté corriendo en `http://localhost:3000`
3. Si usas otros puertos, actualiza `vite.config.js`

---

### **Problema: "Modelo no encontrado"**

**Causa:** Falta el archivo del modelo.

**Solución:**
```bash
# Verifica que exista:
ls api/src/ml_models/modelo_reentrenado.joblib

# Si no existe, contacta al equipo para obtenerlo
```

---

### **Problema: Métricas bajas después de re-entrenamiento**

**Causas Comunes:**
- Pocos datos (menos de 20 instancias por clase)
- Datos desbalanceados
- Etiquetas incorrectas

**Solución:**
1. Revisa los logs del re-entrenamiento para ver distribución de clases
2. Asegúrate de tener datos balanceados (30-40% por cada ODS)
3. Valida que las etiquetas sean correctas

---

## 📖 Documentación Técnica

### **API REST:**
- Swagger UI: http://localhost:3001/api/docs
- Base URL: http://localhost:3001/api/ml

### **Tecnologías:**
- [NestJS](https://docs.nestjs.com/) - Framework backend
- [React](https://react.dev/) - Framework frontend
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Tailwind CSS](https://tailwindcss.com/) - Estilos

### **Algoritmo:**
- **Modelo:** LinearSVC (Support Vector Machine lineal)
- **Vectorización:** TF-IDF con n-grams (1,2)
- **Preprocesamiento:** Stemming (SnowballStemmer) + Stopwords en español
- **Clases:** 3 (ODS 1, 3, 4)

### **Métricas de Desempeño:**
- **Precision:** > 95%
- **Recall:** > 95%
- **F1-Score:** > 95%

---

## 👥 Contribuir

Este es un proyecto académico desarrollado para el UNFPA. Para contribuir:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Añadir nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico para la Universidad de los Andes.

---

## 📞 Soporte

Para problemas técnicos o preguntas:
- Revisa la sección [Solución de Problemas](#-solución-de-problemas)
- Consulta los logs en la terminal
- Verifica la documentación Swagger: http://localhost:3001/api/docs

---

## 🎓 Créditos

**Proyecto Académico - Universidad de los Andes**
- Curso: Business Intelligence
- Semestre: 8vo (2025)
- Cliente: Fondo de Población de las Naciones Unidas (UNFPA)

---

## ✅ Checklist de Instalación

Antes de usar en producción, verifica:

- [ ] Node.js 18+ instalado
- [ ] Python 3.8+ instalado
- [ ] Dependencias de npm instaladas (`npm install` en api/ y web/)
- [ ] Dependencias de Python instaladas (`pip install -r requirements.txt`)
- [ ] Modelo `modelo_reentrenado.joblib` existe en `api/src/ml_models/`
- [ ] Script `test_model.py` pasa exitosamente
- [ ] API responde en http://localhost:3001
- [ ] Frontend carga en http://localhost:3000
- [ ] Endpoint `/predict` funciona con cURL
- [ ] Swagger UI accesible en http://localhost:3001/api/docs

---

**¡Listo para clasificar opiniones y mejorar el impacto de las políticas públicas! 🌍✨**