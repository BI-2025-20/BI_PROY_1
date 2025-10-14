# ============================================
# python_scripts/retrain.py
# Script para re-entrenar el modelo existente
# ============================================

import sys
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

def load_existing_model_and_data():
    """
    Carga el modelo existente y datos históricos si existen
    """
    model_path = Path(__file__).parent.parent / 'ml_models' / 'modelo_reentrenado.joblib'
    data_path = Path(__file__).parent.parent / 'ml_models' / 'training_data.joblib'
    
    modelo_existente = None
    vectorizer_existente = None
    historical_texts = []
    historical_labels = []
    
    # Cargar modelo existente
    if model_path.exists():
        try:
            modelo_existente = joblib.load(model_path)
            print(f"✓ Modelo existente cargado desde: {model_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ No se pudo cargar modelo existente: {e}", file=sys.stderr)
    
    # Cargar vectorizador existente
    vectorizer_path = Path(__file__).parent.parent / 'ml_models' / 'vectorizer.joblib'
    if vectorizer_path.exists():
        try:
            vectorizer_existente = joblib.load(vectorizer_path)
            print(f"✓ Vectorizador existente cargado", file=sys.stderr)
        except Exception as e:
            print(f"⚠ No se pudo cargar vectorizador existente: {e}", file=sys.stderr)
    
    # Cargar datos históricos
    if data_path.exists():
        try:
            data = joblib.load(data_path)
            historical_texts = data.get('texts', [])
            historical_labels = data.get('labels', [])
            print(f"✓ Datos históricos cargados: {len(historical_texts)} instancias", file=sys.stderr)
        except Exception as e:
            print(f"⚠ No se pudieron cargar datos históricos: {e}", file=sys.stderr)
    
    return modelo_existente, vectorizer_existente, historical_texts, historical_labels

def save_training_data(texts, labels):
    """Guarda los datos de entrenamiento para futuras iteraciones"""
    data_path = Path(__file__).parent.parent / 'ml_models' / 'training_data.joblib'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        joblib.dump({'texts': texts, 'labels': labels}, data_path)
        print(f"✓ Datos de entrenamiento guardados: {len(texts)} instancias", file=sys.stderr)
    except Exception as e:
        print(f"⚠ Error al guardar datos de entrenamiento: {e}", file=sys.stderr)

def train_model(texts, labels):
    """
    Re-entrena el modelo con los datos proporcionados
    Implementa re-entrenamiento incremental con histórico acumulativo
    
    Args:
        texts: Lista de strings con textos preprocesados
        labels: Lista de enteros con las etiquetas (1, 3, 4)
    
    Returns:
        Tupla (model, vectorizer, metrics)
    """
    try:
        # Cargar modelo y datos existentes
        modelo_existente, vectorizer_existente, historical_texts, historical_labels = load_existing_model_and_data()
        
        # Combinar con nuevos datos
        all_texts = historical_texts + texts
        all_labels = historical_labels + labels
        
        print("\n" + "="*60, file=sys.stderr)
        print("INFORMACIÓN DEL RE-ENTRENAMIENTO", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"📊 Total de datos acumulados: {len(all_texts)} instancias", file=sys.stderr)
        print(f"   - Datos históricos: {len(historical_texts)}", file=sys.stderr)
        print(f"   - Datos nuevos: {len(texts)}", file=sys.stderr)
        
        # Distribución de clases
        from collections import Counter
        class_distribution = Counter(all_labels)
        print(f"\n📈 Distribución de clases:", file=sys.stderr)
        for cls, count in sorted(class_distribution.items()):
            percentage = (count / len(all_labels)) * 100
            print(f"   - ODS {cls}: {count} instancias ({percentage:.1f}%)", file=sys.stderr)
        
        # Guardar datos actualizados
        save_training_data(all_texts, all_labels)
        
        # Validar cantidad mínima de datos
        if len(all_texts) < 20:
            print(f"\n⚠ ADVERTENCIA: Se recomienda tener al menos 20 instancias para re-entrenar", file=sys.stderr)
            print(f"   Actualmente tienes {len(all_texts)} instancias", file=sys.stderr)
        
        # Split en entrenamiento y validación
        # Si hay pocas instancias, usar un split más conservador
        test_size = 0.2 if len(all_texts) >= 50 else 0.15
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                all_texts, all_labels, 
                test_size=test_size, 
                random_state=42,
                stratify=all_labels
            )
            print(f"\n✓ Split realizado: {len(X_train)} train, {len(X_val)} validation", file=sys.stderr)
        except ValueError as e:
            # Si falla el stratify (muy pocos datos por clase), intentar sin stratify
            print(f"⚠ No se pudo hacer split estratificado: {e}", file=sys.stderr)
            X_train, X_val, y_train, y_val = train_test_split(
                all_texts, all_labels, 
                test_size=test_size, 
                random_state=42
            )
            print(f"✓ Split realizado sin estratificación", file=sys.stderr)
        
        # Decidir si reusar el vectorizador existente o crear uno nuevo
        if vectorizer_existente is not None and len(historical_texts) > 100:
            print("\n🔄 Reusando vectorizador existente (modo incremental)", file=sys.stderr)
            vectorizer = vectorizer_existente
            X_train_vec = vectorizer.transform(X_train)
            X_val_vec = vectorizer.transform(X_val)
        else:
            print("\n🆕 Creando nuevo vectorizador TF-IDF", file=sys.stderr)
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec = vectorizer.transform(X_val)
        
        print(f"✓ Vectorización completada. Dimensión: {X_train_vec.shape}", file=sys.stderr)
        
        # Entrenar nuevo modelo LinearSVC
        print("\n🤖 Entrenando modelo LinearSVC...", file=sys.stderr)
        modelo = LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            dual='auto'  # Automático según tamaño de datos
        )
        modelo.fit(X_train_vec, y_train)
        print("✓ Modelo entrenado exitosamente", file=sys.stderr)
        
        # Predicciones en conjunto de validación
        y_pred = modelo.predict(X_val_vec)
        
        # Calcular métricas
        print("\n📊 CALCULANDO MÉTRICAS DE DESEMPEÑO...", file=sys.stderr)
        
        # Métricas macro (promedio entre clases)
        precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        
        print(f"\n🎯 Métricas Globales:", file=sys.stderr)
        print(f"   - Precision: {precision_macro:.4f} ({precision_macro*100:.2f}%)", file=sys.stderr)
        print(f"   - Recall:    {recall_macro:.4f} ({recall_macro*100:.2f}%)", file=sys.stderr)
        print(f"   - F1-Score:  {f1_macro:.4f} ({f1_macro*100:.2f}%)", file=sys.stderr)
        
        # Métricas por clase
        precision_per_class = precision_score(
            y_val, y_pred, 
            average=None, 
            labels=[1, 3, 4],
            zero_division=0
        )
        recall_per_class = recall_score(
            y_val, y_pred, 
            average=None, 
            labels=[1, 3, 4],
            zero_division=0
        )
        f1_per_class = f1_score(
            y_val, y_pred, 
            average=None, 
            labels=[1, 3, 4],
            zero_division=0
        )
        
        # Reporte de clasificación completo
        print("\n" + "="*60, file=sys.stderr)
        print("REPORTE DE CLASIFICACIÓN DETALLADO", file=sys.stderr)
        print("="*60, file=sys.stderr)
        report = classification_report(y_val, y_pred, labels=[1, 3, 4], target_names=['ODS 1', 'ODS 3', 'ODS 4'])
        print(report, file=sys.stderr)
        
        metrics = {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro),
            'precision_per_class': {
                'ODS_1': float(precision_per_class[0]),
                'ODS_3': float(precision_per_class[1]),
                'ODS_4': float(precision_per_class[2])
            },
            'recall_per_class': {
                'ODS_1': float(recall_per_class[0]),
                'ODS_3': float(recall_per_class[1]),
                'ODS_4': float(recall_per_class[2])
            },
            'f1_score_per_class': {
                'ODS_1': float(f1_per_class[0]),
                'ODS_3': float(f1_per_class[1]),
                'ODS_4': float(f1_per_class[2])
            },
            'model_version': f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_samples': len(all_texts),
            'validation_samples': len(X_val)
        }
        
        return modelo, vectorizer, metrics
    
    except Exception as e:
        print(f"\n❌ ERROR en entrenamiento: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

def save_model(modelo, vectorizer):
    """Guarda el modelo y vectorizador re-entrenados en formato joblib"""
    model_dir = Path(__file__).parent.parent / 'ml_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'modelo_reentrenado.joblib'
    vectorizer_path = model_dir / 'vectorizer.joblib'
    
    # Hacer backup del modelo anterior
    if model_path.exists():
        backup_path = model_dir / f'modelo_reentrenado_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        try:
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"\n💾 Backup del modelo anterior: {backup_path.name}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ No se pudo crear backup: {e}", file=sys.stderr)
    
    # Guardar nuevo modelo con joblib
    try:
        joblib.dump(modelo, model_path, compress=3)
        print(f"✓ Modelo guardado: {model_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error al guardar modelo: {e}", file=sys.stderr)
        raise
    
    try:
        joblib.dump(vectorizer, vectorizer_path, compress=3)
        print(f"✓ Vectorizador guardado: {vectorizer_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error al guardar vectorizador: {e}", file=sys.stderr)
        raise
    
    print(f"\n✅ Re-entrenamiento completado exitosamente!", file=sys.stderr)

if __name__ == '__main__':
    try:
        # Leer datos desde argumentos
        if len(sys.argv) < 2:
            print("Error: No se proporcionaron datos", file=sys.stderr)
            sys.exit(1)
        
        data_json = sys.argv[1]
        data = json.loads(data_json)
        
        texts = data['texts']
        labels = data['labels']
        
        # Validaciones
        if len(texts) != len(labels):
            raise ValueError("La cantidad de textos y labels debe ser igual")
        
        if len(texts) < 5:
            raise ValueError("Se requieren al menos 5 instancias para re-entrenar")
        
        # Validar que las etiquetas sean correctas
        valid_labels = {1, 3, 4}
        invalid = [l for l in labels if l not in valid_labels]
        if invalid:
            raise ValueError(f"Labels inválidos encontrados: {invalid}. Solo se permiten 1, 3, 4")
        
        print("\n" + "🚀"*30, file=sys.stderr)
        print("INICIANDO PROCESO DE RE-ENTRENAMIENTO", file=sys.stderr)
        print("🚀"*30 + "\n", file=sys.stderr)
        
        # Entrenar modelo
        modelo, vectorizer, metrics = train_model(texts, labels)
        
        # Guardar modelo
        save_model(modelo, vectorizer)
        
        print("\n" + "✨"*30, file=sys.stderr)
        print("PROCESO COMPLETADO EXITOSAMENTE", file=sys.stderr)
        print("✨"*30 + "\n", file=sys.stderr)
        
        # Retornar métricas como JSON
        print(json.dumps(metrics))
        
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
