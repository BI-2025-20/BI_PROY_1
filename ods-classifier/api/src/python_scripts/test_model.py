# ============================================
# python_scripts/test_model.py
# ============================================
import joblib
import sys
from pathlib import Path

from compact_func import clean_text, stemmed_tokenizer

# Si tu modelo fue entrenado con clean_text / tokenizer personalizados:

sys.modules['__main__'] = sys.modules.get('__main__', sys)
setattr(sys.modules['__main__'], 'clean_text', clean_text)
setattr(sys.modules['__main__'], 'stemmed_tokenizer', stemmed_tokenizer)

def test_model():
    print("\n" + "="*60)
    print("PRUEBA DE MODELO EXISTENTE")
    print("="*60 + "\n")

    model_dir = Path(__file__).parent.parent / 'ml_models'
    model_path = model_dir / 'modelo_reentrenado.joblib'

    print("📁 Verificando archivos...")
    if not model_path.exists():
        print(f"❌ ERROR: No se encuentra el modelo en: {model_path}")
        return False
    print(f"✓ Modelo encontrado: {model_path}")

    print("\n🤖 Cargando modelo (Pipeline completo)...")
    try:
        modelo = joblib.load(model_path)   # <- ES UN PIPELINE (vector + clf)
        print(f"✓ Modelo cargado: {type(modelo).__name__}")
        print(f"✓ Clases del modelo: {getattr(modelo, 'classes_', 'n/a')}")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return False

    print("\n🧪 Probando con textos de ejemplo...")
    textos_prueba = [
        "Programas de ayuda económica para familias vulnerables reducen la pobreza",
        "Vacunación infantil y atención médica primaria mejoran la salud comunitaria",
        "Educación digital y acceso a tecnología fortalecen el aprendizaje"
    ]

    try:
        # 👇 DIRECTO AL PIPELINE (no vectorices afuera)
        predicciones = modelo.predict(textos_prueba)

        print("\n📋 Resultados:")
        for texto, pred in zip(textos_prueba, predicciones):
            print(f"\n  Texto: {texto[:60]}...")
            print(f"  → Predicción: ODS {pred}")

        print("\n✅ ¡Modelo funciona correctamente!")
        return True

    except Exception as e:
        print(f"❌ Error al hacer predicciones: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)

