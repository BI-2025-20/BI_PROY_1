# ============================================
# python_scripts/predict.py
# Predicciones robustas con modelo existente
# ============================================
import sys, json, joblib, numpy as np
from pathlib import Path

# ---- Shims opcionales si el pickle apunta a __main__.clean_text / stemmed_tokenizer
try:
    from compact_func import clean_text, stemmed_tokenizer  # si existen
    sys.modules['__main__'] = sys.modules.get('__main__', sys)
    setattr(sys.modules['__main__'], 'clean_text', clean_text)
    setattr(sys.modules['__main__'], 'stemmed_tokenizer', stemmed_tokenizer)
except Exception:
    pass

def _find_path(rel):
    """Busca primero en dist/, luego en src/."""
    base = Path(__file__).resolve()
    # __file__ = .../api/dist/python_scripts/predict.py  (build)
    # o            .../api/src/python_scripts/predict.py (dev)
    dist = base.parent.parent / rel  # ../.. -> dist/*
    src  = base.parent.parent / "src" / rel  # ../.. -> src/*
    return dist if dist.exists() else src

def _load_artifacts():
    model_path = _find_path(Path("ml_models") / "modelo_reentrenado.joblib")
    vect_path  = _find_path(Path("ml_models") / "vectorizer.joblib")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    modelo = joblib.load(model_path)

    # ¿Es pipeline?
    is_pipeline = hasattr(modelo, "named_steps") or "Pipeline" in modelo.__class__.__name__

    vectorizer = None
    if not is_pipeline:
        # El modelo es solo clasificador: necesitamos el vectorizador aparte
        if not vect_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vect_path}")
        vectorizer = joblib.load(vect_path)

    return modelo, vectorizer, is_pipeline

def _as_probs_from_decision(scores: np.ndarray) -> np.ndarray:
    # softmax por filas
    exp_x = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def _read_texts():
    # 1) argv[1] como JSON (PRIMERO - más rápido)
    if len(sys.argv) > 1:
        try:
            texts = json.loads(sys.argv[1])
            if isinstance(texts, str):
                texts = [texts]
            return texts
        except Exception:
            pass
    
    # 2) STDIN JSON: {"instances": ["t1","t2",...]} (SEGUNDO - solo como fallback)
    try:
        payload = sys.stdin.read()
        if payload and payload.strip():
            data = json.loads(payload)
            texts = data.get("instances")
            if isinstance(texts, str):
                texts = [texts]
            if texts:
                return texts
    except Exception:
        pass
    
    return []

def main():
    try:
        texts = _read_texts()
        if not texts:
            print(json.dumps({"ok": False, "error": "No instances provided"}))
            sys.exit(1)

        modelo, vectorizer, is_pipeline = _load_artifacts()

        # Vectorizar según caso
        if is_pipeline:
            # Pipeline espera strings crudos
            X_input = texts
        else:
            # Clasificador puro: transformamos con vectorizer
            X_input = vectorizer.transform(texts)

        # Predicciones
        try:
            preds = modelo.predict(X_input)
        except Exception:
            # Si dimos matriz a un pipeline por error, forzamos strings
            preds = modelo.predict(texts)

        # Probabilidades
        try:
            probs = modelo.predict_proba(X_input)  # p.ej. si es CalibratedClassifierCV
        except Exception:
            # Decision function -> softmax
            try:
                scores = modelo.decision_function(X_input)
            except Exception:
                # En último caso, prob = 1 para la clase predicha
                classes = getattr(modelo, "classes_", [])
                onehot = np.zeros((len(texts), len(classes)), dtype=float)
                for i, p in enumerate(preds):
                    j = list(classes).index(p)
                    onehot[i, j] = 1.0
                probs = onehot
            else:
                # Si scores es 1D (binario), conviértelo a 2D
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                probs = _as_probs_from_decision(scores)

        classes = getattr(modelo, "classes_", None)
        if classes is None:
            # intenta sacar clases desde shape de probs
            classes = np.arange(probs.shape[1])

        results = []
        for t, p, pr in zip(texts, preds, probs):
            # índice de la clase predicha
            try:
                idx = list(classes).index(p)
            except ValueError:
                idx = int(np.argmax(pr))
            pred_prob = float(pr[idx])

            class_probs = {f"ODS_{int(cls)}": float(prob) for cls, prob in zip(classes, pr)}

            results.append({
                "label": int(p),
                "probability": pred_prob,
                "class_probabilities": class_probs,
                "text": t
            })

        print(json.dumps({"ok": True, "result": results}))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
